"""
End-to-end interpretable examples of CTC in action.

Picks real test questions, shows what each clean and corrupt agent
produces, and walks through how CTC arrives at its decision.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 -m ctc_llm.paper.end_to_end_example \
        --n-examples 5 --task mmlu --output results/examples.md
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys
from typing import List, Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ctc_llm.tasks.mmlu import load_mmlu, load_truthfulqa, load_arc, Question
from ctc_llm.agents.local_llm_agent import LocalLLMAgent
from ctc_llm.agents.corrupt_agent import make_corrupt_agent
from ctc_llm.conformal.calibrate import calibrate, conformal_set
from ctc_llm.coordination.ctc           import (
    ctc_answer, ctc_focal_answer, ctc_hybrid_answer
)
from ctc_llm.coordination.confidence    import committee_conformal_abstain
from ctc_llm.coordination.majority      import majority_answer
from ctc_llm.coordination.entropy_trust import entropy_trust_answer
from ctc_llm.experiments.runner         import build_q_hat


def _bar(p: float, width: int = 20) -> str:
    n = int(round(p * width))
    return "█" * n + "░" * (width - n)


def _format_dist(probs: np.ndarray, choices: List[str], correct: int,
                 conf_set: List[int] = None) -> str:
    lines = []
    for i, (p, c) in enumerate(zip(probs, choices)):
        letter = "ABCD"[i]
        mark   = "✓" if i == correct else " "
        in_set = " [in C]" if conf_set is not None and i in conf_set else ""
        c_short = c[:55] + "..." if len(c) > 55 else c
        lines.append(f"      {letter}{mark} {_bar(float(p))} {p:.3f}  {c_short}{in_set}")
    return "\n".join(lines)


def walk_question(q: Question, clean_agents: List, q_hat: float, k_corrupt: int,
                  attack: str, seed: int, model_id: str, cache_dir: str) -> str:
    """Generate a Markdown explanation of a single question's CTC walk."""
    n = len(clean_agents)
    rng = random.Random(seed * 100_000)
    corrupt_ids = set(rng.sample(range(n), k=min(k_corrupt, n)))

    out = []
    out.append(f"## Question (subject: {q.subject})\n")
    out.append(f"> **{q.question}**\n")
    out.append("**Choices**")
    for i, c in enumerate(q.choices):
        marker = " ← *correct*" if i == q.correct else ""
        out.append(f"  - **{'ABCD'[i]}**: {c}{marker}")
    out.append("")
    out.append(f"**Setup**: N={n} agents, k={k_corrupt} corrupt under `{attack}` attack, "
               f"shared q̂={q_hat:.4f}\n")

    # Each agent's distribution and conformal set
    out.append("### Step 1 — Each agent emits probabilities + conformal set\n")
    agent_probs: Dict[int, np.ndarray] = {}
    trust: Dict[int, float] = {}
    conf_sets: Dict[int, List[int]] = {}
    for i, agent in enumerate(clean_agents):
        role = "🟥 CORRUPT" if i in corrupt_ids else "🟩 clean"
        out.append(f"**Agent {i} ({role})**")
        if i in corrupt_ids:
            corrupt = make_corrupt_agent(attack, agent, correct_idx=q.correct,
                                          agent_id=i, model_id=model_id,
                                          result_cache_dir=cache_dir)
            probs = corrupt.get_probs(q.question, q.choices)
        else:
            probs = agent.get_probs(q.question, q.choices)
        agent_probs[i] = probs
        cs = conformal_set(probs, q_hat)
        conf_sets[i] = cs
        trust[i] = 1.0 / len(cs)
        out.append("```")
        out.append(_format_dist(probs, q.choices, q.correct, conf_set=cs))
        out.append("```")
        cs_str = "{" + ", ".join("ABCD"[a] for a in cs) + "}"
        out.append(f"  Conformal set: {cs_str}  |  Trust T = 1/|C| = {trust[i]:.3f}\n")

    # Committee union
    union = set()
    for cs in conf_sets.values():
        union.update(cs)
    union_str = "{" + ", ".join("ABCD"[a] for a in sorted(union)) + "}"
    out.append("### Step 2 — Committee union (the conformal abstention signal)\n")
    out.append(f"⋃ᵢ Cᵢ = {union_str}  →  size **{len(union)}**\n")
    if len(union) == 1:
        out.append("✅ Committee agrees on a singleton → **commit to prediction**\n")
    else:
        out.append("⚠️  Committee disagrees → low committee confidence → "
                   "**abstention candidate**\n")

    # CTC-Global score
    out.append("### Step 3 — CTC-Global trust-weighted score per action\n")
    out.append("`score(a) = Σᵢ πᵢ(a) · Tᵢ · 1[a ∈ Cᵢ]`\n")
    scores = np.zeros(4, dtype=np.float64)
    breakdown_lines = []
    for a in range(4):
        contribs = []
        for j, p in agent_probs.items():
            if a in conf_sets[j]:
                contrib = float(p[a]) * trust[j]
                scores[a] += contrib
                contribs.append(f"A{j}:{float(p[a]):.2f}×{trust[j]:.2f}")
        breakdown = " + ".join(contribs) if contribs else "—"
        marker = "  ← argmax" if a == int(np.argmax(scores[:a+1])) and a == int(np.argmax(scores)) else ""
        breakdown_lines.append(f"  {'ABCD'[a]}: {scores[a]:.3f}  =  {breakdown}{marker}")
    out.append("```")
    out.extend(breakdown_lines)
    out.append("```")
    out.append("")

    # Compare methods
    out.append("### Step 4 — Method comparison\n")
    out.append("| Method | Answer | Correct? |")
    out.append("|---|---|---|")
    methods = [
        ("Majority Vote", majority_answer(agent_probs)),
        ("EntropyTrust",  entropy_trust_answer(agent_probs)),
        ("CTC-Focal",     ctc_focal_answer(agent_probs, q_hat)),
        ("CTC-Global",    ctc_answer(agent_probs, q_hat)),
        ("CTC-Hybrid",    ctc_hybrid_answer(agent_probs, q_hat)),
    ]
    cmt_ans, cmt_size, cmt_conc = committee_conformal_abstain(agent_probs, q_hat)
    methods.append(("Committee Conformal (selective)",
                    f"{'ABCD'[cmt_ans]} (|⋃C|={cmt_size}, conc={cmt_conc:.2f})"))
    for name, ans in methods:
        if isinstance(ans, str):
            mark = "✓" if ans.startswith('ABCD'[q.correct]) else "✗"
            out.append(f"| {name} | {ans} | {mark} |")
        else:
            mark = "✓" if ans == q.correct else "✗"
            out.append(f"| {name} | {'ABCD'[ans]} | {mark} |")
    out.append("")
    out.append("---\n")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task",         default="mmlu",
                   choices=["mmlu", "truthfulqa", "arc"])
    p.add_argument("--n-examples",   type=int, default=5)
    p.add_argument("--n-agents",     type=int, default=5)
    p.add_argument("--k-corrupt",    type=int, default=3)
    p.add_argument("--attack",       default="overconfident_extreme")
    p.add_argument("--alpha",        type=float, default=0.10)
    p.add_argument("--model-id",     default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--cache-dir",    default="data/cache")
    p.add_argument("--hf-cache-dir", default="/data/models")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--output",       default="results/examples.md")
    args = p.parse_args()

    # Load task
    if args.task == "mmlu":
        questions = load_mmlu()
    elif args.task == "truthfulqa":
        questions = load_truthfulqa(max_questions=600)
    else:
        questions = load_arc(max_questions=600)

    rng = random.Random(args.seed)
    rng.shuffle(questions)

    # Calibrate
    cal_q = questions[:max(50, len(questions) // 3)]
    test_q = questions[len(cal_q):]
    agents = [LocalLLMAgent(i, model_id=args.model_id,
                             hf_cache_dir=args.hf_cache_dir,
                             result_cache_dir=args.cache_dir)
              for i in range(args.n_agents)]
    q_hat = build_q_hat(agents, cal_q, alpha=args.alpha)
    print(f"q̂ = {q_hat:.4f}  (pooled over {len(cal_q)} cal questions × {args.n_agents} agents)")

    # Pick a mix: interesting cases (mistakes by Majority, saved by CTC)
    picks = []
    for q in test_q:
        if len(picks) >= args.n_examples * 5:
            break
        # Quick check whether this is interesting
        try:
            probs = {i: a.get_probs(q.question, q.choices) for i, a in enumerate(agents)}
        except Exception:
            continue
        picks.append((q, probs))

    rng2 = random.Random(args.seed + 1)
    selected = rng2.sample(picks, min(args.n_examples, len(picks)))

    lines = []
    lines.append(f"# End-to-End CTC Examples")
    lines.append(f"")
    lines.append(f"- Task: **{args.task.upper()}**")
    lines.append(f"- Model: `{args.model_id}`")
    lines.append(f"- N = {args.n_agents}, k = {args.k_corrupt} corrupt, "
                 f"α = {args.alpha}")
    lines.append(f"- Attack: `{args.attack}`")
    lines.append(f"- Shared conformal threshold: q̂ = {q_hat:.4f}")
    lines.append(f"")

    for i, (q, _) in enumerate(selected):
        print(f"  Walking example {i+1}: {q.question[:50]}…")
        block = walk_question(q, agents, q_hat,
                              k_corrupt=args.k_corrupt,
                              attack=args.attack,
                              seed=args.seed + i,
                              model_id=args.model_id,
                              cache_dir=args.cache_dir)
        lines.append(f"# Example {i+1}\n")
        lines.append(block)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
