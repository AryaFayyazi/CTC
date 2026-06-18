"""
Backfill ctc_robust + ctc_adaptive onto existing MCQ records by replaying
coordination on cached agent probs. No LLM inference.

Faithfulness: reproduces the exact cal/test split, agents, corruption seeds,
and stored q_hat used by run_experiments.py. As a self-check it ALSO recomputes
ctc_hybrid per-question accuracy and asserts it matches the stored column, so we
know the replay is faithful before trusting the new columns.

Scope: experiment == "main" (overconfident_extreme, 5 standard agents/model).

Usage: python3 scripts_backfill_adaptive.py [results/raw_results.json]
"""
from __future__ import annotations
import json, os, sys, random, time
sys.path.insert(0, os.path.dirname(__file__))
from typing import Dict, List
import numpy as np

from ctc_llm.tasks.mmlu import load_mmlu, load_truthfulqa, load_arc, MMLU_SUBSET, Question
from ctc_llm.agents.local_llm_agent import LocalLLMAgent
from ctc_llm.agents.corrupt_agent import make_corrupt_agent
from ctc_llm.experiments.runner import build_per_agent_profile
from ctc_llm.coordination.ctc import (
    ctc_hybrid_answer, ctc_robust_answer, ctc_adaptive_answer,
)

HF, CACHE = "/data/models", "data/cache"
NEW = ["ctc_robust", "ctc_adaptive"]


def load_task(name):
    if name == "mmlu":       return load_mmlu(subjects=MMLU_SUBSET)
    if name == "truthfulqa": return load_truthfulqa(max_questions=600)
    if name == "arc":        return load_arc(max_questions=600)
    raise ValueError(name)


def split(qs, seed):
    rng = random.Random(seed); sh = qs[:]; rng.shuffle(sh)
    cal_n = max(50, len(sh) // 3)
    return sh[:cal_n], sh[cal_n:]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/raw_results.json"
    print(f"Loading {path} …")
    records = json.load(open(path))
    todo = [r for r in records if r.get("experiment") == "main"
            and any(f"{m}_accuracy" not in r for m in NEW)]
    print(f"  {len(records)} records, {len(todo)} main records to backfill")

    task_cache, agent_cache, prof_cache = {}, {}, {}
    t0 = time.time(); mism = 0
    for idx, r in enumerate(todo):
        task, model_id, k, attack, seed = (r["task"], r["model_id"],
            r["n_corrupt"], r["attack"], r["seed"])
        n_agents, q_hat = r["n_agents"], r["q_hat"]

        if task not in task_cache:
            task_cache[task] = load_task(task)
        cal_q, test_q = split(task_cache[task], seed)
        if not test_q:
            continue

        ak = (model_id, n_agents)
        if ak not in agent_cache:
            agent_cache[ak] = [LocalLLMAgent(i, model_id=model_id,
                hf_cache_dir=HF, result_cache_dir=CACHE) for i in range(n_agents)]
        agents = agent_cache[ak]

        pk = (model_id, task, seed)
        if pk not in prof_cache:
            prof_cache[pk] = build_per_agent_profile(agents, cal_q)
        profile = prof_cache[pk]
        cal_stats = {i: profile[i]["q"] for i in profile}

        rob, ada, hyb_chk = [], [], []
        for q_idx, q in enumerate(test_q):
            cseed = seed * 100_000 + q_idx
            rng = random.Random(cseed)
            cids = set(rng.sample(range(n_agents), k=min(k, n_agents)))
            ap = {}
            for i, ag in enumerate(agents):
                if i in cids:
                    cor = make_corrupt_agent(attack, ag, correct_idx=q.correct,
                        agent_id=i, model_id=model_id, hf_cache_dir=HF,
                        result_cache_dir=CACHE)
                    ap[i] = cor.get_probs(q.question, q.choices)
                else:
                    ap[i] = ag.get_probs(q.question, q.choices)
            rob.append(int(ctc_robust_answer(ap, q_hat, cal_stats) == q.correct))
            ada.append(int(ctc_adaptive_answer(ap, q_hat, profile) == q.correct))
            hyb_chk.append(int(ctc_hybrid_answer(ap, q_hat) == q.correct))

        # faithfulness self-check against stored ctc_hybrid column
        stored = r.get("ctc_hybrid_per_q")
        if stored and len(stored) == len(hyb_chk):
            if [int(x) for x in stored] != hyb_chk:
                mism += 1
                if mism <= 3:
                    a = np.mean([int(x) for x in stored]); b = np.mean(hyb_chk)
                    print(f"  ! mismatch {model_id.split('/')[-1]}/{task}/k{k}/s{seed}"
                          f"  stored_hybrid={a:.3f} recomputed={b:.3f}")

        r["ctc_robust_accuracy"]   = float(np.mean(rob))
        r["ctc_adaptive_accuracy"] = float(np.mean(ada))
        r["ctc_robust_per_q"]      = rob
        r["ctc_adaptive_per_q"]    = ada

        if (idx + 1) % 50 == 0 or idx + 1 == len(todo):
            el = time.time() - t0; rate = el / (idx + 1)
            print(f"  {idx+1}/{len(todo)}  ({el/60:.1f}m, ETA {rate*(len(todo)-idx-1)/60:.1f}m, "
                  f"mismatches={mism})", flush=True)
            tmp = path + ".tmp"; json.dump(records, open(tmp, "w")); os.replace(tmp, path)

    tmp = path + ".tmp"; json.dump(records, open(tmp, "w")); os.replace(tmp, path)
    print(f"Done. {len(todo)} records, {mism} hybrid-mismatches, {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
