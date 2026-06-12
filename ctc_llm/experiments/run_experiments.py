"""
Phase 2 — Multi-model coordination experiments (reads from cache).

Experimental design
-------------------
Tasks    : MMLU (16-subject subset, primary), TruthfulQA, ARC-Challenge
Models   : Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Phi-3.5-mini-instruct
Agents   : N=5 (primary); N=10 ablation for scaling
Corrupt  : k ∈ {0, 1, 2, 3} of 5  (also {0..5} for N=10)
Attacks  : overconfident, random, subtle, inject_sycophant, inject_deceptive
Alpha    : 0.05, 0.10, 0.20  (primary 0.10)
Seeds    : 20  (different cal/test splits + corrupt-id picks)

For each (model, task, k, attack, alpha, seed):
  - Pool-calibrate q̂ on calibration set
  - Run all 7 coordination methods on test set
  - Record per-question accuracy for cross-seed 95% CI + Wilcoxon

Output: results/raw_results.json (resumable — skips already-done conditions).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ctc_llm.tasks.mmlu import (
    load_mmlu, load_truthfulqa, load_arc, load_gpqa, MMLU_SUBSET
)
from ctc_llm.tasks.mmlu import Question
from ctc_llm.agents.local_llm_agent import LocalLLMAgent
from ctc_llm.experiments.runner import (
    build_q_hat, build_per_agent_q_hat, run_questions, METHODS
)


# ── Default configuration ────────────────────────────────────────────────────

N_AGENTS_MAIN = 5
N_AGENTS_BIG  = 10
N_CORRUPT_5   = [0, 1, 2, 3]
N_CORRUPT_10  = [0, 2, 4, 6]
ATTACK_MAIN   = "overconfident_extreme"   # stronger primary attack that breaks Entropy
ATTACKS_ALL   = ["overconfident", "overconfident_extreme", "coordinated_extreme",
                 "random", "subtle",
                 "inject_sycophant", "inject_deceptive"]
ALPHA_MAIN    = 0.10
ALPHAS        = [0.05, 0.10, 0.20]
MODELS_DEFAULT = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "allenai/Olmo-3-7B-Instruct",         # 5th model for heterogeneous committee
]

# Heterogeneous committee: each committee slot is a different model.
# We pin the persona to Alpha for all hetero agents so the only thing
# varying across the committee is the base model.
HETERO_COMMITTEE = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "allenai/Olmo-3-7B-Instruct",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _condition_key(r: Dict) -> tuple:
    return (r.get("model_id"), r.get("task"), r.get("n_agents"),
            r.get("n_corrupt"), r.get("attack"),
            round(float(r.get("alpha", 0.0)), 6),
            r.get("seed"), r.get("experiment", "main"))


def _add_or_replace(records: List[Dict], new_record: Dict) -> None:
    """Append new_record, or replace an existing record with the same condition."""
    k = _condition_key(new_record)
    for i, r in enumerate(records):
        if _condition_key(r) == k:
            records[i] = new_record
            return
    records.append(new_record)


def _load_existing(path: str) -> List[Dict]:
    """Load existing records and DEDUPLICATE by condition key.

    When duplicates exist (same condition, different schema), keep the one
    with the MOST fields (i.e. the newest, post-backfill version).
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return []
    if not (isinstance(data, list) and data):
        return []
    if not ("task" in data[0] and "model_id" in data[0]):
        print(f"Found incompatible records in {path} — starting fresh")
        return []

    seen: Dict[tuple, Dict] = {}
    for r in data:
        k = _condition_key(r)
        if k not in seen or len(r) > len(seen[k]):
            seen[k] = r
    out = list(seen.values())
    if len(out) < len(data):
        print(f"Resuming: {len(data)} → {len(out)} records after dedup in {path}")
    else:
        print(f"Resuming: {len(out)} records already in {path}")
    return out


def _is_done(existing, model_id, task, n_agents, k, attack, alpha, seed, experiment):
    """Match condition AND require all CORE methods are present.

    Core = anything except `self_consistency` and `debate`, which are
    cheap to backfill from cached probs (see scripts_backfill_baselines.py).
    This avoids wasting a 24h SLURM job re-running cells just to add
    two cheap aggregators.
    """
    CHEAP_BACKFILL = {"self_consistency", "debate",
                      "mixture_of_agents", "llm_judge"}
    core = [m for m in METHODS if m not in CHEAP_BACKFILL]
    required = {f"{m}_accuracy" for m in core}
    for r in existing:
        if (r.get("model_id") == model_id and
            r["task"] == task and
            r["n_agents"] == n_agents and
            r["n_corrupt"] == k and
            r["attack"] == attack and
            abs(r["alpha"] - alpha) < 1e-6 and
            r["seed"] == seed and
            r.get("experiment", "main") == experiment and
            required.issubset(r.keys())):
            return True
    return False


def _build_record(model_id, task, n_agents, k, attack, alpha, seed, q_hat,
                  res, experiment) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "model_id":    model_id,
        "task":        task,
        "n_agents":    n_agents,
        "n_corrupt":   k,
        "corrupt_frac": k / n_agents,
        "attack":      attack,
        "alpha":       alpha,
        "q_hat":       q_hat,
        "seed":        seed,
        "experiment":  experiment,
        "n_questions": res["n_questions"],
        "ctc_coverage":      res["ctc_coverage"],
        "ctc_mean_set_size": res["ctc_mean_set_size"],
        # Selective-prediction headline metric
        "committee_accuracy": res.get("committee_accuracy", 0.0),
    }
    for m in METHODS:
        record[f"{m}_accuracy"] = res[f"{m}_accuracy"]
        record[f"{m}_per_q"]    = res[f"{m}_per_q"]
    # Per-question confidences and committee abstention signal
    # (needed for risk-coverage curves)
    for ck in ["vanilla_conf", "average_conf", "majority_conf",
               "entropy_conf", "self_consistency_conf", "debate_conf",
               "mixture_of_agents_conf", "llm_judge_conf",
               "ctc_conf", "ctc_hybrid_conf",
               "committee_acc", "committee_set_size",
               "committee_score_concentration"]:
        key = f"{ck}_per_q"
        if key in res:
            record[key] = res[key]
    return record


def _save(results: List[Dict], path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f)
    os.replace(tmp, path)


# ── Task loaders ──────────────────────────────────────────────────────────────

def load_all_tasks(args) -> Dict[str, List[Question]]:
    out: Dict[str, List[Question]] = {}
    if "mmlu" in args.tasks:
        subjects = args.subjects or MMLU_SUBSET
        print(f"Loading MMLU ({len(subjects)} subjects)…")
        out["mmlu"] = load_mmlu(subjects=subjects)
        print(f"  {len(out['mmlu'])} questions")
    if "truthfulqa" in args.tasks:
        print("Loading TruthfulQA…")
        out["truthfulqa"] = load_truthfulqa(max_questions=600)
        print(f"  {len(out['truthfulqa'])} questions")
    if "arc" in args.tasks:
        print("Loading ARC-Challenge…")
        out["arc"] = load_arc(max_questions=600)
        print(f"  {len(out['arc'])} questions")
    if "gpqa" in args.tasks:
        print("Loading GPQA…")
        out["gpqa"] = load_gpqa(max_questions=600)
        print(f"  {len(out['gpqa'])} questions")
    return out


# ── Main runner ───────────────────────────────────────────────────────────────

def run_all(args) -> None:
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    task_questions = load_all_tasks(args)
    all_results = _load_existing(args.output)

    models = args.models if args.models else MODELS_DEFAULT
    primary_model = models[0]
    print(f"\nModels      : {models}")
    print(f"Primary     : {primary_model}")
    print(f"Tasks       : {list(task_questions.keys())}")
    print(f"Seeds       : {args.n_seeds}")
    print(f"Cache dir   : {args.cache_dir}\n")

    t0 = time.time()
    n_done_before = len(all_results)

    # ════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: Main multi-model grid
    # All models × all tasks × all k × overconfident × α=0.10 × 20 seeds
    # ════════════════════════════════════════════════════════════════════════
    print("=" * 68)
    print("EXPERIMENT 1: MAIN MULTI-MODEL (overconfident, α=0.10)")
    print("=" * 68)
    for model_id in models:
        agents = [LocalLLMAgent(i, model_id=model_id,
                                 hf_cache_dir=args.hf_cache_dir,
                                 result_cache_dir=args.cache_dir)
                  for i in range(N_AGENTS_MAIN)]

        for task, questions in task_questions.items():
            for seed in range(args.n_seeds):
                rng = random.Random(seed)
                shuffled = questions[:]
                rng.shuffle(shuffled)
                cal_n = max(50, len(shuffled) // 3)
                cal_q, test_q = shuffled[:cal_n], shuffled[cal_n:]
                if not test_q:
                    continue

                for k in N_CORRUPT_5:
                    if _is_done(all_results, model_id, task, N_AGENTS_MAIN,
                                k, ATTACK_MAIN, ALPHA_MAIN, seed, "main"):
                        continue
                    q_hat = build_q_hat(agents, cal_q, alpha=ALPHA_MAIN)
                    per_q_hat = build_per_agent_q_hat(agents, cal_q, alpha=ALPHA_MAIN)
                    res   = run_questions(
                        agents, test_q, n_corrupt=k,
                        attack_type=ATTACK_MAIN, q_hat=q_hat, seed=seed,
                        model_id=model_id, hf_cache_dir=args.hf_cache_dir,
                        result_cache_dir=args.cache_dir,
                        per_agent_q_hat=per_q_hat,
                    )
                    record = _build_record(model_id, task, N_AGENTS_MAIN, k,
                                           ATTACK_MAIN, ALPHA_MAIN, seed,
                                           q_hat, res, "main")
                    _add_or_replace(all_results, record)
                    _save(all_results, args.output)

                # Per-seed progress log
                done = len(all_results) - n_done_before
                elapsed = time.time() - t0
                if done > 0:
                    rate = elapsed / done
                    print(f"  [E1] {model_id.split('/')[-1]:30s} "
                          f"{task:11s} seed={seed:2d}  "
                          f"({done} records, {elapsed/60:.1f}m)",
                          flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: Attack & alpha ablation (PRIMARY MODEL only)
    # ════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 68)
    print(f"EXPERIMENT 2: ATTACK × ALPHA ABLATION ({primary_model.split('/')[-1]})")
    print("=" * 68)
    agents = [LocalLLMAgent(i, model_id=primary_model,
                             hf_cache_dir=args.hf_cache_dir,
                             result_cache_dir=args.cache_dir)
              for i in range(N_AGENTS_MAIN)]
    for task, questions in task_questions.items():
        for seed in range(args.n_seeds):
            rng = random.Random(seed)
            shuffled = questions[:]
            rng.shuffle(shuffled)
            cal_n = max(50, len(shuffled) // 3)
            cal_q, test_q = shuffled[:cal_n], shuffled[cal_n:]
            if not test_q:
                continue

            for attack in ATTACKS_ALL:
                for alpha in ALPHAS:
                    for k in N_CORRUPT_5:
                        if _is_done(all_results, primary_model, task,
                                    N_AGENTS_MAIN, k, attack, alpha, seed,
                                    "ablation"):
                            continue
                        q_hat = build_q_hat(agents, cal_q, alpha=alpha)
                        per_q_hat = build_per_agent_q_hat(agents, cal_q, alpha=alpha)
                        res   = run_questions(
                            agents, test_q, n_corrupt=k,
                            attack_type=attack, q_hat=q_hat, seed=seed,
                            model_id=primary_model,
                            hf_cache_dir=args.hf_cache_dir,
                            result_cache_dir=args.cache_dir,
                            per_agent_q_hat=per_q_hat,
                        )
                        record = _build_record(primary_model, task,
                                               N_AGENTS_MAIN, k, attack,
                                               alpha, seed, q_hat, res,
                                               "ablation")
                        _add_or_replace(all_results, record)
                        _save(all_results, args.output)
            elapsed = time.time() - t0
            print(f"  [E2] {task:11s} seed={seed:2d}  "
                  f"({len(all_results)} records, {elapsed/60:.1f}m)",
                  flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 3: Scaling (N=10) on MMLU, primary model
    # ════════════════════════════════════════════════════════════════════════
    if "mmlu" in task_questions:
        print()
        print("=" * 68)
        print(f"EXPERIMENT 3: N=10 SCALING ({primary_model.split('/')[-1]}, MMLU)")
        print("=" * 68)
        agents10 = [LocalLLMAgent(i, model_id=primary_model,
                                   hf_cache_dir=args.hf_cache_dir,
                                   result_cache_dir=args.cache_dir)
                    for i in range(N_AGENTS_BIG)]
        for seed in range(args.n_seeds):
            rng = random.Random(seed)
            shuffled = task_questions["mmlu"][:]
            rng.shuffle(shuffled)
            cal_n = max(50, len(shuffled) // 3)
            cal_q, test_q = shuffled[:cal_n], shuffled[cal_n:]
            if not test_q:
                continue
            for k in N_CORRUPT_10:
                if _is_done(all_results, primary_model, "mmlu",
                            N_AGENTS_BIG, k, ATTACK_MAIN, ALPHA_MAIN, seed,
                            "scaling"):
                    continue
                q_hat = build_q_hat(agents10, cal_q, alpha=ALPHA_MAIN)
                per_q_hat = build_per_agent_q_hat(agents10, cal_q, alpha=ALPHA_MAIN)
                res   = run_questions(
                    agents10, test_q, n_corrupt=k,
                    attack_type=ATTACK_MAIN, q_hat=q_hat, seed=seed,
                    model_id=primary_model,
                    hf_cache_dir=args.hf_cache_dir,
                    result_cache_dir=args.cache_dir,
                    per_agent_q_hat=per_q_hat,
                )
                record = _build_record(primary_model, "mmlu", N_AGENTS_BIG, k,
                                       ATTACK_MAIN, ALPHA_MAIN, seed,
                                       q_hat, res, "scaling")
                _add_or_replace(all_results, record)
                _save(all_results, args.output)
            elapsed = time.time() - t0
            print(f"  [E3] N=10 mmlu seed={seed:2d}  "
                  f"({len(all_results)} records, {elapsed/60:.1f}m)",
                  flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 4: HETEROGENEOUS committee — each slot is a different model
    # All slots use the same persona (Alpha) so the only variable is the LLM.
    # ════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 68)
    print(f"EXPERIMENT 4: HETEROGENEOUS COMMITTEE ({len(HETERO_COMMITTEE)} models)")
    print("=" * 68)
    # Build a 5-agent committee: agent i = (HETERO_COMMITTEE[i], persona=Alpha)
    hetero_agents = []
    for i, mid in enumerate(HETERO_COMMITTEE):
        # Use agent_id=0 → Alpha persona, but model_id varies
        hetero_agents.append(LocalLLMAgent(
            agent_id=0, model_id=mid,
            hf_cache_dir=args.hf_cache_dir,
            result_cache_dir=args.cache_dir,
        ))
    # Identifier for the hetero committee — used in records
    HETERO_ID = "hetero-committee:" + ",".join(m.split("/")[-1] for m in HETERO_COMMITTEE)

    for task, questions in task_questions.items():
        for seed in range(args.n_seeds):
            rng = random.Random(seed)
            shuffled = questions[:]
            rng.shuffle(shuffled)
            cal_n = max(50, len(shuffled) // 3)
            cal_q, test_q = shuffled[:cal_n], shuffled[cal_n:]
            if not test_q:
                continue
            for k in N_CORRUPT_5:
                if _is_done(all_results, HETERO_ID, task, len(hetero_agents),
                            k, ATTACK_MAIN, ALPHA_MAIN, seed, "hetero"):
                    continue
                q_hat = build_q_hat(hetero_agents, cal_q, alpha=ALPHA_MAIN)
                per_q_hat = build_per_agent_q_hat(hetero_agents, cal_q, alpha=ALPHA_MAIN)
                res = run_questions(
                    hetero_agents, test_q, n_corrupt=k,
                    attack_type=ATTACK_MAIN, q_hat=q_hat, seed=seed,
                    # For corrupt agents we still need a model_id — use the
                    # primary model's tokenizer for any LLM-based attack.
                    model_id=primary_model,
                    hf_cache_dir=args.hf_cache_dir,
                    result_cache_dir=args.cache_dir,
                    per_agent_q_hat=per_q_hat,
                )
                record = _build_record(HETERO_ID, task, len(hetero_agents), k,
                                       ATTACK_MAIN, ALPHA_MAIN, seed,
                                       q_hat, res, "hetero")
                _add_or_replace(all_results, record)
                _save(all_results, args.output)
            elapsed = time.time() - t0
            print(f"  [E4-hetero] {task:11s} seed={seed:2d}  "
                  f"({len(all_results)} records, {elapsed/60:.1f}m)",
                  flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 5: DOMAIN-SPECIALISED PERSONAS on primary model
    # 5 agents = same primary model + 5 different domain-expert personas.
    # MMLU only (only task with clean domain structure).
    # ════════════════════════════════════════════════════════════════════════
    if "mmlu" in task_questions:
        from ctc_llm.agents.local_llm_agent import DOMAIN_PERSONAS
        print()
        print("=" * 68)
        print(f"EXPERIMENT 5: DOMAIN-SPECIALISED PERSONAS ({primary_model.split('/')[-1]}, MMLU)")
        print("=" * 68)
        domain_agents = [LocalLLMAgent(
            agent_id=100 + i,   # offset so cache key differs from generic
            model_id=primary_model,
            hf_cache_dir=args.hf_cache_dir,
            result_cache_dir=args.cache_dir,
            persona=DOMAIN_PERSONAS[i],
        ) for i in range(len(DOMAIN_PERSONAS))]
        DOMAIN_ID = f"domain:{primary_model.split('/')[-1]}"

        for seed in range(args.n_seeds):
            rng = random.Random(seed)
            shuffled = task_questions["mmlu"][:]
            rng.shuffle(shuffled)
            cal_n = max(50, len(shuffled) // 3)
            cal_q, test_q = shuffled[:cal_n], shuffled[cal_n:]
            if not test_q:
                continue
            for k in N_CORRUPT_5:
                if _is_done(all_results, DOMAIN_ID, "mmlu",
                            len(domain_agents), k, ATTACK_MAIN, ALPHA_MAIN,
                            seed, "domain"):
                    continue
                q_hat = build_q_hat(domain_agents, cal_q, alpha=ALPHA_MAIN)
                per_q_hat = build_per_agent_q_hat(domain_agents, cal_q, alpha=ALPHA_MAIN)
                res = run_questions(
                    domain_agents, test_q, n_corrupt=k,
                    attack_type=ATTACK_MAIN, q_hat=q_hat, seed=seed,
                    model_id=primary_model,
                    hf_cache_dir=args.hf_cache_dir,
                    result_cache_dir=args.cache_dir,
                    per_agent_q_hat=per_q_hat,
                )
                record = _build_record(DOMAIN_ID, "mmlu", len(domain_agents), k,
                                       ATTACK_MAIN, ALPHA_MAIN, seed,
                                       q_hat, res, "domain")
                _add_or_replace(all_results, record)
                _save(all_results, args.output)
            elapsed = time.time() - t0
            print(f"  [E5-domain] mmlu seed={seed:2d}  "
                  f"({len(all_results)} records, {elapsed/60:.1f}m)",
                  flush=True)

    _save(all_results, args.output)
    print(f"\nDone. {len(all_results)} records saved to {args.output}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks",        nargs="+", default=["mmlu", "truthfulqa", "arc"])
    p.add_argument("--subjects",     nargs="*", default=None)
    p.add_argument("--n-seeds",      type=int,  default=20)
    p.add_argument("--models",       nargs="+", default=None,
                   help="HF model IDs; default = 3-model suite")
    p.add_argument("--hf-cache-dir", default="/data/models")
    p.add_argument("--cache-dir",    default="data/cache")
    p.add_argument("--output",       default="results/raw_results.json")
    return p.parse_args()


if __name__ == "__main__":
    run_all(parse_args())
