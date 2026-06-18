"""
Phase 2b — Free-form (non-MCQ) coordination experiments.

Both tasks use SequenceLogProbAgent: the model scores each candidate answer
by computing log P(candidate | context) — the full sequence probability,
NOT a first-token MCQ logit.

  gsm8k       "Problem: ...\\n\\nAnswer:" + integer candidates from answer pool
  hellaswag   Sentence stem + 4 full sentence endings

The downstream coordination code (majority, entropy-trust, CTC-Hybrid) is
IDENTICAL to the MCQ pipeline — only the probability extraction differs.

Output: results/raw_results_freeform.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ctc_llm.tasks.freeform import (
    FreeFormQuestion, load_gsm8k_free, load_hellaswag_seq,
)
from ctc_llm.agents.freeform_agent import SequenceLogProbAgent
from ctc_llm.agents.local_llm_agent import LocalLLMAgent
from ctc_llm.conformal.calibrate import calibrate as mcq_calibrate
from ctc_llm.conformal.calibrate import conformal_set as mcq_conformal_set
from ctc_llm.coordination.ctc import (
    ctc_answer, ctc_hybrid_answer, ctc_calibrated_answer, ctc_robust_answer,
    ctc_adaptive_answer, _entropy_np,
)
from ctc_llm.conformal.calibrate import compute_nonconformity_scores
from ctc_llm.coordination.majority import majority_answer
from ctc_llm.coordination.entropy_trust import entropy_trust_answer


# ── Config ────────────────────────────────────────────────────────────────────

N_AGENTS  = 5
N_CORRUPT = [0, 1, 2, 3]
ALPHA     = 0.10
N_SEEDS   = 20
ATTACKS   = ["overconfident", "random", "subtle"]

METHODS = ["vanilla", "average", "majority", "entropy_trust",
           "ctc_hybrid", "ctc_global", "ctc_calibrated", "ctc_robust",
           "ctc_adaptive"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _condition_key(r: Dict) -> tuple:
    return (r.get("model_id"), r.get("task"), r.get("n_agents"),
            r.get("n_corrupt"), r.get("attack"),
            round(float(r.get("alpha", 0.0)), 6),
            r.get("seed"), r.get("experiment", "freeform"))


def _is_done(existing, model_id, task, n_agents, k, attack, alpha, seed):
    required = {f"{m}_accuracy" for m in METHODS}
    for r in existing:
        if (r.get("model_id") == model_id and r["task"] == task and
                r["n_agents"] == n_agents and r["n_corrupt"] == k and
                r["attack"] == attack and abs(r["alpha"] - alpha) < 1e-6 and
                r["seed"] == seed and
                r.get("experiment", "freeform") == "freeform" and
                required.issubset(r.keys())):
            return True
    return False


def _load_existing(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(results: List[Dict], path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f)
    os.replace(tmp, path)


def _add_or_replace(records: List[Dict], new: Dict) -> None:
    k = _condition_key(new)
    for i, r in enumerate(records):
        if _condition_key(r) == k:
            records[i] = new
            return
    records.append(new)


# ── Corrupt agent factory (for np.ndarray distributions) ─────────────────────

def _make_corrupt_probs(clean_probs: np.ndarray, attack: str,
                        correct_idx: int, rng: random.Random) -> np.ndarray:
    n = len(clean_probs)
    if attack == "overconfident":
        # 0.97 mass on a wrong answer (the one with lowest clean probability)
        masked = clean_probs.copy()
        masked[correct_idx] = np.inf
        worst = int(np.argmin(masked))
        p = np.full(n, (1.0 - 0.97) / (n - 1), dtype=np.float32)
        p[worst] = 0.97
        return p
    elif attack == "random":
        return np.ones(n, dtype=np.float32) / n
    else:  # subtle
        q = 1.0 - clean_probs
        q = np.clip(q, 1e-6, None)
        return (q / q.sum()).astype(np.float32)


# ── Per-seed runner (shared for both tasks) ───────────────────────────────────

def _run_seed(
    agents: List[SequenceLogProbAgent],
    cal_q: List[FreeFormQuestion],
    test_q: List[FreeFormQuestion],
    n_corrupt: int,
    attack: str,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed + 42)

    def correct_idx(q: FreeFormQuestion) -> int:
        return q.candidates.index(q.correct_answer)

    # Calibrate q_hat on pooled clean-agent distributions
    cal_probs = [agents[i].get_probs(q) for q in cal_q for i in range(len(agents))]
    cal_labels = [correct_idx(q) for q in cal_q for _ in range(len(agents))]
    _, q_hat = mcq_calibrate(cal_probs, cal_labels, alpha=ALPHA)

    # Per-agent clean calibration q̂ (used by CTC-Calibrated). Calibration
    # always uses each agent's CLEAN distribution — corruption is applied
    # only at test time, so a corrupt agent deviates from its own clean
    # calibration profile, which is exactly the signal CTC-Calibrated uses.
    cal_labels_per = [correct_idx(q) for q in cal_q]
    per_agent_q_hat: Dict[int, float] = {}
    per_agent_cal_stats: Dict[int, tuple] = {}
    per_agent_profile: Dict[int, dict] = {}
    for i in range(len(agents)):
        agent_cal = [agents[i].get_probs(q) for q in cal_q]
        _, per_agent_q_hat[i] = mcq_calibrate(agent_cal, cal_labels_per, alpha=ALPHA)
        nc = compute_nonconformity_scores(agent_cal, cal_labels_per)
        per_agent_cal_stats[i] = (float(np.mean(nc)), float(np.std(nc)))
        ent = np.array([_entropy_np(p) for p in agent_cal])
        per_agent_profile[i] = {"H": (float(ent.mean()), float(ent.std())),
                                "q": (float(nc.mean()), float(nc.std()))}

    corrupt_ids = set(rng.sample(range(len(agents)), n_corrupt))

    per_q: Dict[str, List] = {m: [] for m in METHODS}
    per_q["committee_set_size"] = []
    per_q["committee_acc"] = []

    for q in test_q:
        cidx = correct_idx(q)
        committee: Dict[int, np.ndarray] = {}

        for i, ag in enumerate(agents):
            clean_p = ag.get_probs(q)
            if i in corrupt_ids:
                committee[i] = _make_corrupt_probs(clean_p, attack, cidx, rng)
            else:
                committee[i] = clean_p

        n_cands = len(q.candidates)
        mean_p = np.mean([committee[i] for i in range(len(agents))], axis=0)

        preds = {
            "vanilla":        int(np.argmax(committee[0])),
            "average":        int(np.argmax(mean_p)),
            "majority":       int(majority_answer(committee)),
            "entropy_trust":  int(entropy_trust_answer(committee)),
            "ctc_hybrid":     int(ctc_hybrid_answer(committee, q_hat)),
            "ctc_global":     int(ctc_answer(committee, q_hat)),
            "ctc_calibrated": int(ctc_calibrated_answer(committee, q_hat,
                                                        per_agent_q_hat)),
            "ctc_robust":     int(ctc_robust_answer(committee, q_hat,
                                                    per_agent_cal_stats)),
            "ctc_adaptive":   int(ctc_adaptive_answer(committee, q_hat,
                                                      per_agent_profile)),
        }

        # Committee abstention: union of conformal sets
        union = set()
        for p in committee.values():
            union |= set(mcq_conformal_set(p, q_hat))

        for m in METHODS:
            per_q[m].append(1 if preds[m] == cidx else 0)
        per_q["committee_set_size"].append(len(union))
        per_q["committee_acc"].append(1 if preds["ctc_hybrid"] == cidx else 0)

    return {
        "n_questions": len(test_q),
        "q_hat": q_hat,
        **{f"{m}_accuracy": float(np.mean(per_q[m])) for m in METHODS},
        **{f"{m}_per_q": per_q[m] for m in METHODS},
        "committee_set_size_per_q": per_q["committee_set_size"],
        "committee_acc_per_q": per_q["committee_acc"],
        "committee_accuracy": float(np.mean(per_q["committee_acc"])),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_freeform(args) -> None:
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    all_results = _load_existing(args.output)
    print(f"Resuming: {len(all_results)} existing records in {args.output}")

    primary_model = args.models[0]
    t0 = time.time()

    task_loaders = {
        "gsm8k":     lambda: load_gsm8k_free(max_questions=args.max_questions),
        "hellaswag": lambda: load_hellaswag_seq(max_questions=args.max_questions),
    }

    for task in args.tasks:
        if task not in task_loaders:
            continue
        print(f"\n{'='*68}")
        print(f"FREE-FORM TASK: {task.upper()}  (sequence log-prob scoring)")
        print(f"{'='*68}")
        questions = task_loaders[task]()
        print(f"  {len(questions)} questions loaded")

        agents = [
            SequenceLogProbAgent(i, model_id=primary_model,
                                 hf_cache_dir=args.hf_cache_dir,
                                 result_cache_dir=args.cache_dir,
                                 device="cpu")   # GPU inference already cached
            for i in range(N_AGENTS)
        ]

        for seed in range(N_SEEDS):
            rng = random.Random(seed)
            shuffled = questions[:]
            rng.shuffle(shuffled)
            cal_n = max(50, len(shuffled) // 3)
            cal_q, test_q = shuffled[:cal_n], shuffled[cal_n:]
            if not test_q:
                continue

            for k in N_CORRUPT:
                for attack in ATTACKS:
                    if _is_done(all_results, primary_model, task,
                                N_AGENTS, k, attack, ALPHA, seed):
                        continue
                    res = _run_seed(agents, cal_q, test_q, k, attack, seed)
                    record = {
                        "model_id":     primary_model,
                        "task":         task,
                        "experiment":   "freeform",
                        "n_agents":     N_AGENTS,
                        "n_corrupt":    k,
                        "corrupt_frac": k / N_AGENTS,
                        "attack":       attack,
                        "alpha":        ALPHA,
                        "seed":         seed,
                        **res,
                    }
                    _add_or_replace(all_results, record)
                    _save(all_results, args.output)

            elapsed = time.time() - t0
            print(f"  [{task}] seed={seed:2d}  "
                  f"({len(all_results)} records, {elapsed/60:.1f}m)", flush=True)

    _save(all_results, args.output)
    print(f"\nDone. {len(all_results)} records saved to {args.output}")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks",         nargs="+", default=["gsm8k", "hellaswag"])
    p.add_argument("--models",        nargs="+", default=["Qwen/Qwen2.5-7B-Instruct"])
    p.add_argument("--hf-cache-dir",  default="/data/models")
    p.add_argument("--cache-dir",     default="data/cache_freeform")
    p.add_argument("--output",        default="results/raw_results_freeform.json")
    p.add_argument("--max-questions", type=int, default=400)
    return p.parse_args()


if __name__ == "__main__":
    run_freeform(_parse_args())
