"""
Focused validation: does ctc_robust (scale-normalised calibration trust)
also hold on the PEAKED-base MCQ task ARC — the case that broke ctc_calibrated
(ARC k=3 overconfident = 0.094)?  Reads only from the existing MCQ cache.
"""
import random, numpy as np
from ctc_llm.tasks.mmlu import load_arc, split_questions
from ctc_llm.agents.local_llm_agent import LocalLLMAgent
from ctc_llm.agents.corrupt_agent import make_corrupt_agent
from ctc_llm.conformal.calibrate import calibrate, compute_nonconformity_scores
from ctc_llm.coordination.majority import majority_answer
from ctc_llm.coordination.entropy_trust import entropy_trust_answer
from ctc_llm.coordination.ctc import (
    ctc_answer, ctc_hybrid_answer, ctc_calibrated_answer, ctc_robust_answer,
)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
N, ALPHA, ATTACK = 5, 0.10, "overconfident"
qs = load_arc(max_questions=600)
agents = [LocalLLMAgent(i, model_id=MODEL, hf_cache_dir="/data/models",
                        result_cache_dir="data/cache", device="cpu")
          for i in range(N)]

def evaluate(k, seeds=range(10)):
    acc = {m: [] for m in ["majority","entropy","ctc_global","ctc_hybrid","ctc_calibrated","ctc_robust"]}
    for seed in seeds:
        rng = random.Random(seed)
        sh = qs[:]; rng.shuffle(sh)
        cal_n = max(50, len(sh)//3)
        cal_q, test_q = sh[:cal_n], sh[cal_n:]
        # pooled + per-agent calibration
        allp, allc = [], []
        per_qhat, per_stats = {}, {}
        for i, ag in enumerate(agents):
            ps = [ag.get_probs(q.question, q.choices) for q in cal_q]
            cs = [q.correct for q in cal_q]
            allp += ps; allc += cs
            _, qi = calibrate(ps, cs, alpha=ALPHA); per_qhat[i] = qi
            nc = compute_nonconformity_scores(ps, cs)
            per_stats[i] = (float(np.mean(nc)), float(np.std(nc)))
        _, q_hat = calibrate(allp, allc, alpha=ALPHA)

        rc = random.Random(seed + 42)
        corrupt_ids = set(rc.sample(range(N), k))
        per_q = {m: [] for m in acc}
        for q in test_q:
            committee = {}
            for i, ag in enumerate(agents):
                if i in corrupt_ids:
                    cor = make_corrupt_agent(ATTACK, ag, correct_idx=q.correct,
                                             agent_id=i, model_id=MODEL,
                                             hf_cache_dir="/data/models",
                                             result_cache_dir="data/cache")
                    committee[i] = cor.get_probs(q.question, q.choices)
                else:
                    committee[i] = ag.get_probs(q.question, q.choices)
            c = q.correct
            per_q["majority"].append(majority_answer(committee) == c)
            per_q["entropy"].append(entropy_trust_answer(committee) == c)
            per_q["ctc_global"].append(ctc_answer(committee, q_hat) == c)
            per_q["ctc_hybrid"].append(ctc_hybrid_answer(committee, q_hat, per_qhat) == c)
            per_q["ctc_calibrated"].append(ctc_calibrated_answer(committee, q_hat, per_qhat) == c)
            per_q["ctc_robust"].append(ctc_robust_answer(committee, q_hat, per_stats) == c)
        for m in acc:
            acc[m].append(np.mean(per_q[m]))
    return {m: float(np.mean(v)) for m, v in acc.items()}

print(f"ARC | {ATTACK} | N={N} | Qwen2.5-7B  (10 seeds)\n")
print(f"{'k':>2} | {'major':>7} {'entrpy':>7} {'GLOBAL':>7} {'hybrid':>7} {'CALIB':>7} {'ROBUST':>7}")
for k in [1, 2, 3]:
    r = evaluate(k)
    print(f"{k:>2} | "+ " ".join(f"{r[m]:7.3f}" for m in
          ["majority","entropy","ctc_global","ctc_hybrid","ctc_calibrated","ctc_robust"]))
