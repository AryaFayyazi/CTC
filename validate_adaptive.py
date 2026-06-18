"""
Unified validation of ctc_adaptive (calibration-anomaly trust) on BOTH
a peaked-base MCQ task (ARC) and a flat-base free-form task (HellaSwag),
across k=0..3 and all three attacks. Reads only from existing caches.
"""
import random, numpy as np
from ctc_llm.conformal.calibrate import (
    calibrate, compute_nonconformity_scores, conformal_set,
)
from ctc_llm.coordination.majority import majority_answer
from ctc_llm.coordination.entropy_trust import entropy_trust_answer
from ctc_llm.coordination.ctc import (
    ctc_answer, ctc_hybrid_answer, ctc_calibrated_answer,
    ctc_robust_answer, ctc_adaptive_answer, _entropy_np,
)

MODEL, N, ALPHA, SEEDS = "Qwen/Qwen2.5-7B-Instruct", 5, 0.10, range(8)
METHODS = ["majority","entropy","ctc_hybrid","ctc_robust","ctc_adaptive"]
ATTACKS = ["overconfident", "random", "subtle"]


def profile(probs_list):
    """Per-agent clean-calibration feature profile: means/stds of (H, q)."""
    H = np.array([_entropy_np(p) for p in probs_list])
    q = np.array([1.0 - float(np.max(p)) for p in probs_list])
    return {"H": (float(H.mean()), float(H.std())),
            "q": (float(q.mean()), float(q.std()))}


def run_task(name, getp, questions, correct_of, corrupt_fn, attacks):
    print(f"\n===== {name} =====")
    print(f"{'attack':13} {'k':>1} | " + " ".join(f"{m[:7]:>8}" for m in METHODS))
    agg = {(a, k): {m: [] for m in METHODS} for a in attacks for k in range(4)}
    for seed in SEEDS:
        rng = random.Random(seed); sh = questions[:]; rng.shuffle(sh)
        cn = max(50, len(sh)//3); cal_q, test_q = sh[:cn], sh[cn:]
        allp, allc, per_qhat, per_prof = [], [], {}, {}
        for i in range(N):
            ps = [getp(i, q) for q in cal_q]; cs = [correct_of(q) for q in cal_q]
            allp += ps; allc += cs
            _, per_qhat[i] = calibrate(ps, cs, alpha=ALPHA)
            per_prof[i] = profile(ps)
        _, q_hat = calibrate(allp, allc, alpha=ALPHA)
        for attack in attacks:
            for k in range(4):
                rc = random.Random(seed + 42); cids = set(rc.sample(range(N), k))
                acc = {m: [] for m in METHODS}
                for q in test_q:
                    c = correct_of(q); comm = {}
                    for i in range(N):
                        comm[i] = corrupt_fn(i, q, c, attack) if i in cids else getp(i, q)
                    acc["majority"].append(majority_answer(comm) == c)
                    acc["entropy"].append(entropy_trust_answer(comm) == c)
                    acc["ctc_hybrid"].append(ctc_hybrid_answer(comm, q_hat, per_qhat) == c)
                    acc["ctc_robust"].append(ctc_robust_answer(comm, q_hat,
                        {i: per_prof[i]["q"] for i in per_prof}) == c)
                    acc["ctc_adaptive"].append(ctc_adaptive_answer(comm, q_hat, per_prof) == c)
                for m in METHODS:
                    agg[(attack, k)][m].append(np.mean(acc[m]))
    for attack in attacks:
        for k in range(4):
            r = {m: np.mean(agg[(attack, k)][m]) for m in METHODS}
            print(f"{attack:13} {k:>1} | " + " ".join(f"{r[m]:8.3f}" for m in METHODS))


# ---- ARC (peaked MCQ) ----
from ctc_llm.tasks.mmlu import load_arc
from ctc_llm.agents.local_llm_agent import LocalLLMAgent
from ctc_llm.agents.corrupt_agent import make_corrupt_agent
arc_q = load_arc(max_questions=600)
arc_ag = [LocalLLMAgent(i, model_id=MODEL, hf_cache_dir="/data/models",
          result_cache_dir="data/cache", device="cpu") for i in range(N)]
run_task("ARC (peaked MCQ)",
         lambda i, q: arc_ag[i].get_probs(q.question, q.choices),
         arc_q, lambda q: q.correct,
         lambda i, q, c, atk: make_corrupt_agent(atk, arc_ag[i], correct_idx=c,
             agent_id=i, model_id=MODEL, hf_cache_dir="/data/models",
             result_cache_dir="data/cache").get_probs(q.question, q.choices),
         ATTACKS)

# ---- free-form (flat) ----
from ctc_llm.tasks.freeform import load_hellaswag_seq, load_gsm8k_free
from ctc_llm.agents.freeform_agent import SequenceLogProbAgent
from ctc_llm.experiments.run_freeform import _make_corrupt_probs
cidx = lambda q: q.candidates.index(q.correct_answer)
for tname, loader, cdir in [("GSM8K (flat)", load_gsm8k_free, "data/cache_freeform"),
                            ("HellaSwag (flat)", load_hellaswag_seq, "data/cache_freeform")]:
    ff_q = loader(max_questions=400)
    ff_ag = [SequenceLogProbAgent(i, model_id=MODEL, hf_cache_dir="/data/models",
             result_cache_dir=cdir, device="cpu") for i in range(N)]
    run_task(tname,
             lambda i, q, _ag=ff_ag: _ag[i].get_probs(q), ff_q, cidx,
             lambda i, q, c, atk, _ag=ff_ag: _make_corrupt_probs(_ag[i].get_probs(q),
                 atk, c, random.Random(0)),
             ATTACKS)
