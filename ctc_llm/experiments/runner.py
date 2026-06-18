"""
Core per-question runner for multi-agent coordination experiments.

Runs every coordination method on the same agent_probs dict so methods
see identical inputs.  Tracks per-question accuracy for cross-seed CI.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Any

import numpy as np

from ctc_llm.tasks.mmlu import Question
from ctc_llm.agents.base import Agent
from ctc_llm.agents.corrupt_agent import make_corrupt_agent
from ctc_llm.conformal.calibrate import calibrate, conformal_set
from ctc_llm.coordination.vanilla       import vanilla_answer
from ctc_llm.coordination.majority      import majority_answer
from ctc_llm.coordination.entropy_trust import entropy_trust_answer
from ctc_llm.coordination.average       import average_answer
from ctc_llm.coordination.ctc           import (
    ctc_answer, ctc_focal_answer, ctc_agreement_answer,
    ctc_calibrated_answer, ctc_hybrid_answer,
    ctc_robust_answer, ctc_adaptive_answer, _entropy_np,
)
from ctc_llm.coordination.confidence    import (
    vanilla_confidence, average_confidence, majority_confidence,
    entropy_confidence, ctc_confidence, ctc_hybrid_confidence,
    committee_conformal_abstain,
)
from ctc_llm.coordination.self_consistency import (
    self_consistency_answer, self_consistency_confidence,
)
from ctc_llm.coordination.debate            import (
    debate_answer, debate_confidence,
)
from ctc_llm.coordination.mixture_of_agents import (
    mixture_of_agents_answer, mixture_of_agents_confidence,
)
from ctc_llm.coordination.llm_judge         import (
    llm_judge_answer, llm_judge_confidence,
)


# Methods reported in tables/figures
METHODS = [
    "vanilla", "average", "majority", "entropy",
    # 2022-2024 standard baselines
    "self_consistency", "debate", "mixture_of_agents", "llm_judge",
    # Our family
    "ctc", "ctc_focal", "ctc_agreement",
    "ctc_calibrated", "ctc_hybrid",
    # Unified / generalizable trust (calibration-anomaly)
    "ctc_robust", "ctc_adaptive",
]


# ── Calibration ───────────────────────────────────────────────────────────────

def build_q_hat(
    agents: List[Agent],
    cal_questions: List[Question],
    alpha: float = 0.10,
) -> float:
    """Pool calibration data from all clean agents and compute q̂."""
    all_probs:   List[np.ndarray] = []
    all_correct: List[int]        = []
    for agent in agents:
        for q in cal_questions:
            all_probs.append(agent.get_probs(q.question, q.choices))
            all_correct.append(q.correct)
    _, q_hat = calibrate(all_probs, all_correct, alpha=alpha)
    return float(q_hat)


def build_per_agent_q_hat(
    agents: List[Agent],
    cal_questions: List[Question],
    alpha: float = 0.10,
) -> Dict[int, float]:
    """Per-agent q̂ computed from each agent's clean calibration distribution.

    Returns {agent_id: q_hat_i}. Used by CTC-Calibrated to detect
    test-time deviations from calibration distribution.
    """
    out: Dict[int, float] = {}
    for i, agent in enumerate(agents):
        ps, cs = [], []
        for q in cal_questions:
            ps.append(agent.get_probs(q.question, q.choices))
            cs.append(q.correct)
        _, q_i = calibrate(ps, cs, alpha=alpha)
        out[i] = float(q_i)
    return out


def build_per_agent_profile(
    agents: List[Agent],
    cal_questions: List[Question],
) -> Dict[int, dict]:
    """Per-agent clean-calibration feature profile for ctc_robust / ctc_adaptive.

    Returns {agent_id: {"H": (mu, sigma), "q": (mu, sigma)}} where H is the
    Shannon entropy and q = 1 - max(pi) is the top-answer nonconformity, both
    summarised over the agent's clean calibration distribution.
    """
    out: Dict[int, dict] = {}
    for i, agent in enumerate(agents):
        Hs, qs = [], []
        for q in cal_questions:
            p = agent.get_probs(q.question, q.choices)
            Hs.append(_entropy_np(p))
            qs.append(1.0 - float(np.max(p)))
        Hs, qs = np.asarray(Hs), np.asarray(qs)
        out[i] = {"H": (float(Hs.mean()), float(Hs.std())),
                  "q": (float(qs.mean()), float(qs.std()))}
    return out


# ── Single question ──────────────────────────────────────────────────────────

def run_question(
    question: Question,
    clean_agents: List[Agent],
    n_corrupt: int,
    attack_type: str,
    q_hat: float,
    corrupt_seed: int,
    model_id: Optional[str]         = None,
    hf_cache_dir: Optional[str]     = None,
    result_cache_dir: Optional[str] = None,
    per_agent_q_hat: Optional[Dict[int, float]] = None,
    per_agent_profile: Optional[Dict[int, dict]] = None,
) -> Dict[str, Any]:
    """Run one question with k agents corrupted (selection seeded by corrupt_seed)."""
    n_agents    = len(clean_agents)
    rng         = random.Random(corrupt_seed)
    corrupt_ids = set(rng.sample(range(n_agents), k=min(n_corrupt, n_agents)))

    agent_probs: Dict[int, np.ndarray] = {}
    for i, agent in enumerate(clean_agents):
        if i in corrupt_ids:
            # Use the SLOT-specific model_id, not the experiment-level one.
            # This matters for heterogeneous committees (each slot a different
            # base LLM).  Fall back to the caller-supplied model_id otherwise.
            slot_model_id = getattr(agent, "model_id", model_id)
            corrupt = make_corrupt_agent(
                attack_type, agent,
                correct_idx=question.correct,
                agent_id=i,
                model_id=slot_model_id,
                hf_cache_dir=hf_cache_dir,
                result_cache_dir=result_cache_dir,
            )
            agent_probs[i] = corrupt.get_probs(question.question, question.choices)
        else:
            agent_probs[i] = agent.get_probs(question.question, question.choices)

    correct = question.correct

    sc_seed = (corrupt_seed * 31 + 17) & 0x7FFFFFFF
    ans = {
        "vanilla":        vanilla_answer(agent_probs, focal_agent=0),
        "average":        average_answer(agent_probs),
        "majority":       majority_answer(agent_probs),
        "entropy":        entropy_trust_answer(agent_probs),
        "self_consistency":   self_consistency_answer(agent_probs, seed=sc_seed),
        "debate":             debate_answer(agent_probs),
        "mixture_of_agents":  mixture_of_agents_answer(agent_probs),
        "llm_judge":          llm_judge_answer(agent_probs),
        "ctc":                ctc_answer(agent_probs, q_hat),
        "ctc_focal":      ctc_focal_answer(agent_probs, q_hat),
        "ctc_agreement":  ctc_agreement_answer(agent_probs, q_hat),
        "ctc_calibrated": ctc_calibrated_answer(agent_probs, q_hat,
                                                per_agent_q_hat=per_agent_q_hat),
        "ctc_hybrid":     ctc_hybrid_answer(agent_probs, q_hat,
                                            per_agent_q_hat=per_agent_q_hat),
        "ctc_robust":     ctc_robust_answer(
            agent_probs, q_hat,
            per_agent_cal_stats={i: per_agent_profile[i]["q"]
                                 for i in per_agent_profile}
            if per_agent_profile else None),
        "ctc_adaptive":   ctc_adaptive_answer(agent_probs, q_hat,
                                              per_agent_profile=per_agent_profile),
    }

    # ── Confidence / abstention signals per method ────────────────────────
    _, conf_vanilla  = vanilla_confidence(agent_probs)
    _, conf_average  = average_confidence(agent_probs)
    _, conf_majority = majority_confidence(agent_probs)
    _, conf_entropy  = entropy_confidence(agent_probs)
    _, conf_sc       = self_consistency_confidence(agent_probs, seed=sc_seed)
    _, conf_debate   = debate_confidence(agent_probs)
    _, conf_moa      = mixture_of_agents_confidence(agent_probs)
    _, conf_judge    = llm_judge_confidence(agent_probs)
    _, conf_ctc, _   = ctc_confidence(agent_probs, q_hat)
    _, conf_hyb, _   = ctc_hybrid_confidence(agent_probs, q_hat)

    # Committee-wide conformal abstention signal — the headline novelty
    cmt_ans, cmt_set_size, cmt_score_concentration = \
        committee_conformal_abstain(agent_probs, q_hat)
    committee_acc = float(cmt_ans == correct)

    # Empirical coverage: probability the correct answer is in *any* clean agent's
    # conformal set.  We average across the (non-corrupt) clean agents.
    cs_clean = [conformal_set(agent_probs[i], q_hat)
                for i in range(n_agents) if i not in corrupt_ids]
    coverage = float(np.mean([1.0 if correct in cs else 0.0 for cs in cs_clean])) \
               if cs_clean else 0.0
    set_size = float(np.mean([len(cs) for cs in cs_clean])) if cs_clean else 0.0

    out = {f"{m}_acc": float(ans[m] == correct) for m in METHODS}
    out["ctc_set_size"]   = set_size
    out["ctc_covered"]    = coverage
    out["corrupt_ids"]    = list(corrupt_ids)
    out["correct"]        = correct
    # Per-method confidence for risk-coverage analysis
    out["vanilla_conf"]   = conf_vanilla
    out["average_conf"]   = conf_average
    out["majority_conf"]  = conf_majority
    out["entropy_conf"]   = conf_entropy
    out["self_consistency_conf"] = conf_sc
    out["debate_conf"]    = conf_debate
    out["mixture_of_agents_conf"] = conf_moa
    out["llm_judge_conf"] = conf_judge
    out["ctc_conf"]       = conf_ctc
    out["ctc_hybrid_conf"] = conf_hyb
    # Committee-wide conformal abstention (HEADLINE: selective prediction)
    out["committee_acc"]              = committee_acc
    out["committee_set_size"]         = cmt_set_size
    out["committee_score_concentration"] = cmt_score_concentration
    return out


# ── Many questions ───────────────────────────────────────────────────────────

def run_questions(
    clean_agents: List[Agent],
    test_questions: List[Question],
    n_corrupt: int,
    attack_type: str,
    q_hat: float,
    seed: int = 0,
    model_id: Optional[str]         = None,
    hf_cache_dir: Optional[str]     = None,
    result_cache_dir: Optional[str] = None,
    per_agent_q_hat: Optional[Dict[int, float]] = None,
    per_agent_profile: Optional[Dict[int, dict]] = None,
) -> Dict[str, Any]:
    """Run all test questions for a given (n_corrupt, attack) configuration."""
    # Accumulate per-question accuracy and confidence for every method
    acc_keys = [f"{m}_acc" for m in METHODS]
    conf_keys = ["vanilla_conf", "average_conf", "majority_conf",
                 "entropy_conf", "self_consistency_conf", "debate_conf",
                 "mixture_of_agents_conf", "llm_judge_conf",
                 "ctc_conf", "ctc_hybrid_conf"]
    cmt_keys = ["committee_acc", "committee_set_size", "committee_score_concentration"]
    other_keys = ["ctc_set_size", "ctc_covered"]
    all_keys = acc_keys + conf_keys + cmt_keys + other_keys

    results: Dict[str, List[float]] = {k: [] for k in all_keys}

    for idx, q in enumerate(test_questions):
        r = run_question(
            question         = q,
            clean_agents     = clean_agents,
            n_corrupt        = n_corrupt,
            attack_type      = attack_type,
            q_hat            = q_hat,
            corrupt_seed     = seed * 100_000 + idx,
            model_id         = model_id,
            hf_cache_dir     = hf_cache_dir,
            result_cache_dir = result_cache_dir,
            per_agent_q_hat  = per_agent_q_hat,
            per_agent_profile = per_agent_profile,
        )
        for k in results:
            if k in r:
                results[k].append(r[k])

    summary: Dict[str, Any] = {}
    for m in METHODS:
        vals = results[f"{m}_acc"]
        summary[f"{m}_accuracy"] = float(np.mean(vals)) if vals else 0.0
        summary[f"{m}_per_q"]    = vals
    # Per-question confidence (needed for risk-coverage curves)
    for k in conf_keys + cmt_keys:
        summary[f"{k}_per_q"] = results[k]
    summary["ctc_coverage"]      = float(np.mean(results["ctc_covered"])) if results["ctc_covered"] else 0.0
    summary["ctc_mean_set_size"] = float(np.mean(results["ctc_set_size"])) if results["ctc_set_size"] else 0.0
    summary["committee_accuracy"] = float(np.mean(results["committee_acc"])) if results["committee_acc"] else 0.0
    summary["n_questions"]       = len(test_questions)
    return summary
