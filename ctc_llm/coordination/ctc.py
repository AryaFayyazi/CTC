"""
Conformal Trust Coordination (CTC) for LLM agents.

Two variants
------------
ctc_answer (PRIMARY, global aggregation)
    score(a) = Σ_i  π_i(a) · T_i · 1[a ∈ C_i]
    No focal agent — every agent contributes its own belief weighted by its
    conformal trust, conditioned on the action being in its conformal set.

ctc_focal_answer (ABLATION, original focal version)
    score_i(a) = π_i(a) · Σ_j T_j · 1[a ∈ C_j]
    Picks the agent with the smallest conformal set as focal and uses *its*
    belief π_i.  Vulnerable when corrupt agents also produce singleton sets
    (e.g. overconfident attack on a confident base model), because they can
    win the focal-selection tie-break.

Why global is more robust
-------------------------
A corrupt overconfident agent has |C|=1 (singleton on the wrong answer).
In the focal version it may be selected as focal → score collapses to
that agent's own (wrong) distribution.
In the global version it contributes only its own π · T to its single
action; the (N-k) clean agents contribute π · T to the correct answer.
When clean agents collectively outweigh the corrupt block, the answer
is correct — and this holds even when the base model is highly confident.

Trust score
-----------
    T_j = 1 / |C_j|
A compact conformal set indicates confident-yet-calibrated prediction.

Conformal coverage
------------------
By split-conformal calibration, P(a* ∈ C_j) ≥ 1 - α for every agent j
that draws from the same calibration distribution as test.  This means
the correct answer almost always *contributes* to the global score; the
question is whether incorrect answers also accumulate enough weight to
beat it.  CTC's design ensures incorrect answers can only accumulate
weight from agents whose conformal sets contain them — a corrupt agent
endorsing a wrong answer must "spend" its trust budget on that wrong
action, leaving correct-answer endorsement to the clean majority.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from ctc_llm.conformal.calibrate import conformal_set


def ctc_answer(
    agent_probs: Dict[int, np.ndarray],
    q_hat: float,
) -> int:
    """
    PRIMARY: global trust-weighted CTC.

        score(a) = Σ_i  π_i(a) · T_i · 1[a ∈ C_i]

    Returns argmax_a score(a).
    """
    # Conformal sets and per-agent trust
    conf_sets: Dict[int, List[int]] = {}
    trust:     Dict[int, float]     = {}
    for j, p in agent_probs.items():
        cs           = conformal_set(p, q_hat)
        conf_sets[j] = cs
        trust[j]     = 1.0 / len(cs)

    n_choices = len(next(iter(agent_probs.values())))
    scores    = np.zeros(n_choices, dtype=np.float64)

    for a in range(n_choices):
        for j, p in agent_probs.items():
            if a in conf_sets[j]:
                scores[a] += float(p[a]) * trust[j]

    return int(np.argmax(scores))


def ctc_focal_answer(
    agent_probs: Dict[int, np.ndarray],
    q_hat: float,
    focal_agent: Optional[int] = None,
) -> int:
    """
    ABLATION: original focal-agent CTC (for comparison).

    Picks the agent with the smallest conformal set as focal (highest
    trust, ties broken by id), then:

        score_i(a) = π_i(a) · Σ_j T_j · 1[a ∈ C_j]
    """
    conf_sets: Dict[int, List[int]] = {}
    trust:     Dict[int, float]     = {}
    for j, p in agent_probs.items():
        cs           = conformal_set(p, q_hat)
        conf_sets[j] = cs
        trust[j]     = 1.0 / len(cs)

    if focal_agent is None:
        focal_agent = min(trust, key=lambda j: (len(conf_sets[j]), j))

    pi_i = agent_probs[focal_agent]
    n_choices = len(pi_i)

    scores = np.zeros(n_choices, dtype=np.float64)
    for a in range(n_choices):
        W_a = sum(trust[j] for j in agent_probs if a in conf_sets[j])
        scores[a] = float(pi_i[a]) * W_a

    return int(np.argmax(scores))


def ctc_hybrid_answer(
    agent_probs: Dict[int, np.ndarray],
    q_hat: float,
    per_agent_q_hat: Optional[Dict[int, float]] = None,
) -> int:
    """
    PRIMARY (best-of-both): adaptive CTC ⊕ confidence-trust hybrid.

    Insight: the trust signal 1/|C| saturates to 1 for all agents when
    conformal sets collapse to singletons (the high-base-confidence regime).
    In that regime, an entropy/probability-based trust signal still has
    discrimination.  Conversely, when sets are diverse (the uncertain
    regime), conformal trust is highly informative.

    We combine them:

        T_i = T_size_i  ×  T_entropy_i
        T_size_i    = 1 / |C_i|              (CTC's structural trust)
        T_entropy_i = 1 / (H(π_i) + ε)       (resolves singleton ties)

    Then:
        score(a) = Σ_i  π_i(a) · T_i · 1[a ∈ C_i]

    When sets are diverse: T_size dominates → behaves like CTC.
    When sets are all singletons: T_entropy dominates → behaves like Entropy.
    In between: smooth interpolation.

    `per_agent_q_hat` is accepted for API symmetry; currently unused
    by this aggregator but kept so all CTC variants share a signature.
    """
    conf_sets: Dict[int, List[int]] = {}
    trust:     Dict[int, float]     = {}
    for j, p in agent_probs.items():
        cs = conformal_set(p, q_hat)
        conf_sets[j] = cs
        t_size = 1.0 / len(cs)
        h      = float(-np.sum(np.clip(p, 1e-12, None) *
                               np.log(np.clip(p, 1e-12, None))))
        t_ent  = 1.0 / (h + 1e-8)
        trust[j] = t_size * t_ent

    n_choices = len(next(iter(agent_probs.values())))
    scores = np.zeros(n_choices, dtype=np.float64)
    for a in range(n_choices):
        for j, p in agent_probs.items():
            if a in conf_sets[j]:
                scores[a] += float(p[a]) * trust[j]
    if scores.max() == 0:
        return ctc_answer(agent_probs, q_hat)
    return int(np.argmax(scores))


def ctc_calibrated_answer(
    agent_probs: Dict[int, np.ndarray],
    q_hat: float,
    per_agent_q_hat: Optional[Dict[int, float]] = None,
) -> int:
    """
    PRIMARY (rigorous): calibration-aware CTC.

    Each agent's trust at test time combines two signals:
      (i)  conformal-set compactness: T_size = 1 / |C_i|
      (ii) calibration consistency:   T_cal = exp(-|q_test - q_cal|)
           where q_test = nonconformity on this question (if known) and
           q_cal = per-agent q̂ from calibration set.

    A corrupt agent has an *atypical* per-question nonconformity profile
    relative to its calibration distribution.  We use this as a robustness
    signal that is INDEPENDENT of conformal set size — so the algorithm
    keeps signal even when all sets are singletons.

    score(a) = Σ_i  π_i(a) · T_i · 1[a ∈ C_i]
       T_i = T_size_i × T_cal_i
       T_cal_i = exp(- |  (1 - max π_i)  -  q_cal_i  |  )

    The intuition: at test time, a clean agent's nonconformity
    (1 - max π_i) should be close to its calibration-set distribution.
    A corrupt agent that systematically lies has different nonconformity
    statistics — it concentrates probability on a wrong answer with
    confidence (1 - max π_i) ≈ 0, which is atypical if q_cal > 0.
    """
    conf_sets: Dict[int, List[int]] = {}
    trust:     Dict[int, float]     = {}

    for j, p in agent_probs.items():
        cs           = conformal_set(p, q_hat)
        conf_sets[j] = cs
        t_size       = 1.0 / len(cs)

        # Calibration-aware trust: penalise atypical nonconformity
        if per_agent_q_hat is not None and j in per_agent_q_hat:
            q_test = 1.0 - float(np.max(p))
            t_cal  = float(np.exp(-abs(q_test - per_agent_q_hat[j])))
        else:
            t_cal = 1.0

        trust[j] = t_size * t_cal

    n_choices = len(next(iter(agent_probs.values())))
    scores = np.zeros(n_choices, dtype=np.float64)
    for a in range(n_choices):
        for j, p in agent_probs.items():
            if a in conf_sets[j]:
                scores[a] += float(p[a]) * trust[j]

    if scores.max() == 0:
        return ctc_answer(agent_probs, q_hat)
    return int(np.argmax(scores))


def ctc_robust_answer(
    agent_probs: Dict[int, np.ndarray],
    q_hat: float,
    per_agent_cal_stats: Optional[Dict[int, tuple]] = None,
) -> int:
    """
    GLOBAL CANDIDATE: scale-normalised calibration-aware CTC.

    Goal: a single trust rule that is robust to the overconfident attack on
    BOTH peaked-base tasks (ARC/MMLU, where honest agents are very confident)
    AND flat-base tasks (GSM8K/HellaSwag, where honest agents are diffuse).

    The failure of the two existing trust terms is that each is tied to an
    ABSOLUTE confidence scale:
      • 1/(H+ε)  (CTC-Hybrid)  rewards low entropy → fatal when honest agents
        are flatter than the attacker (free-form).
      • exp(-|q_test - q_cal|) (CTC-Calibrated) uses an absolute deviation →
        vanishes when honest agents are already as peaked as the attacker
        (ARC), because |q_test - q_cal| is then tiny for the attacker too.

    Fix: measure how atypical the agent's test-time nonconformity is RELATIVE
    TO ITS OWN calibration spread (a per-agent z-score / conformal surprise):

        z_i      = (q_test_i - μ_cal_i) / (σ_cal_i + ε)
        T_cal_i  = exp(-|z_i|)
        T_i      = (1/|C_i|) · T_cal_i
        score(a) = Σ_i π_i(a) · T_i · 1[a ∈ C_i]

    On a peaked base (σ_cal small) even a small absolute deviation by the
    attacker becomes a large z-score → it is down-weighted.  On a flat base
    (σ_cal large) the honest agents stay near z≈0 while the attacker's
    1-max(π)=0.03 sits many σ below μ_cal → again down-weighted.  The signal
    is therefore scale-free, which is the property a single global method
    needs.

    `per_agent_cal_stats[j] = (μ_cal_j, σ_cal_j)` — mean and std of the
    agent's clean-calibration nonconformity scores.  Falls back to CTC-Global
    when stats are unavailable.
    """
    conf_sets: Dict[int, List[int]] = {}
    trust:     Dict[int, float]     = {}

    for j, p in agent_probs.items():
        cs           = conformal_set(p, q_hat)
        conf_sets[j] = cs
        t_size       = 1.0 / len(cs)

        if per_agent_cal_stats is not None and j in per_agent_cal_stats:
            mu, sigma = per_agent_cal_stats[j]
            q_test    = 1.0 - float(np.max(p))
            z         = (q_test - mu) / (sigma + 1e-8)
            t_cal     = float(np.exp(-abs(z)))
        else:
            t_cal = 1.0

        trust[j] = t_size * t_cal

    n_choices = len(next(iter(agent_probs.values())))
    scores = np.zeros(n_choices, dtype=np.float64)
    for a in range(n_choices):
        for j, p in agent_probs.items():
            if a in conf_sets[j]:
                scores[a] += float(p[a]) * trust[j]

    if scores.max() == 0:
        return ctc_answer(agent_probs, q_hat)
    return int(np.argmax(scores))


def _entropy_np(p: np.ndarray) -> float:
    pc = np.clip(p, 1e-12, None)
    return float(-np.sum(pc * np.log(pc)))


def ctc_adaptive_answer(
    agent_probs: Dict[int, np.ndarray],
    q_hat: float,
    per_agent_profile: Optional[Dict[int, dict]] = None,
    beta: Optional[float] = None,
    tau_frac: float = 0.15,
) -> int:
    """
    UNIFIED / ADAPTIVE CTC — task-agnostic, self-tuning robust trust.

    Why a fixed trust cannot generalise
    -----------------------------------
    The optimal way to treat *confidence* flips between task families:
      • peaked / expert bases (ARC, MMLU): an honest agent that is RIGHT is the
        single most confident member of the committee (H≈0), so a confident-
        correct honest agent should be REWARDED — entropy-trust 1/(H+ε) wins,
        and it survives even a 3/5 coordinated overconfident_extreme attack.
      • flat bases (GSM8K, HellaSwag): honest agents are diffuse, so the most
        confident member is the OVERCONFIDENT ATTACKER — rewarding confidence is
        fatal; confidence must instead be treated as SUSPICIOUS.
    These are opposite, so no fixed function of confidence works everywhere
    (empirically: entropy-trust wins ARC-k3 but →0 on free-form; the symmetric
    anomaly trust wins free-form but →0 on ARC-extreme-k3).

    Two complementary, scale-free signals
    --------------------------------------
    Each agent is profiled on its OWN clean calibration set by the per-question
    distribution of (H = entropy, q = 1-max(π)) → (μ_H, σ_H, μ_q, σ_q).

      (1) Anomaly suppression — ALWAYS safe.  Penalise deviation from the
          agent's own clean profile (a corrupt agent's outputs are atypical):
              anomaly_i = max(|z_H|, |z_q|),  z· = (·_test − μ·)/(σ· + ε)
              T_anom_i  = exp(−anomaly_i)
      (2) Confidence reward — only when confidence is INFORMATIVE.  Reward low
          entropy, but raised to an adaptive exponent β that the committee reads
          off its OWN calibration: peaked base ⇒ β→1 (trust confidence), flat
          base ⇒ β→0 (ignore confidence):
              β = exp(−μ_H^cal / (tau_frac · log n_choices))
          where μ_H^cal is the committee-mean calibration entropy.

        T_i      = (1/|C_i|) · T_anom_i · (1/(H_i+ε))^β
        score(a) = Σ_i π_i(a) · T_i · 1[a ∈ C_i]

    The single rule is fully data-driven: it auto-selects "expert mode" on
    peaked tasks and "skeptic mode" on flat tasks, with no per-task tuning.
    `beta` may be passed explicitly; otherwise it is derived from the profile.
    Falls back to CTC-Global when no profile is available.
    """
    n_choices = len(next(iter(agent_probs.values())))

    if per_agent_profile is None:
        return ctc_answer(agent_probs, q_hat)

    # Adaptive confidence-reward exponent β from the committee's calibration.
    if beta is None:
        mu_H_cal = float(np.mean([per_agent_profile[j]["H"][0]
                                  for j in per_agent_profile]))
        tau = tau_frac * math.log(max(n_choices, 2))
        beta = float(np.exp(-mu_H_cal / (tau + 1e-12)))

    conf_sets: Dict[int, List[int]] = {}
    trust:     Dict[int, float]     = {}
    for j, p in agent_probs.items():
        cs           = conformal_set(p, q_hat)
        conf_sets[j] = cs
        t_size       = 1.0 / len(cs)
        H_test       = _entropy_np(p)

        if j in per_agent_profile:
            muH, sH = per_agent_profile[j]["H"]
            muq, sq = per_agent_profile[j]["q"]
            q_test  = 1.0 - float(np.max(p))
            anomaly = max(abs((H_test - muH) / (sH + 1e-8)),
                          abs((q_test - muq) / (sq + 1e-8)))
            t_anom  = float(np.exp(-anomaly))
        else:
            t_anom = 1.0

        t_conf   = (1.0 / (H_test + 1e-8)) ** beta if beta > 1e-6 else 1.0
        trust[j] = t_size * t_anom * t_conf

    scores = np.zeros(n_choices, dtype=np.float64)
    for a in range(n_choices):
        for j, p in agent_probs.items():
            if a in conf_sets[j]:
                scores[a] += float(p[a]) * trust[j]

    if scores.max() == 0:
        return ctc_answer(agent_probs, q_hat)
    return int(np.argmax(scores))


def ctc_agreement_answer(
    agent_probs: Dict[int, np.ndarray],
    q_hat: float,
) -> int:
    """
    ABLATION: agreement-weighted CTC.

    Trust = (mean Jaccard agreement with peers) × (1 / |C|).
    Agents whose conformal sets overlap with peers' get higher trust.
    """
    conf_sets: Dict[int, set] = {}
    for j, p in agent_probs.items():
        conf_sets[j] = set(conformal_set(p, q_hat))

    # Pairwise Jaccard agreement
    agreement: Dict[int, float] = {}
    agents = list(conf_sets.keys())
    for i in agents:
        scores = []
        for j in agents:
            if i == j:
                continue
            ci, cj = conf_sets[i], conf_sets[j]
            inter = len(ci & cj)
            union = max(len(ci | cj), 1)
            scores.append(inter / union)
        agreement[i] = sum(scores) / max(len(scores), 1)

    trust = {j: agreement[j] / len(conf_sets[j]) for j in agents}

    n_choices = len(next(iter(agent_probs.values())))
    score_vec = np.zeros(n_choices, dtype=np.float64)
    for a in range(n_choices):
        for j, p in agent_probs.items():
            if a in conf_sets[j]:
                score_vec[a] += float(p[a]) * trust[j]

    # Fallback if all scores zero (no agreement at all)
    if score_vec.max() == 0:
        return ctc_answer(agent_probs, q_hat)

    return int(np.argmax(score_vec))


def ctc_answers_all(
    agent_probs: Dict[int, np.ndarray],
    q_hat: float,
) -> Dict[int, int]:
    """
    Compute the CTC answer for *every* agent (each acting as focal).
    Returns dict {agent_id: answer}.  Used by the focal ablation.
    """
    return {j: ctc_focal_answer(agent_probs, q_hat, focal_agent=j)
            for j in agent_probs}
