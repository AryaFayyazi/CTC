"""
Per-method confidence scores.

Used for selective prediction: each method produces a confidence score in [0,1]
for its chosen answer.  Sweeping a threshold τ yields a risk-coverage curve.
A method "abstains" on questions where confidence < τ.

For CTC variants, we provide TWO natural abstention signals:
  (i)  the trust-weighted score for the chosen answer (continuous, [0,∞))
  (ii) the conformal set size |C| — abstain when |C| > τ_size  (discrete)

The conformal-set-size signal is special: it inherits a marginal coverage
guarantee from the conformal calibration, so abstention thresholds have
formal meaning, unlike entropy/probability thresholds.
"""

from __future__ import annotations
from collections import Counter
from typing import Dict, Tuple

import numpy as np

from ctc_llm.conformal.calibrate import conformal_set


# ── Confidences (one per method) ─────────────────────────────────────────────

def vanilla_confidence(agent_probs: Dict[int, np.ndarray], focal: int = 0) -> Tuple[int, float]:
    p = agent_probs[focal]
    a = int(np.argmax(p))
    return a, float(p[a])


def average_confidence(agent_probs: Dict[int, np.ndarray]) -> Tuple[int, float]:
    mean = np.mean(list(agent_probs.values()), axis=0)
    a = int(np.argmax(mean))
    return a, float(mean[a])


def majority_confidence(agent_probs: Dict[int, np.ndarray]) -> Tuple[int, float]:
    votes = Counter(int(np.argmax(p)) for p in agent_probs.values())
    a, c  = votes.most_common(1)[0]
    return a, float(c) / len(agent_probs)


def entropy_confidence(agent_probs: Dict[int, np.ndarray]) -> Tuple[int, float]:
    n_choices = len(next(iter(agent_probs.values())))
    weighted  = np.zeros(n_choices, dtype=np.float64)
    weights   = []
    for p in agent_probs.values():
        h = float(-np.sum(np.clip(p, 1e-12, None) * np.log(np.clip(p, 1e-12, None))))
        w = 1.0 / (h + 1e-8)
        weighted += w * p
        weights.append(w)
    weighted /= sum(weights)
    a = int(np.argmax(weighted))
    return a, float(weighted[a])


def ctc_confidence(agent_probs: Dict[int, np.ndarray], q_hat: float) -> Tuple[int, float, int]:
    """
    Returns (answer, normalized_score, set_size_at_focal_agent).

    set_size is the agent-0 conformal set size — the conformal-blessed
    abstention signal (|C| > 1 = abstain).
    """
    conf_sets = {j: conformal_set(p, q_hat) for j, p in agent_probs.items()}
    trust     = {j: 1.0 / len(conf_sets[j]) for j in agent_probs}
    n_choices = len(next(iter(agent_probs.values())))
    scores    = np.zeros(n_choices, dtype=np.float64)
    for a in range(n_choices):
        for j, p in agent_probs.items():
            if a in conf_sets[j]:
                scores[a] += float(p[a]) * trust[j]
    total = scores.sum()
    if total == 0:
        a = int(np.argmax(np.mean(list(agent_probs.values()), axis=0)))
        return a, 0.0, len(conf_sets[0])
    a = int(np.argmax(scores))
    return a, float(scores[a] / total), len(conf_sets[0])


def ctc_hybrid_confidence(agent_probs: Dict[int, np.ndarray], q_hat: float) -> Tuple[int, float, int]:
    """Same as ctc_confidence but using the hybrid trust (T_size × 1/H)."""
    conf_sets = {j: conformal_set(p, q_hat) for j, p in agent_probs.items()}
    trust = {}
    for j, p in agent_probs.items():
        t_size = 1.0 / len(conf_sets[j])
        h      = float(-np.sum(np.clip(p, 1e-12, None) * np.log(np.clip(p, 1e-12, None))))
        trust[j] = t_size * (1.0 / (h + 1e-8))
    n_choices = len(next(iter(agent_probs.values())))
    scores    = np.zeros(n_choices, dtype=np.float64)
    for a in range(n_choices):
        for j, p in agent_probs.items():
            if a in conf_sets[j]:
                scores[a] += float(p[a]) * trust[j]
    total = scores.sum()
    if total == 0:
        a = int(np.argmax(np.mean(list(agent_probs.values()), axis=0)))
        return a, 0.0, len(conf_sets[0])
    a = int(np.argmax(scores))
    return a, float(scores[a] / total), len(conf_sets[0])


# ── Committee-level abstention (the headline novel signal) ───────────────────

def committee_conformal_abstain(
    agent_probs: Dict[int, np.ndarray],
    q_hat: float,
) -> Tuple[int, int, float]:
    """
    SELECTIVE PREDICTION SIGNAL for decentralised LLM committees.

    Returns (answer, committee_set_size, score_concentration).

      answer : argmax of CTC-Hybrid score
      committee_set_size : |⋃_i C_i| — size of the union of all agents'
        conformal sets.  This is the COMMITTEE-WIDE uncertainty set.
        - Size 1 = perfect committee consensus (all agents' sets are a
          singleton on the same action) → high confidence
        - Size > 1 = committee disagrees on which actions are plausible
          → abstain or escalate to human review
      score_concentration : top1 / total of CTC-Hybrid score; a tie-breaker
        for selective ranking at the same set size.

    THEORETICAL GUARANTEE  (informal):
      If we predict only when committee_set_size == 1, then conditional on
      prediction, the chosen action is the unique action that ALL agents
      include in their conformal sets.  By marginal conformal coverage,
      each clean agent's set covers the true answer with prob ≥ 1-α; by
      union bound and intersection, the committee's singleton set covers
      the true answer with prob ≥ 1 - N·α  (loose) or better with
      independence-of-errors assumptions.
    """
    conf_sets = {j: conformal_set(p, q_hat) for j, p in agent_probs.items()}
    union     = set()
    for cs in conf_sets.values():
        union.update(cs)

    # Use CTC-Hybrid for the answer choice
    trust = {}
    for j, p in agent_probs.items():
        t_size = 1.0 / len(conf_sets[j])
        h      = float(-np.sum(np.clip(p, 1e-12, None) * np.log(np.clip(p, 1e-12, None))))
        trust[j] = t_size * (1.0 / (h + 1e-8))
    n_choices = len(next(iter(agent_probs.values())))
    scores    = np.zeros(n_choices, dtype=np.float64)
    for a in range(n_choices):
        for j, p in agent_probs.items():
            if a in conf_sets[j]:
                scores[a] += float(p[a]) * trust[j]
    total = scores.sum()
    if total == 0:
        a = int(np.argmax(np.mean(list(agent_probs.values()), axis=0)))
        return a, len(union), 0.0
    a = int(np.argmax(scores))
    return a, len(union), float(scores[a] / total)
