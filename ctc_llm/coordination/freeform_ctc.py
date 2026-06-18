"""
CTC coordination methods for free-form (Dict-based) agent distributions.

All methods operate on List[Dict[str, float]] (one distribution per agent)
and return the predicted answer string.

Methods implemented
-------------------
freeform_vanilla        argmax of the distribution of agent 0 (single-agent)
freeform_average        argmax of the mean distribution (ensemble)
freeform_majority       plurality vote on each agent's argmax
freeform_entropy_trust  weighted by 1 / (H(dist) + eps)
freeform_ctc_hybrid     CTC-Hybrid: conformal trust × entropy-based weight
                        (main method — identical formula to MCQ, string keys)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ctc_llm.conformal.freeform_calibrate import freeform_conformal_set


AgentDist = Dict[str, float]
_EPS = 1e-8


def _entropy(dist: AgentDist) -> float:
    """Shannon entropy in nats."""
    return -sum(p * math.log(p + _EPS) for p in dist.values() if p > 0)


def _argmax(dist: AgentDist) -> str:
    return max(dist, key=dist.get)


def _all_candidates(agent_dists: List[AgentDist]) -> Set[str]:
    return {a for d in agent_dists for a in d}


# ── Baseline methods ──────────────────────────────────────────────────────────

def freeform_vanilla(agent_dists: List[AgentDist]) -> str:
    """Single-agent baseline: argmax of agent 0's distribution."""
    return _argmax(agent_dists[0])


def freeform_average(agent_dists: List[AgentDist]) -> str:
    """Average ensemble: argmax of mean distribution over all agents."""
    candidates = _all_candidates(agent_dists)
    mean_p = {a: sum(d.get(a, 0.0) for d in agent_dists) / len(agent_dists)
              for a in candidates}
    return _argmax(mean_p)


def freeform_majority(agent_dists: List[AgentDist]) -> str:
    """Plurality vote on each agent's most probable answer."""
    votes: Dict[str, int] = {}
    for d in agent_dists:
        winner = _argmax(d)
        votes[winner] = votes.get(winner, 0) + 1
    return max(votes, key=votes.get)


def freeform_entropy_trust(agent_dists: List[AgentDist]) -> str:
    """
    Entropy-trust weighted vote.

    Weight of agent i = 1 / (H(p_i) + eps) — low entropy ↔ high trust.
    Vulnerable to overconfident attacks (exactly like MCQ entropy-trust).
    """
    candidates = _all_candidates(agent_dists)
    weighted: Dict[str, float] = {a: 0.0 for a in candidates}
    for d in agent_dists:
        w = 1.0 / (_entropy(d) + _EPS)
        for a, p in d.items():
            weighted[a] = weighted.get(a, 0.0) + w * p
    return _argmax(weighted)


# ── CTC-Hybrid (main method) ──────────────────────────────────────────────────

def freeform_ctc_hybrid(
    agent_dists: List[AgentDist],
    q_hat: float,
) -> str:
    """
    CTC-Hybrid for free-form distributions.

    score(a) = Σ_i  p_i(a) · (1/|C_i|) · (1/(H(p_i)+ε)) · 1[a ∈ C_i]

    This is IDENTICAL to the MCQ formula — the only change is that a
    ranges over arbitrary strings rather than exactly {A,B,C,D}.
    """
    candidates = _all_candidates(agent_dists)
    scores: Dict[str, float] = {a: 0.0 for a in candidates}

    for d in agent_dists:
        conf_set = freeform_conformal_set(d, q_hat)
        if not conf_set:
            continue
        set_size = len(conf_set)
        h_trust  = 1.0 / (_entropy(d) + _EPS)
        for a in conf_set:
            p_ia = d.get(a, 0.0)
            scores[a] = scores.get(a, 0.0) + p_ia * (1.0 / set_size) * h_trust

    if not any(v > 0 for v in scores.values()):
        return freeform_average(agent_dists)
    return _argmax(scores)


# ── Committee-conformal abstention ────────────────────────────────────────────

def freeform_committee_abstain(
    agent_dists: List[AgentDist],
    q_hat: float,
) -> Tuple[str, int, float]:
    """
    Committee-conformal abstention for free-form distributions.

    Returns (predicted_answer, union_set_size, score_concentration).
    Abstain when union_set_size > 1 (unanimous agreement on a single answer).
    """
    from functools import reduce
    union: Set[str] = reduce(
        lambda s, d: s | freeform_conformal_set(d, q_hat),
        agent_dists,
        set(),
    )
    pred  = freeform_ctc_hybrid(agent_dists, q_hat)
    total = sum(
        sum(d.get(a, 0.0) for d in agent_dists)
        for a in union
    )
    if total > 0:
        pred_mass = sum(d.get(pred, 0.0) for d in agent_dists)
        concentration = pred_mass / (total + _EPS)
    else:
        concentration = 0.0
    return pred, len(union), concentration
