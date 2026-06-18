"""
Conformal calibration for free-form (Dict-based) agent distributions.

Nonconformity score: s_i(q) = 1 - p_i(correct_answer)
Conformal set:       C_i = {a : p_i(a) >= 1 - q_hat}

All formulas are identical to the MCQ case — the only difference is that
the distribution is over arbitrary strings instead of exactly 4 letters.
"""

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np


AgentDist = Dict[str, float]


def freeform_calibrate(
    agent_dists: List[AgentDist],
    correct_answers: List[str],
    alpha: float = 0.10,
) -> float:
    """
    Split-conformal calibration on free-form distributions.

    agent_dists[i] = probability distribution for question i
                     (pooled over all agents for that question)
    correct_answers[i] = canonical correct answer string for question i

    Returns q_hat such that empirical coverage >= 1 - alpha.
    """
    scores: List[float] = []
    for dist, ans in zip(agent_dists, correct_answers):
        p_correct = dist.get(ans.strip(), 0.0)
        scores.append(1.0 - p_correct)

    n = len(scores)
    if n == 0:
        return 1.0
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = float(np.clip(level, 0.0, 1.0))
    return float(np.quantile(scores, level))


def freeform_conformal_set(dist: AgentDist, q_hat: float) -> Set[str]:
    """Return {a : p(a) >= 1 - q_hat}."""
    threshold = 1.0 - q_hat
    return {a for a, p in dist.items() if p >= threshold}
