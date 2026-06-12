"""
Conformal calibration for LLM agent trust.

Given a calibration set of (agent_probs, correct_answer) pairs, compute the
conformal threshold q̂ such that:

    P(correct_answer ∈ conformal_set(q)) ≥ 1 − α

This is the standard split-conformal guarantee (Vovk et al. 2005,
Angelopoulos & Bates 2023).

Nonconformity score
-------------------
    s(q, a*) = 1 − π(a* | q)

where π(a* | q) is the agent's probability for the correct answer a*.
A low score means the agent was confident on the right answer (conforming).
A high score means it was surprised (nonconforming).

Conformal set at test time
--------------------------
    C(q) = { a : 1 − π(a | q) ≤ q̂ }
         = { a : π(a | q) ≥ 1 − q̂ }

Trust score (used by CTC)
-------------------------
    T = 1 / max(1, |C(q)|)

A small conformal set → high trust.  A corrupt overconfident agent gets a
small set on the *wrong* action, but CTC's own-policy gate neutralises it.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def compute_nonconformity_scores(
    probs_list: List[np.ndarray],
    correct_indices: List[int],
) -> np.ndarray:
    """
    Compute nonconformity scores for a calibration set.

    Parameters
    ----------
    probs_list : list of 1-D arrays of length n_choices
    correct_indices : list of correct answer indices (0-indexed)

    Returns
    -------
    scores : np.ndarray of shape (len(probs_list),), values in [0, 1]
    """
    scores = np.array(
        [1.0 - float(p[a]) for p, a in zip(probs_list, correct_indices)],
        dtype=np.float64,
    )
    return scores


def calibrate(
    probs_list: List[np.ndarray],
    correct_indices: List[int],
    alpha: float = 0.10,
) -> Tuple[np.ndarray, float]:
    """
    Compute the conformal threshold q̂.

    Uses the finite-sample corrected quantile:
        q̂ = Quantile_{⌈(n+1)(1−α)/n⌉}(scores)

    which guarantees P(a* ∈ C) ≥ 1−α in expectation over the calibration data.

    Returns
    -------
    scores  : nonconformity scores for each calibration example
    q_hat   : conformal threshold
    """
    scores = compute_nonconformity_scores(probs_list, correct_indices)
    n      = len(scores)
    # Finite-sample correction: ceiling of (n+1)(1-α)/n quantile
    level  = math.ceil((n + 1) * (1.0 - alpha)) / n
    level  = min(level, 1.0)
    q_hat  = float(np.quantile(scores, level, method="higher"))
    return scores, q_hat


def conformal_set(probs: np.ndarray, q_hat: float) -> List[int]:
    """
    Return the conformal set for a single prediction.

    C = {a : π(a) ≥ 1 − q̂}

    If the set is empty (agent is very uncertain with q̂ < 1 − max(π)),
    fall back to the top-1 action to avoid division by zero in trust scores.
    """
    threshold = 1.0 - q_hat
    cs = [a for a in range(len(probs)) if probs[a] >= threshold]
    if not cs:
        cs = [int(np.argmax(probs))]
    return cs


def trust_score(probs: np.ndarray, q_hat: float) -> float:
    """T = 1 / |C|.  Smaller set → higher trust."""
    return 1.0 / len(conformal_set(probs, q_hat))
