"""Empirical coverage and set-size metrics for conformal prediction."""

from __future__ import annotations
from typing import List
import numpy as np
from .calibrate import conformal_set


def empirical_coverage(
    probs_list: List[np.ndarray],
    correct_indices: List[int],
    q_hat: float,
) -> float:
    """Fraction of test examples where the correct answer is in the conformal set."""
    covered = sum(
        correct_indices[i] in conformal_set(probs_list[i], q_hat)
        for i in range(len(probs_list))
    )
    return covered / len(probs_list)


def mean_set_size(probs_list: List[np.ndarray], q_hat: float) -> float:
    """Average conformal set size (smaller = more informative)."""
    return float(np.mean([len(conformal_set(p, q_hat)) for p in probs_list]))
