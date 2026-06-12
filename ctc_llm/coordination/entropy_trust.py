"""
Entropy-weighted trust voting.

Each agent's vote is weighted by 1 / H(π_j) where H is Shannon entropy.
Low-entropy (confident) agents dominate.

Vulnerability: an overconfident *wrong* agent has H ≈ 0 → weight → ∞,
completely dominating the vote.  CTC is immune to this.
"""
from typing import Dict
import numpy as np


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, None)
    return float(-np.sum(p * np.log(p)))


def entropy_trust_answer(agent_probs: Dict[int, np.ndarray]) -> int:
    """Weighted sum of distributions, weights = 1/H(π_j), take argmax."""
    weighted = np.zeros(len(next(iter(agent_probs.values()))), dtype=np.float64)
    for p in agent_probs.values():
        h = _entropy(p)
        w = 1.0 / (h + 1e-8)
        weighted += w * p
    return int(np.argmax(weighted))
