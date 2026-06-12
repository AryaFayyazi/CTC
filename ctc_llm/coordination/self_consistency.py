"""
Self-Consistency baseline (Wang et al., 2022).

In the standard self-consistency setup, a *single* model is sampled K times
with temperature > 0 and the most-frequent answer wins.  In our committee
setting we have N persona-distinct agents, but a fair comparison still uses
the SC ranking signal: weight each agent's predicted answer by 1/K (uniform)
and majority-vote, where K = number of distinct answers seen.

To make this a true SC baseline (not equivalent to plurality), we:
  1. Convert each agent's distribution into K samples by drawing from π
     (deterministic seed-controlled, so it's reproducible from cached probs).
  2. Pool all K · N samples and majority-vote.

For test-time efficiency we use K=10 samples per agent.  When all agents are
near-deterministic (top-1 ≈ 1.0), this reduces to plurality vote — which is
the standard SC failure mode on greedy-decoded LLMs and a real finding.

References
----------
Wang et al. (2022). Self-Consistency Improves Chain of Thought Reasoning in
Language Models. ICLR 2023.
"""

from __future__ import annotations
from collections import Counter
from typing import Dict, Tuple

import numpy as np


def self_consistency_answer(
    agent_probs: Dict[int, np.ndarray],
    n_samples_per_agent: int = 10,
    seed: int = 0,
) -> int:
    """
    Returns the SC majority-vote answer.

        Sample K answers per agent ∼ πᵢ
        Pool all K · N samples
        Return mode.

    Deterministic given (agent_probs, seed).
    """
    rng = np.random.RandomState(seed)
    votes = Counter()
    for j, p in agent_probs.items():
        p_safe = np.clip(p, 1e-12, None)
        p_safe = p_safe / p_safe.sum()
        samples = rng.choice(len(p_safe), size=n_samples_per_agent, p=p_safe)
        votes.update(int(s) for s in samples)
    return int(votes.most_common(1)[0][0])


def self_consistency_confidence(
    agent_probs: Dict[int, np.ndarray],
    n_samples_per_agent: int = 10,
    seed: int = 0,
) -> Tuple[int, float]:
    """Answer + confidence = (mode count) / (total samples)."""
    rng = np.random.RandomState(seed)
    votes = Counter()
    total = 0
    for j, p in agent_probs.items():
        p_safe = np.clip(p, 1e-12, None)
        p_safe = p_safe / p_safe.sum()
        samples = rng.choice(len(p_safe), size=n_samples_per_agent, p=p_safe)
        votes.update(int(s) for s in samples)
        total += n_samples_per_agent
    ans, count = votes.most_common(1)[0]
    return int(ans), count / max(total, 1)
