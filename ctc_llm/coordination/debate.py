"""
Multi-Agent Debate baseline (Du et al., 2023; simplified single-round).

Original Du et al. (2023) "Improving Factuality and Reasoning in Language
Models through Multiagent Debate" runs multiple rounds of agents reading
each other's answers and updating.  We approximate one round here from
cached probabilities:

  1. Each agent emits πᵢ (initial belief, cached from one LLM forward pass).
  2. Agent i sees the *modal answer* of the committee {argmax πⱼ : j≠i}.
  3. Each agent updates: π'ᵢ = β · πᵢ + (1−β) · δ(committee_mode)
     where β = 0.5 (anchoring weight to own belief).
  4. Final answer = argmax over the trust-weighted sum of π'.

This captures debate's core mechanism — "see others, update belief, vote" —
without needing multi-round LLM inference.

References
----------
Du et al. (2023). Improving Factuality and Reasoning in Language Models
through Multiagent Debate. ICML 2024.
"""

from __future__ import annotations
from collections import Counter
from typing import Dict, Tuple

import numpy as np


def debate_answer(
    agent_probs: Dict[int, np.ndarray],
    anchor_weight: float = 0.5,
) -> int:
    """
    Single-round debate aggregation:

        committee_mode = mode( argmax πⱼ )
        π'ᵢ = β · πᵢ + (1−β) · δ(committee_mode)
        answer = argmax_a Σᵢ π'ᵢ(a)
    """
    # 1) Each agent's top-1 vote → committee mode
    top1 = [int(np.argmax(p)) for p in agent_probs.values()]
    committee_mode, _ = Counter(top1).most_common(1)[0]

    # 2) Each agent updates toward the committee mode
    n_choices = len(next(iter(agent_probs.values())))
    summed = np.zeros(n_choices, dtype=np.float64)
    for p in agent_probs.values():
        delta = np.zeros(n_choices)
        delta[committee_mode] = 1.0
        p_updated = anchor_weight * p + (1.0 - anchor_weight) * delta
        summed += p_updated

    return int(np.argmax(summed))


def debate_confidence(
    agent_probs: Dict[int, np.ndarray],
    anchor_weight: float = 0.5,
) -> Tuple[int, float]:
    """Returns (answer, normalized score for chosen answer)."""
    top1 = [int(np.argmax(p)) for p in agent_probs.values()]
    committee_mode, _ = Counter(top1).most_common(1)[0]

    n_choices = len(next(iter(agent_probs.values())))
    summed = np.zeros(n_choices, dtype=np.float64)
    for p in agent_probs.values():
        delta = np.zeros(n_choices); delta[committee_mode] = 1.0
        p_updated = anchor_weight * p + (1.0 - anchor_weight) * delta
        summed += p_updated
    summed /= summed.sum()
    ans = int(np.argmax(summed))
    return ans, float(summed[ans])
