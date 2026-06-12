"""
Majority voting (plurality vote on each agent's top-1 answer).

Equivalent to self-consistency (Wang et al. 2022) without chain-of-thought.
Robust to random corrupt agents but fails when corrupt agents are a majority.
"""
from typing import Dict
import numpy as np
from collections import Counter


def majority_answer(agent_probs: Dict[int, np.ndarray]) -> int:
    """Plurality vote over each agent's argmax answer."""
    votes = [int(np.argmax(p)) for p in agent_probs.values()]
    return Counter(votes).most_common(1)[0][0]
