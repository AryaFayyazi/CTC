"""
Equal-weight ensemble (average of all agent distributions).

Simple and strong baseline — equivalent to Bayesian model averaging under
uniform agent priors.
"""
from typing import Dict
import numpy as np


def average_answer(agent_probs: Dict[int, np.ndarray]) -> int:
    """Argmax of the mean probability distribution across all agents."""
    stacked = np.stack(list(agent_probs.values()))
    return int(np.argmax(stacked.mean(axis=0)))
