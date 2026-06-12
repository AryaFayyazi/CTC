"""Vanilla: single-agent baseline (no coordination)."""
from typing import Dict
import numpy as np


def vanilla_answer(agent_probs: Dict[int, np.ndarray], focal_agent: int = 0) -> int:
    """Return the argmax of one agent's own distribution (no coordination)."""
    return int(np.argmax(agent_probs[focal_agent]))
