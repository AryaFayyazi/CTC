"""Abstract agent interface for LLM-based multi-agent coordination."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np


class Agent(ABC):
    """
    Base class for all agents (clean or corrupt).

    Every agent exposes one method:
        get_probs(question, choices) -> np.ndarray of shape (len(choices),)

    The returned array is a valid probability distribution (non-negative, sums to 1).
    """

    @abstractmethod
    def get_probs(self, question: str, choices: List[str]) -> np.ndarray:
        """Return probability distribution over choices."""

    def get_answer(self, question: str, choices: List[str]) -> int:
        """Return the most likely choice index (argmax of probs)."""
        return int(np.argmax(self.get_probs(question, choices)))
