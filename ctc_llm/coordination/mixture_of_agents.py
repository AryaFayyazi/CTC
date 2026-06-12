"""
Mixture-of-Agents (MoA) baseline — Wang et al. NeurIPS 2024.

Reference
---------
Wang et al. (2024). "Mixture-of-Agents Enhances Large Language Model
Capabilities." NeurIPS 2024.   arXiv:2406.04692
(One of the most cited 2024 multi-agent LLM aggregation methods.)

Key idea
--------
Multi-layer LLM aggregation: in each layer L>1, every agent re-aggregates
from the previous layer's outputs.  In the original paper this requires
actual LLM calls.  We capture the *aggregation rule* without re-querying
LLMs:

  Layer 1 (cached):  πᵢ⁽¹⁾(a | q)   — the model's initial distribution
  Layer 2:           πᵢ⁽²⁾(a | q) = (1-γ) · πᵢ⁽¹⁾ + γ · mean_j πⱼ⁽¹⁾
  Output:            argmax_a   mean_i πᵢ⁽²⁾(a)

where γ ∈ [0,1] is the "blend weight" (γ=0.5 in our experiments).  This
is the linear-time approximation of MoA used in subsequent extensions.

Why it's a fair comparison: it implements MoA's core mechanism
("aggregate then re-aggregate") with the same cached probabilities every
other method uses.  We do not pretend it is the full MoA recipe — we
document this approximation in the paper.
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


def mixture_of_agents_answer(
    agent_probs: Dict[int, np.ndarray],
    n_layers: int = 2,
    blend: float = 0.5,
) -> int:
    """Two-layer (default) MoA aggregation."""
    probs = list(agent_probs.values())
    current = [np.asarray(p, dtype=np.float64) for p in probs]
    n_choices = len(current[0])
    for _ in range(n_layers - 1):
        mean = np.mean(current, axis=0)
        current = [(1 - blend) * p + blend * mean for p in current]
    final = np.mean(current, axis=0)
    return int(np.argmax(final))


def mixture_of_agents_confidence(
    agent_probs: Dict[int, np.ndarray],
    n_layers: int = 2,
    blend: float = 0.5,
) -> Tuple[int, float]:
    """Returns (answer, max-prob of final aggregated distribution)."""
    probs = list(agent_probs.values())
    current = [np.asarray(p, dtype=np.float64) for p in probs]
    for _ in range(n_layers - 1):
        mean = np.mean(current, axis=0)
        current = [(1 - blend) * p + blend * mean for p in current]
    final = np.mean(current, axis=0)
    ans = int(np.argmax(final))
    return ans, float(final[ans])
