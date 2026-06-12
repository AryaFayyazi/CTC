"""
LLM-as-Judge baseline.

Used across many 2024-2025 LLM multi-agent papers; perhaps the most
common alternative to majority vote in LLM-judge style evaluations.

Refs
----
Zheng et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot
Arena."  NeurIPS Datasets & Benchmarks.   (foundational)

Aggregation rule
----------------
Designate the lowest-id (most "central") agent as the *judge*.  Score each
candidate action by the judge's own probability for it, multiplied by the
number of OTHER agents whose top-1 vote matches:

    score(a) = π_judge(a) * | { j ≠ judge : argmax πⱼ = a } |

answer = argmax score.

This captures the spirit: a single judge weighing votes from the rest by
its own belief.
"""

from __future__ import annotations
from collections import Counter
from typing import Dict, Tuple
import numpy as np


def llm_judge_answer(
    agent_probs: Dict[int, np.ndarray],
    judge_id: int = 0,
) -> int:
    if judge_id not in agent_probs:
        judge_id = min(agent_probs.keys())
    judge = agent_probs[judge_id]
    n_choices = len(judge)

    # Top-1 vote count from non-judge agents
    votes = Counter()
    for j, p in agent_probs.items():
        if j == judge_id:
            continue
        votes[int(np.argmax(p))] += 1

    scores = np.zeros(n_choices)
    for a in range(n_choices):
        scores[a] = float(judge[a]) * votes.get(a, 0)

    if scores.sum() == 0:
        # fall back to judge's own top-1 if nobody agrees
        return int(np.argmax(judge))
    return int(np.argmax(scores))


def llm_judge_confidence(
    agent_probs: Dict[int, np.ndarray],
    judge_id: int = 0,
) -> Tuple[int, float]:
    if judge_id not in agent_probs:
        judge_id = min(agent_probs.keys())
    judge = agent_probs[judge_id]
    n_choices = len(judge)
    votes = Counter()
    for j, p in agent_probs.items():
        if j == judge_id:
            continue
        votes[int(np.argmax(p))] += 1
    scores = np.zeros(n_choices)
    for a in range(n_choices):
        scores[a] = float(judge[a]) * votes.get(a, 0)
    if scores.sum() == 0:
        ans = int(np.argmax(judge))
        return ans, float(judge[ans])
    total = scores.sum()
    ans = int(np.argmax(scores))
    return ans, float(scores[ans] / total)
