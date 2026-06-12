"""
Claude API agent with disk-based caching.

Each agent has a unique persona (system prompt) so the committee of N agents
produces diverse probability distributions — mimicking a real multi-LLM system
where different agents have different roles / specialisations.

Caching strategy
----------------
Every (agent_id, question, choices) triple is hashed and the API response
stored in data/cache/{hash}.json.  Re-running never pays for the same call twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import List, Optional

import numpy as np

from .base import Agent


# ── Agent personas ────────────────────────────────────────────────────────────

AGENT_PERSONAS: List[str] = [
    (
        "You are Agent-Alpha, a meticulous academic reviewer. "
        "You carefully weigh each answer option and assign calibrated probabilities "
        "that reflect your genuine uncertainty."
    ),
    (
        "You are Agent-Beta, a domain expert with deep subject knowledge. "
        "You assign high confidence to answers you are certain about and spread "
        "probability appropriately when unsure."
    ),
    (
        "You are Agent-Gamma, a systematic analyst. "
        "You eliminate clearly wrong options first, then distribute probability "
        "among the remaining plausible answers based on your reasoning."
    ),
    (
        "You are Agent-Delta, a rigorous scholar trained across many disciplines. "
        "You express honest uncertainty through your probability estimates — "
        "never overconfident, never underconfident."
    ),
    (
        "You are Agent-Epsilon, a critical thinker who challenges assumptions. "
        "You provide well-reasoned probability distributions that reflect "
        "your careful evaluation of all answer choices."
    ),
]

_PROB_PROMPT_TEMPLATE = """\
Answer the following multiple-choice question by providing a probability \
distribution over the four choices. Your probabilities must be non-negative \
and sum to exactly 1.0.

Question: {question}

A) {A}
B) {B}
C) {C}
D) {D}

Respond with ONLY a valid JSON object and nothing else:
{{"A": <float>, "B": <float>, "C": <float>, "D": <float>}}"""


class ClaudeAgent(Agent):
    """
    Clean LLM agent backed by the Claude Messages API.

    Parameters
    ----------
    agent_id : int
        Determines which persona (system prompt) this agent uses.
    model : str
        Anthropic model name.  Defaults to Haiku for cost efficiency.
    cache_dir : str
        Directory where JSON response caches are stored.
    temperature : float
        Sampling temperature.  0 = deterministic (reproducible).
    """

    def __init__(
        self,
        agent_id: int,
        model: str = "claude-haiku-4-5-20251001",
        cache_dir: str = "data/cache",
        temperature: float = 0.0,
    ):
        import anthropic  # lazy import so non-API code paths don't require it
        self.agent_id    = agent_id
        self.model       = model
        self.cache_dir   = cache_dir
        self.temperature = temperature
        self.system      = AGENT_PERSONAS[agent_id % len(AGENT_PERSONAS)]
        self._client     = anthropic.Anthropic()
        os.makedirs(cache_dir, exist_ok=True)

    # ── Public interface ──────────────────────────────────────────────────────

    def get_probs(self, question: str, choices: List[str]) -> np.ndarray:
        if len(choices) != 4:
            raise ValueError("Only 4-choice questions are supported.")

        key  = self._cache_key(question, choices)
        path = os.path.join(self.cache_dir, f"{key}.json")

        if os.path.exists(path):
            with open(path) as f:
                return np.array(json.load(f), dtype=np.float32)

        probs = self._call_api(question, choices)
        with open(path, "w") as f:
            json.dump(probs.tolist(), f)
        return probs

    # ── Internals ─────────────────────────────────────────────────────────────

    def _cache_key(self, question: str, choices: List[str]) -> str:
        blob = json.dumps({
            "agent_id": self.agent_id,
            "model":    self.model,
            "system":   self.system,
            "question": question,
            "choices":  choices,
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def _build_prompt(self, question: str, choices: List[str]) -> str:
        return _PROB_PROMPT_TEMPLATE.format(
            question=question,
            A=choices[0], B=choices[1], C=choices[2], D=choices[3],
        )

    def _call_api(self, question: str, choices: List[str]) -> np.ndarray:
        prompt = self._build_prompt(question, choices)
        for attempt in range(4):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=80,
                    temperature=self.temperature,
                    system=self.system,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text.strip()
                return self._parse_probs(text)
            except Exception as exc:
                if attempt == 3:
                    # Fallback: uniform distribution rather than crashing
                    return np.ones(4, dtype=np.float32) / 4
                wait = 2 ** attempt
                time.sleep(wait)
        return np.ones(4, dtype=np.float32) / 4

    def _parse_probs(self, text: str) -> np.ndarray:
        # Extract JSON blob (may be wrapped in markdown code fences)
        m = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if not m:
            return np.ones(4, dtype=np.float32) / 4
        try:
            d = json.loads(m.group())
            probs = np.array([
                float(d.get("A", 0.25)),
                float(d.get("B", 0.25)),
                float(d.get("C", 0.25)),
                float(d.get("D", 0.25)),
            ], dtype=np.float32)
            probs = np.clip(probs, 1e-6, None)
            return probs / probs.sum()
        except Exception:
            return np.ones(4, dtype=np.float32) / 4
