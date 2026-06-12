"""
Corrupt (adversarial) agent wrappers.

All attacks are *programmatic* — no API calls required.  This keeps
experiments cheap and the threat model explicit and reproducible.

Three attack types
------------------
overconfident  (main attack)
    Places 0.97 probability on the *worst* answer (the one with lowest
    probability under the clean policy).  Models a prompt-injection attack
    that makes an agent extremely confident in the wrong answer.
    Defeats entropy-trust (low entropy → high trust) but NOT CTC
    (clean agents' high own-probability neutralises the wrong endorsement).

random
    Returns a uniform distribution.  Models a faulty / unresponsive agent.
    Easy for all methods to handle.

subtle
    Uses the clean agent's distribution but swaps probability mass so the
    correct answer is ranked last.  Models a sophisticated adversary that
    sounds plausible but is systematically wrong.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import Agent


class OverconfidentWrongAgent(Agent):
    """
    0.97 probability on the worst answer, 0.01 spread over the rest.

    If `correct_idx` is provided (adaptive attack), the worst answer is
    always the answer that is NOT correct.  Without it, worst = argmin(base_probs).
    """

    def __init__(
        self,
        base_agent: Agent,
        correct_idx: Optional[int] = None,
        wrong_prob: float = 0.97,
    ):
        self.base        = base_agent
        self.correct_idx = correct_idx
        self.wrong_prob  = wrong_prob

    def get_probs(self, question: str, choices: List[str]) -> np.ndarray:
        n = len(choices)
        base_p = self.base.get_probs(question, choices)

        if self.correct_idx is not None:
            # Adaptive: pick worst answer that is NOT the correct one
            masked = base_p.copy()
            masked[self.correct_idx] = np.inf   # exclude correct
            worst = int(np.argmin(masked))
        else:
            worst = int(np.argmin(base_p))

        q = np.full(n, (1.0 - self.wrong_prob) / (n - 1), dtype=np.float32)
        q[worst] = self.wrong_prob
        return q


class RandomAgent(Agent):
    """Uniform distribution — models a faulty / non-responsive agent."""

    def get_probs(self, question: str, choices: List[str]) -> np.ndarray:
        n = len(choices)
        return np.ones(n, dtype=np.float32) / n


class CoordinatedExtremeAgent(Agent):
    """
    Extreme coordinated attack: all corrupt agents output ~1.0 probability
    on the SAME wrong answer.  This is the most damaging realistic attack
    against multi-agent committees:

      - Defeats Majority Vote when k > N/2  (3 vs 2 cleans)
      - Defeats Avg-Ensemble (3 × 1.0 > 2 × clean_prob)
      - Defeats EntropyTrust (entropy ≈ 0 = max trust, all on wrong)
      - Defeats CTC-Focal (focal often picked from corrupt singleton)

    Coordination assumption: attackers collude by attacking the same target.
    The target is chosen deterministically: the lexically-first non-correct
    answer index — i.e. all corrupt agents pick the SAME wrong answer.
    """

    def __init__(self, base_agent: Agent, correct_idx: Optional[int] = None,
                 wrong_prob: float = 1.0 - 1e-4):
        self.base        = base_agent
        self.correct_idx = correct_idx
        self.wrong_prob  = wrong_prob

    def get_probs(self, question: str, choices: List[str]) -> np.ndarray:
        n = len(choices)
        # Deterministic target: lowest-index wrong answer (all corrupts agree)
        target = 0 if self.correct_idx != 0 else 1
        q = np.full(n, (1.0 - self.wrong_prob) / (n - 1), dtype=np.float32)
        q[target] = self.wrong_prob
        return q


class SubtleWrongAgent(Agent):
    """
    Inverts the base distribution so the correct answer becomes the least
    probable.  Models a sophisticated adversary that sounds confident but
    is systematically wrong.
    """

    def __init__(self, base_agent: Agent, correct_idx: Optional[int] = None):
        self.base        = base_agent
        self.correct_idx = correct_idx

    def get_probs(self, question: str, choices: List[str]) -> np.ndarray:
        p = self.base.get_probs(question, choices).copy()
        # Flip: 1 - p, then re-normalise
        q = 1.0 - p
        q = np.clip(q, 1e-6, None)
        return (q / q.sum()).astype(np.float32)


def make_corrupt_agent(
    attack: str,
    base_agent: Agent,
    correct_idx: Optional[int] = None,
    agent_id: int = 0,
    model_id: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    result_cache_dir: Optional[str] = None,
) -> Agent:
    """
    Factory for corrupt agents.

    Programmatic attacks (no LLM call):
        overconfident         (0.97 mass on wrong — naive prompt injection)
        overconfident_extreme (1 - 1e-4 mass — extreme attack that beats EntropyTrust)
        coordinated_extreme   (1 - 1e-4 on same wrong answer for all corrupts,
                               most damaging realistic attack)
        random                (uniform, weak attack)
        subtle                (invert clean distribution)

    Real prompt-injection attacks (cached LLM forward pass):
        inject_sycophant, inject_deceptive
    """
    if attack == "overconfident":
        return OverconfidentWrongAgent(base_agent, correct_idx=correct_idx, wrong_prob=0.97)
    elif attack == "overconfident_extreme":
        return OverconfidentWrongAgent(base_agent, correct_idx=correct_idx, wrong_prob=1.0 - 1e-4)
    elif attack == "coordinated_extreme":
        return CoordinatedExtremeAgent(base_agent, correct_idx=correct_idx)
    elif attack == "random":
        return RandomAgent()
    elif attack == "subtle":
        return SubtleWrongAgent(base_agent, correct_idx=correct_idx)
    elif attack in ("inject_sycophant", "inject_deceptive"):
        # Lazy import to avoid loading model unless needed
        from .prompt_inject_agent import PromptInjectionAgent
        kind = attack.split("_", 1)[1]   # "sycophant" or "deceptive"
        sub  = "sycophant_wrong" if kind == "sycophant" else "deceptive"
        kwargs = {}
        if model_id is not None:         kwargs["model_id"] = model_id
        if hf_cache_dir is not None:     kwargs["hf_cache_dir"] = hf_cache_dir
        if result_cache_dir is not None: kwargs["result_cache_dir"] = result_cache_dir
        return PromptInjectionAgent(agent_id, attack=sub, **kwargs)
    else:
        raise ValueError(f"Unknown attack type: {attack!r}")
