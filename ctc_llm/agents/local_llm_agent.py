"""
Local LLM agent using Qwen2.5-7B-Instruct (or any HuggingFace causal LM).

Probability extraction
----------------------
We use the *log-probability* approach rather than asking the model to
output JSON probabilities.  For each question we feed the full chat-
formatted prompt and read the next-token logits at the position where
the model would write its answer.  We then take the log-softmax values
for tokens "A", "B", "C", "D" and re-normalise them to a proper
probability simplex.

This is the standard approach used in MMLU benchmarking papers and is
more reliable than asking a 7B model to produce valid JSON distributions.

Agent diversity
---------------
Five agents share the same base model but receive distinct system-prompt
personas.  The persona text shifts attention patterns so each agent
produces a slightly different probability distribution — modelling a
realistic multi-agent committee with different specialisations.

Caching
-------
Identical to ClaudeAgent: every (agent_id, model, persona, question,
choices) triple is SHA-256 hashed and the 4-element float32 array is
stored in  data/cache/{hash}.json.  Re-running never recomputes a call.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import ClassVar, Dict, List, Optional, Tuple

import numpy as np

from .base import Agent


# ── Agent personas ────────────────────────────────────────────────────────────

# Domain-specialised personas, used by the "domain" experiment.
# Each persona is an *expertise hint* — the model still answers every
# question, but its calibration profile shifts toward its specialty.
# Indexed by DOMAIN_PERSONAS[domain_idx].
DOMAIN_PERSONAS: List[str] = [
    (
        "You are a mathematician with deep expertise in algebra, calculus, "
        "discrete mathematics, statistics, and formal logic. You answer "
        "questions carefully, applying mathematical reasoning where relevant."
    ),
    (
        "You are a natural scientist with deep expertise in physics, "
        "chemistry, biology, and medicine. You answer questions carefully, "
        "applying scientific reasoning where relevant."
    ),
    (
        "You are a humanities scholar with deep expertise in history, "
        "philosophy, religion, literature, and ethics. You answer questions "
        "carefully, applying humanistic reasoning where relevant."
    ),
    (
        "You are a social scientist with deep expertise in economics, "
        "psychology, sociology, political science, and law. You answer "
        "questions carefully, applying social-scientific reasoning where relevant."
    ),
    (
        "You are a generalist with strong applied-knowledge expertise in "
        "computer science, security, engineering, business, and miscellaneous "
        "professional fields. You answer questions carefully, applying "
        "practical reasoning where relevant."
    ),
]


AGENT_PERSONAS: List[str] = [
    (
        "You are Agent-Alpha, a meticulous academic reviewer. "
        "You carefully weigh each answer option before choosing."
    ),
    (
        "You are Agent-Beta, a domain expert with deep subject knowledge. "
        "You assign high confidence to answers you are certain about."
    ),
    (
        "You are Agent-Gamma, a systematic analyst. "
        "You eliminate clearly wrong options first, then pick the best remaining answer."
    ),
    (
        "You are Agent-Delta, a rigorous scholar trained across many disciplines. "
        "You reason carefully before committing to an answer."
    ),
    (
        "You are Agent-Epsilon, a critical thinker who challenges assumptions. "
        "You evaluate all choices carefully before selecting one."
    ),
]

_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "A) {A}\nB) {B}\nC) {C}\nD) {D}\n\n"
    "Answer with just the letter (A, B, C, or D):"
)

# HuggingFace model identifier (loaded from /data/models cache)
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_CACHE_DIR = "/data/models"


class LocalLLMAgent(Agent):
    """
    LLM agent backed by a local HuggingFace model.

    The model and tokenizer are loaded ONCE (class-level singleton keyed by
    model_id) so that N agents in the same process share the same weights.

    Parameters
    ----------
    agent_id    : 0-4, determines which persona is used
    model_id    : HuggingFace model name (must be present in hf_cache_dir)
    hf_cache_dir: where HuggingFace caches the model weights
    result_cache_dir: where per-query JSON caches are stored
    device      : "cuda" or "cpu"
    """

    # Shared model / tokenizer across all instances
    _loaded: ClassVar[Dict[str, Tuple]] = {}

    def __init__(
        self,
        agent_id: int,
        model_id: str = DEFAULT_MODEL_ID,
        hf_cache_dir: str = DEFAULT_CACHE_DIR,
        result_cache_dir: str = "data/cache",
        device: str = "cuda",
        persona: Optional[str] = None,
    ):
        """
        persona: if provided, override the default persona for this agent_id.
                  Used by domain-specialised + heterogeneous-committee
                  experiments.
        """
        self.agent_id         = agent_id
        self.model_id         = model_id
        self.hf_cache_dir     = hf_cache_dir
        self.result_cache_dir = result_cache_dir
        self.device           = device
        self.persona          = (persona if persona is not None
                                 else AGENT_PERSONAS[agent_id % len(AGENT_PERSONAS)])
        os.makedirs(result_cache_dir, exist_ok=True)
        # Model loaded lazily on first cache miss (Phase 2 never needs it)

    # ── Public interface ──────────────────────────────────────────────────────

    def get_probs(self, question: str, choices: List[str]) -> np.ndarray:
        if len(choices) != 4:
            raise ValueError("Only 4-choice questions are supported.")

        key  = self._cache_key(question, choices)
        path = os.path.join(self.result_cache_dir, f"{key}.json")

        if os.path.exists(path):
            with open(path) as f:
                return np.array(json.load(f), dtype=np.float32)

        # Cache miss: load model if not yet loaded, then infer
        self._ensure_model_loaded()
        probs = self._infer(question, choices)
        with open(path, "w") as f:
            json.dump(probs.tolist(), f)
        return probs

    # ── Internals ─────────────────────────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        if self.model_id not in LocalLLMAgent._loaded:
            LocalLLMAgent._loaded[self.model_id] = _load_model(
                self.model_id, self.hf_cache_dir, self.device
            )
        self._tok, self._model, self._letter_ids = LocalLLMAgent._loaded[self.model_id]

    def _cache_key(self, question: str, choices: List[str]) -> str:
        blob = json.dumps({
            "agent_id": self.agent_id,
            "model_id": self.model_id,
            "persona":  self.persona,
            "question": question,
            "choices":  choices,
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def _build_prompt(self, question: str, choices: List[str]) -> str:
        """Build chat-formatted prompt. Requires tokenizer (call _ensure_model_loaded first)."""
        user_msg = _USER_TEMPLATE.format(
            question=question,
            A=choices[0], B=choices[1], C=choices[2], D=choices[3],
        )
        messages = [
            {"role": "system", "content": self.persona},
            {"role": "user",   "content": user_msg},
        ]
        return self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _infer(self, question: str, choices: List[str]) -> np.ndarray:
        import torch
        prompt = self._build_prompt(question, choices)
        inputs = self._tok(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self._model.device)

        with torch.no_grad():
            logits = self._model(**inputs).logits[0, -1, :]

        return _logits_to_probs(logits, self._letter_ids)


# ── Batched fetch (used by fetch_responses.py for efficiency) ─────────────────

def batch_compute_probs(
    questions: List,           # List[Question]
    n_agents: int,
    model_id: str = DEFAULT_MODEL_ID,
    hf_cache_dir: str = DEFAULT_CACHE_DIR,
    result_cache_dir: str = "data/cache",
    device: str = "cuda",
    batch_size: int = 32,
) -> None:
    """
    Compute and cache probabilities for all (agent, question) pairs.

    Batches prompts across agents and questions for maximum GPU throughput.
    Already-cached pairs are skipped.
    """
    import torch, time

    os.makedirs(result_cache_dir, exist_ok=True)

    # Load model once (eagerly, since batch_compute_probs always needs it)
    if model_id not in LocalLLMAgent._loaded:
        LocalLLMAgent._loaded[model_id] = _load_model(model_id, hf_cache_dir, device)
    tok, model, letter_ids = LocalLLMAgent._loaded[model_id]

    # Build agents (model already loaded; _ensure_model_loaded will be no-op)
    agents_list = [
        LocalLLMAgent(i, model_id=model_id, hf_cache_dir=hf_cache_dir,
                      result_cache_dir=result_cache_dir, device=device)
        for i in range(n_agents)
    ]
    # Propagate loaded model to each agent so _build_prompt works
    for ag in agents_list:
        ag._tok, ag._model, ag._letter_ids = LocalLLMAgent._loaded[model_id]

    pending: List[tuple] = []  # (agent, question)
    for agent in agents_list:
        for q in questions:
            key  = agent._cache_key(q.question, q.choices)
            path = os.path.join(result_cache_dir, f"{key}.json")
            if not os.path.exists(path):
                pending.append((agent, q))

    total   = len(pending)
    cached  = len(agents_list) * len(questions) - total
    print(f"  {cached} already cached, {total} to compute")

    if total == 0:
        return

    t0   = time.time()
    done = 0

    for start in range(0, total, batch_size):
        batch     = pending[start : start + batch_size]
        prompts   = [ag._build_prompt(q.question, q.choices) for ag, q in batch]

        # Tokenise with left-padding so last position is always answer position
        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            last_logits = model(**enc).logits[:, -1, :]  # (B, vocab)

        for idx, (ag, q) in enumerate(batch):
            probs = _logits_to_probs(last_logits[idx], letter_ids)
            key   = ag._cache_key(q.question, q.choices)
            path  = os.path.join(result_cache_dir, f"{key}.json")
            with open(path, "w") as f:
                json.dump(probs.tolist(), f)

        done += len(batch)
        if done % 500 == 0 or done == total:
            elapsed = time.time() - t0
            eta     = elapsed / done * (total - done) if done else 0
            pct     = 100 * done / total
            print(f"    {done}/{total}  ({pct:.0f}%)  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s",
                  flush=True)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _load_model(model_id: str, hf_cache_dir: str, device: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {model_id} …", flush=True)
    tok = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=hf_cache_dir,
        local_files_only=True,
    )
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=hf_cache_dir,
        local_files_only=True,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"  Loaded on {next(model.parameters()).device}", flush=True)

    # Token IDs for answer letters
    letter_ids = [tok.encode(l, add_special_tokens=False)[-1] for l in "ABCD"]
    return tok, model, letter_ids


def _logits_to_probs(logits, letter_ids) -> np.ndarray:
    """Extract softmax-normalised probabilities for A/B/C/D."""
    import torch
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    scores    = np.array([log_probs[t].item() for t in letter_ids], dtype=np.float64)
    scores   -= scores.max()          # numerical stability
    probs     = np.exp(scores)
    probs    /= probs.sum()
    return probs.astype(np.float32)
