"""
Agents for free-form (non-MCQ) tasks.

SequenceLogProbAgent
    Scores each candidate answer by FULL SEQUENCE log-probability:
      log P(candidate | context) = Σ_t log P(token_t | context + candidate[:t])

    Used for both tasks:
      HellaSwag : context = sentence stem, candidates = 4 full endings
      GSM8K     : context = "Problem: ...\\n\\nAnswer:", candidates = integer strings

    This is categorically different from the MCQ approach, which extracts
    logits only for the first token (A/B/C/D).  Here the model evaluates
    the entire candidate string as a natural language continuation.

    Returns np.ndarray(n_candidates) — compatible with the existing MCQ
    coordination pipeline (majority, entropy-trust, CTC-Hybrid, etc.).

batch_compute_seq_logprobs
    Phase-1 batched caching function (same API as batch_compute_probs).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import List, Tuple

import numpy as np

from ctc_llm.agents.local_llm_agent import (
    LocalLLMAgent, _load_model, DEFAULT_MODEL_ID, DEFAULT_CACHE_DIR,
)
from ctc_llm.tasks.freeform import FreeFormQuestion


class SequenceLogProbAgent:
    """
    Scores each candidate continuation by full sequence log-probability.

    log P(candidate | context) = Σ_t log P(token_t | context + candidate[:t])

    The context is q.question; candidates are q.candidates.
    Probabilities are obtained via softmax of length-normalised log-probs.
    Results are cached to disk identically to LocalLLMAgent.
    """

    def __init__(
        self,
        agent_id: int,
        model_id: str = DEFAULT_MODEL_ID,
        hf_cache_dir: str = DEFAULT_CACHE_DIR,
        result_cache_dir: str = "data/cache_freeform",
        device: str = "cuda",
    ):
        self.agent_id         = agent_id
        self.model_id         = model_id
        self.hf_cache_dir     = hf_cache_dir
        self.result_cache_dir = result_cache_dir
        self.device           = device
        os.makedirs(result_cache_dir, exist_ok=True)

    def get_probs(self, q: FreeFormQuestion) -> np.ndarray:
        """Return softmax of length-normalised sequence log-probs over q.candidates."""
        if not q.candidates:
            raise ValueError("SequenceLogProbAgent requires q.candidates")

        key  = self._cache_key(q)
        path = os.path.join(self.result_cache_dir, f"{key}.json")

        if os.path.exists(path):
            with open(path) as f:
                return np.array(json.load(f), dtype=np.float32)

        self._ensure_model()
        probs = self._score(q)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(probs.tolist(), f)
        os.replace(tmp, path)
        return probs

    # ── Internals ──────────────────────────────────────────────────────────────

    def _ensure_model(self) -> None:
        if self.model_id not in LocalLLMAgent._loaded:
            LocalLLMAgent._loaded[self.model_id] = _load_model(
                self.model_id, self.hf_cache_dir, self.device
            )
        self._tok, self._model, _ = LocalLLMAgent._loaded[self.model_id]

    def _cache_key(self, q: FreeFormQuestion) -> str:
        blob = json.dumps({
            "type":       "seq_logprob",
            "agent_id":   self.agent_id,
            "model_id":   self.model_id,
            "question":   q.question,
            "candidates": q.candidates,
        }, sort_keys=True)
        return "seq_" + hashlib.sha256(blob.encode()).hexdigest()[:22]

    def _score(self, q: FreeFormQuestion) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        ctx_text = q.question.strip()
        ctx_ids  = self._tok.encode(ctx_text, add_special_tokens=True)
        log_probs = []

        for candidate in q.candidates:
            full_text = ctx_text + candidate
            enc = self._tok(
                full_text, return_tensors="pt",
                truncation=True, max_length=512,
            ).to(self._model.device)
            ids = enc["input_ids"][0]

            with torch.no_grad():
                logits = self._model(**enc).logits[0]   # (seq_len, vocab)

            lp = F.log_softmax(logits, dim=-1)
            cand_start = len(ctx_ids)
            score = 0.0
            n_cand = 0
            for t in range(cand_start, len(ids)):
                if t < lp.shape[0]:
                    score += lp[t - 1, ids[t]].item()
                    n_cand += 1
            # Length-normalise to avoid bias toward shorter candidates
            log_probs.append(score / max(1, n_cand))

        arr = np.array(log_probs, dtype=np.float64)
        arr -= arr.max()                    # numerical stability
        probs = np.exp(arr)
        return (probs / probs.sum()).astype(np.float32)


# ── batch_compute_seq_logprobs (Phase 1) ──────────────────────────────────────

def batch_compute_seq_logprobs(
    questions: List[FreeFormQuestion],
    n_agents: int,
    model_id: str = DEFAULT_MODEL_ID,
    hf_cache_dir: str = DEFAULT_CACHE_DIR,
    result_cache_dir: str = "data/cache_freeform",
    device: str = "cuda",
    batch_size: int = 16,
) -> None:
    """
    Compute and cache sequence log-probs for all (agent_id, question) pairs.
    Skips already-cached results.
    """
    import time

    if model_id not in LocalLLMAgent._loaded:
        LocalLLMAgent._loaded[model_id] = _load_model(model_id, hf_cache_dir, device)

    agents = [
        SequenceLogProbAgent(i, model_id=model_id, hf_cache_dir=hf_cache_dir,
                             result_cache_dir=result_cache_dir, device=device)
        for i in range(n_agents)
    ]
    for ag in agents:
        ag._tok, ag._model, _ = LocalLLMAgent._loaded[model_id]

    pending: List[Tuple] = []
    for ag in agents:
        for q in questions:
            path = os.path.join(result_cache_dir, f"{ag._cache_key(q)}.json")
            if not os.path.exists(path):
                pending.append((ag, q))

    total = len(pending)
    print(f"  [seq-logprob] {total} pending (agent, question) pairs")
    if total == 0:
        return

    t0 = time.time()
    done = 0
    for start in range(0, total, batch_size):
        chunk = pending[start: start + batch_size]
        for ag, q in chunk:
            probs = ag._score(q)
            path  = os.path.join(result_cache_dir, f"{ag._cache_key(q)}.json")
            tmp   = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(probs.tolist(), f)
            os.replace(tmp, path)
        done += len(chunk)
        if done % 100 == 0 or done == total:
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done) if done else 0
            print(f"    {done}/{total} ({100*done/total:.0f}%)  "
                  f"{elapsed:.0f}s  ETA {eta:.0f}s", flush=True)
