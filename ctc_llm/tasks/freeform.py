"""
Free-form (non-MCQ) task loaders for CTC generalization experiments.

Two tasks:
  gsm8k       Grade-school math — open-ended integer answer generation.
              No answer choices are shown; the model must generate the answer.
              Correct answer is the canonical integer string ("18", "42", …).

  hellaswag   Commonsense sentence completion — scored by FULL SEQUENCE
              log-probability P(ending | ctx), NOT by first-token MCQ logit.
              The model sees no A/B/C/D labels; it scores each candidate
              continuation as free text.

Both use a uniform FreeFormQuestion interface.
"""

from __future__ import annotations

import re
import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FreeFormQuestion:
    question: str
    correct_answer: str          # canonical correct answer string
    correct_aliases: List[str]   # all accepted strings (incl. canonical)
    subject: str
    candidates: Optional[List[str]] = field(default=None)
    # candidates: fixed-vocabulary tasks only (e.g. HellaSwag 4 endings).
    # None → truly open-ended (GSM8K); agents generate answers by sampling.

    def is_correct(self, pred: str) -> bool:
        pred = pred.strip().lower()
        return any(pred == a.strip().lower() for a in self.correct_aliases)


# ── GSM8K (open-ended math generation) ───────────────────────────────────────

def _extract_gsm8k_int(answer_str: str) -> Optional[str]:
    """Extract the final integer from 'chain ... #### N' format."""
    m = re.search(r"####\s*([\-\d,]+)", answer_str)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        return str(int(raw))   # normalise: remove leading zeros, strip sign if +
    except ValueError:
        return None


def load_gsm8k_free(
    max_questions: int = 400,
    seed: int = 0,
    split: str = "test",
    n_distractors: int = 3,
) -> List[FreeFormQuestion]:
    """
    Load GSM8K in SEQUENCE-SCORING mode.

    The model receives the problem and must score each candidate answer
    via log P(candidate | problem context) — exactly the same mechanism
    as HellaSwag, just with integer strings as candidates instead of
    sentence continuations.

    For each question we build a candidate set:
      correct_answer + n_distractors sampled from the real answer pool.

    The framing is:
      question  = "Problem: {text}\\n\\nAnswer:"
      candidates = [" 18", " 42", ...]  (space-prefixed integers)

    No answer choices, labels, or MCQ formatting are shown to the model.
    """
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("gsm8k", "main", cache_dir="/data/models", split=split)
    except Exception as e:
        raise RuntimeError(f"Could not load GSM8K: {e}")

    rng = random.Random(seed)

    # First pass: collect ALL valid (question_text, correct_int_str) pairs
    raw: List[tuple] = []
    for row in ds:
        ans = _extract_gsm8k_int(row["answer"])
        if ans is not None:
            raw.append((row["question"], ans))

    if not raw:
        raise RuntimeError("GSM8K: no questions with parseable answers")

    # Build pool from the FULL dataset (not just the subsample)
    # so distractors are always available
    answer_pool: List[str] = list({a for _, a in raw})

    rng.shuffle(raw)

    questions: List[FreeFormQuestion] = []
    for q_text, correct_str in raw:
        if len(questions) >= max_questions:
            break
        wrongs = [a for a in answer_pool if a != correct_str]
        if len(wrongs) < n_distractors:
            continue
        distractors = rng.sample(wrongs, n_distractors)
        # Candidates as space-prefixed strings (natural continuation after "Answer:")
        candidates = [f" {correct_str}"] + [f" {d}" for d in distractors]
        rng.shuffle(candidates)
        correct_candidate = f" {correct_str}"
        questions.append(FreeFormQuestion(
            question=f"Problem: {q_text}\n\nAnswer:",
            correct_answer=correct_candidate,
            correct_aliases=[correct_candidate, correct_str],
            subject="gsm8k",
            candidates=candidates,
        ))

    return questions


# ── HellaSwag (full-sequence log-prob scoring) ────────────────────────────────

def load_hellaswag_seq(
    max_questions: int = 400,
    seed: int = 0,
    split: str = "validation",
) -> List[FreeFormQuestion]:
    """
    Load HellaSwag in SEQUENCE-SCORING mode.

    Unlike the MCQ approach (which extracts first-token logits for A/B/C/D),
    here each agent scores each candidate ending by computing the full
    sequence log-probability log P(ending | context) token by token.
    This is closer to how a language model naturally processes text
    completion — no MCQ labels, no special tokens required.

    FreeFormQuestion.candidates = the 4 full ending strings.
    FreeFormQuestion.correct_answer = the correct ending string.
    """
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("hellaswag", cache_dir="/data/models", split=split)
    except Exception as e:
        raise RuntimeError(f"Could not load HellaSwag: {e}")

    rng = random.Random(seed)
    questions: List[FreeFormQuestion] = []

    for row in ds:
        endings = row["endings"]
        if len(endings) != 4:
            continue
        try:
            label = int(row["label"])
        except (ValueError, TypeError):
            continue
        if label not in (0, 1, 2, 3):
            continue
        ctx = row["ctx"].strip()
        if not ctx:
            continue
        correct_ending = endings[label]
        questions.append(FreeFormQuestion(
            question=ctx,
            correct_answer=correct_ending,
            correct_aliases=[correct_ending],
            subject="hellaswag",
            candidates=list(endings),   # the 4 candidate continuations
        ))

    rng.shuffle(questions)
    return questions[:max_questions]
