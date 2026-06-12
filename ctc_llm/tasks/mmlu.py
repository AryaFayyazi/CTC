"""
MMLU, TruthfulQA, and ARC-Challenge loaders.

All tasks share a uniform Question interface:
    question : str           — the question text
    choices  : List[str]     — exactly 4 answer options
    correct  : int           — 0-indexed correct choice
    subject  : str           — task / subject label
"""

from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Question:
    question: str
    choices: List[str]   # length == 4
    correct: int         # 0-3
    subject: str

    @property
    def correct_letter(self) -> str:
        return "ABCD"[self.correct]


# ── MMLU ─────────────────────────────────────────────────────────────────────

MMLU_CATEGORIES: Dict[str, List[str]] = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy",
        "college_biology", "college_chemistry", "college_computer_science",
        "college_mathematics", "college_physics", "computer_security",
        "electrical_engineering", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios",
        "philosophy", "prehistory", "professional_law", "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_psychology", "human_sexuality", "political_science",
        "professional_psychology", "public_relations", "security_studies",
        "sociology", "us_foreign_policy",
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition",
        "professional_accounting", "professional_medicine", "virology",
    ],
}

ALL_MMLU_SUBJECTS: List[str] = [s for subs in MMLU_CATEGORIES.values() for s in subs]

# Representative 16-subject subset (4 per category) for quick runs
MMLU_SUBSET: List[str] = [
    # STEM
    "college_mathematics", "high_school_physics",
    "computer_security", "machine_learning",
    # Humanities
    "formal_logic", "moral_scenarios",
    "philosophy", "high_school_us_history",
    # Social Sciences
    "high_school_psychology", "political_science",
    "econometrics", "sociology",
    # Other
    "medical_genetics", "nutrition",
    "business_ethics", "clinical_knowledge",
]


def load_mmlu(
    split: str = "test",
    subjects: Optional[List[str]] = None,
    max_per_subject: Optional[int] = None,
    seed: int = 0,
    cache_dir: str = "data",
) -> List[Question]:
    """
    Load MMLU questions via HuggingFace datasets.
    Falls back to downloading a CSV snapshot if datasets is unavailable.
    """
    if subjects is None:
        subjects = MMLU_SUBSET

    try:
        from datasets import load_dataset  # type: ignore
        questions: List[Question] = []
        for subj in subjects:
            try:
                ds = load_dataset("cais/mmlu", subj, split=split)
                rows = list(ds)
            except Exception:
                continue
            if max_per_subject:
                rng = random.Random(seed)
                rows = rng.sample(rows, min(max_per_subject, len(rows)))
            for row in rows:
                choices = row["choices"]
                if len(choices) != 4:
                    continue
                questions.append(Question(
                    question=row["question"],
                    choices=choices,
                    correct=int(row["answer"]),
                    subject=subj,
                ))
        return questions
    except ImportError:
        raise RuntimeError(
            "Install HuggingFace datasets: pip install datasets"
        )


# ── TruthfulQA (MC1 format) ───────────────────────────────────────────────────

def load_truthfulqa(
    max_questions: int = 500,
    seed: int = 0,
    split: str = "validation",
) -> List[Question]:
    """
    Load TruthfulQA in MC1 format (single correct answer from 4 options).
    We sample 4 choices per question: 1 correct + 3 random incorrect.
    """
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("truthful_qa", "multiple_choice", split=split)
    except Exception as e:
        raise RuntimeError(f"Could not load TruthfulQA: {e}")

    rng = random.Random(seed)
    questions: List[Question] = []

    for row in ds:
        mc_targets = row["mc1_targets"]
        choices_all = mc_targets["choices"]
        labels_all  = mc_targets["labels"]   # 1=correct, 0=wrong

        correct_opts = [c for c, l in zip(choices_all, labels_all) if l == 1]
        wrong_opts   = [c for c, l in zip(choices_all, labels_all) if l == 0]

        if not correct_opts or len(wrong_opts) < 3:
            continue

        correct_choice = rng.choice(correct_opts)
        wrong_choices  = rng.sample(wrong_opts, 3)
        four_choices   = [correct_choice] + wrong_choices
        rng.shuffle(four_choices)
        correct_idx    = four_choices.index(correct_choice)

        questions.append(Question(
            question=row["question"],
            choices=four_choices,
            correct=correct_idx,
            subject="truthfulqa",
        ))

    rng.shuffle(questions)
    return questions[:max_questions]


# ── ARC-Challenge ─────────────────────────────────────────────────────────────

def load_arc(
    max_questions: int = 500,
    seed: int = 0,
    split: str = "test",
) -> List[Question]:
    """
    Load ARC-Challenge (4-choice science questions).
    Filters to questions that have exactly 4 options.
    """
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("ai2_arc", "ARC-Challenge", split=split)
    except Exception as e:
        raise RuntimeError(f"Could not load ARC: {e}")

    rng = random.Random(seed)
    questions: List[Question] = []

    label_map = {"A": 0, "B": 1, "C": 2, "D": 3,
                 "1": 0, "2": 1, "3": 2, "4": 3}

    for row in ds:
        choices = row["choices"]["text"]
        labels  = row["choices"]["label"]
        if len(choices) != 4:
            continue
        ans_key = row["answerKey"]
        if ans_key not in label_map:
            continue
        correct_idx = label_map[ans_key]
        questions.append(Question(
            question=row["question"],
            choices=choices,
            correct=correct_idx,
            subject="arc_challenge",
        ))

    rng.shuffle(questions)
    return questions[:max_questions]


# ── GPQA (Graduate-level Google-Proof Q&A — 2024+ standard hard benchmark) ───

def load_gpqa(
    max_questions: int = 500,
    seed: int = 0,
    split: str = "test",
) -> List[Question]:
    """
    Load GPQA (Graduate-level Q&A, casimiir/gpqa version on the HuggingFace
    Hub).  448 multiple-choice questions across Physics, Chemistry, and
    Biology at the graduate level.  Used as the hard-reasoning benchmark
    in many 2024+ multi-agent and conformal-prediction LLM papers.

    Format: {question, choices: [A,B,C,D], answer: letter A/B/C/D}
    """
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("casimiir/gpqa", cache_dir="/data/models",
                          split=split)
    except Exception as e:
        raise RuntimeError(f"Could not load GPQA: {e}")

    rng = random.Random(seed)
    questions: List[Question] = []
    letter_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    for row in ds:
        choices = row["choices"]
        if len(choices) != 4:
            continue
        ans_key = row["answer"]
        if ans_key not in letter_map:
            continue
        correct_idx = letter_map[ans_key]
        questions.append(Question(
            question=row["question"],
            choices=list(choices),
            correct=correct_idx,
            subject=row.get("subdomain", "gpqa"),
        ))

    rng.shuffle(questions)
    return questions[:max_questions]


# ── Utilities ─────────────────────────────────────────────────────────────────

def split_questions(
    questions: List[Question],
    cal_fraction: float = 0.33,
    seed: int = 0,
) -> tuple[List[Question], List[Question]]:
    """Split into calibration and test sets, stratified by subject."""
    from collections import defaultdict
    by_subject: Dict[str, List[Question]] = defaultdict(list)
    for q in questions:
        by_subject[q.subject].append(q)

    cal, test = [], []
    rng = random.Random(seed)
    for subj_qs in by_subject.values():
        rng.shuffle(subj_qs)
        n_cal = max(1, int(len(subj_qs) * cal_fraction))
        cal.extend(subj_qs[:n_cal])
        test.extend(subj_qs[n_cal:])
    return cal, test
