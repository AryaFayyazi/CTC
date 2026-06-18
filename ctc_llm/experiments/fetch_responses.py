"""
Phase 1 — Pre-fetch and cache all LLM responses.

For EACH model in --models, we cache:
  • 5 clean agents × all MCQ questions   (10 agents on the primary model)
  • 2 injection-attack personas × N agents × all MCQ questions
  • [free-form] 5 sampling agents × GSM8K questions
  • [free-form] 5 seq-logprob agents × HellaSwag questions

Then Phase 2 runs purely from cache with no GPU.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m ctc_llm.experiments.fetch_responses \\
        --models Qwen/Qwen2.5-7B-Instruct \\
        --tasks mmlu truthfulqa arc gsm8k hellaswag --batch-size 64
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ctc_llm.tasks.mmlu import (
    load_mmlu, load_truthfulqa, load_arc, load_gpqa, MMLU_SUBSET
)
from ctc_llm.tasks.mmlu import Question
from ctc_llm.agents.local_llm_agent  import batch_compute_probs
from ctc_llm.agents.prompt_inject_agent import batch_compute_injection_probs


MODELS_DEFAULT = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "allenai/Olmo-3-7B-Instruct",         # 5th model for heterogeneous committee
]
INJECTION_ATTACKS    = ["sycophant_wrong", "deceptive"]
N_AGENTS_CLEAN_MAIN  = 5    # all models need 5 personas for the main experiment
N_AGENTS_CLEAN_BIG   = 10   # primary model only — for the N=10 scaling ablation
N_AGENTS_INJECT      = 5    # primary model only — injection attacks for ablation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks",         nargs="+", default=["mmlu", "truthfulqa", "arc"])
    p.add_argument("--subjects",      nargs="*", default=None)
    p.add_argument("--models",        nargs="+", default=None)
    p.add_argument("--hf-cache-dir",  default="/data/models")
    p.add_argument("--cache-dir",     default="data/cache")
    p.add_argument("--freeform-cache-dir", default="data/cache_freeform")
    p.add_argument("--batch-size",    type=int, default=64)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--skip-injection", action="store_true")
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.freeform_cache_dir, exist_ok=True)
    models = args.models if args.models else MODELS_DEFAULT

    # ── MCQ tasks ─────────────────────────────────────────────────────────────
    mcq_tasks = [t for t in args.tasks if t in ("mmlu", "truthfulqa", "arc", "gpqa")]
    all_questions: List[Question] = []
    if "mmlu" in mcq_tasks:
        subjects = args.subjects or MMLU_SUBSET
        print(f"Loading MMLU ({len(subjects)} subjects)…")
        qs = load_mmlu(split="test", subjects=subjects)
        print(f"  {len(qs)} questions")
        all_questions.extend(qs)
    if "truthfulqa" in mcq_tasks:
        print("Loading TruthfulQA…")
        qs = load_truthfulqa(max_questions=600)
        print(f"  {len(qs)} questions")
        all_questions.extend(qs)
    if "arc" in mcq_tasks:
        print("Loading ARC-Challenge…")
        qs = load_arc(max_questions=600)
        print(f"  {len(qs)} questions")
        all_questions.extend(qs)
    if "gpqa" in mcq_tasks:
        print("Loading GPQA…")
        qs = load_gpqa(max_questions=600)
        print(f"  {len(qs)} questions")
        all_questions.extend(qs)

    seen = set()
    unique: List[Question] = []
    for q in all_questions:
        key = (q.question, tuple(q.choices))
        if key not in seen:
            seen.add(key)
            unique.append(q)
    all_questions = unique
    primary_model = models[0]

    if all_questions:
        print(f"\nUnique MCQ questions : {len(all_questions)}")
        print(f"Models               : {models}")
        print(f"Primary (ablations)  : {primary_model}")

        for model_id in models:
            is_primary = (model_id == primary_model)
            n_clean    = N_AGENTS_CLEAN_BIG if is_primary else N_AGENTS_CLEAN_MAIN

            if any(s in model_id for s in ["70B", "72B"]):
                bs = 2
            elif any(s in model_id for s in ["32B", "30B", "20B", "14B", "13B"]):
                bs = 4
            else:
                bs = args.batch_size

            print(f"━" * 56)
            print(f"MODEL: {model_id}  ({'PRIMARY' if is_primary else 'aux'})  bs={bs}")
            t0 = time.time()

            print(f"\n[clean] {n_clean} agents × {len(all_questions)} questions")
            batch_compute_probs(
                questions        = all_questions,
                n_agents         = n_clean,
                model_id         = model_id,
                hf_cache_dir     = args.hf_cache_dir,
                result_cache_dir = args.cache_dir,
                device           = args.device,
                batch_size       = bs,
            )

            if is_primary and not args.skip_injection:
                print(f"\n[injection] {len(INJECTION_ATTACKS)} attacks × "
                      f"{N_AGENTS_INJECT} agents × {len(all_questions)} questions")
                batch_compute_injection_probs(
                    questions           = all_questions,
                    attacks             = INJECTION_ATTACKS,
                    n_agents_per_attack = N_AGENTS_INJECT,
                    model_id            = model_id,
                    hf_cache_dir        = args.hf_cache_dir,
                    result_cache_dir    = args.cache_dir,
                    device              = args.device,
                    batch_size          = bs,
                )

            if model_id != models[-1]:
                import torch, gc
                from ctc_llm.agents.local_llm_agent import LocalLLMAgent
                if model_id in LocalLLMAgent._loaded:
                    del LocalLLMAgent._loaded[model_id]
                gc.collect()
                torch.cuda.empty_cache()

            print(f"\n  Model {model_id} done in {time.time()-t0:.0f}s")

    # ── Free-form tasks (primary model only) ──────────────────────────────────
    freeform_tasks = [t for t in args.tasks if t in ("gsm8k", "hellaswag")]
    if freeform_tasks:
        print(f"\n{'━'*56}")
        print(f"FREE-FORM INFERENCE  (primary model: {primary_model})")
        print(f"{'━'*56}")
        from ctc_llm.agents.local_llm_agent import LocalLLMAgent

        if primary_model not in LocalLLMAgent._loaded:
            from ctc_llm.agents.local_llm_agent import _load_model
            LocalLLMAgent._loaded[primary_model] = _load_model(
                primary_model, args.hf_cache_dir, args.device
            )

        if "gsm8k" in freeform_tasks:
            from ctc_llm.tasks.freeform import load_gsm8k_free
            from ctc_llm.agents.freeform_agent import batch_compute_seq_logprobs
            print("Loading GSM8K (sequence log-prob scoring)…")
            gsm_qs = load_gsm8k_free(max_questions=400)
            print(f"  {len(gsm_qs)} questions, {len(gsm_qs[0].candidates)} candidates each")
            batch_compute_seq_logprobs(
                questions        = gsm_qs,
                n_agents         = N_AGENTS_CLEAN_MAIN,
                model_id         = primary_model,
                hf_cache_dir     = args.hf_cache_dir,
                result_cache_dir = args.freeform_cache_dir,
                device           = args.device,
                batch_size       = 16,
            )

        if "hellaswag" in freeform_tasks:
            from ctc_llm.tasks.freeform import load_hellaswag_seq
            from ctc_llm.agents.freeform_agent import batch_compute_seq_logprobs
            print("Loading HellaSwag (sequence log-prob scoring)…")
            hs_qs = load_hellaswag_seq(max_questions=400)
            print(f"  {len(hs_qs)} questions, {len(hs_qs[0].candidates)} candidates each")
            batch_compute_seq_logprobs(
                questions        = hs_qs,
                n_agents         = N_AGENTS_CLEAN_MAIN,
                model_id         = primary_model,
                hf_cache_dir     = args.hf_cache_dir,
                result_cache_dir = args.freeform_cache_dir,
                device           = args.device,
                batch_size       = 16,
            )

    n_mcq = len([f for f in os.listdir(args.cache_dir) if f.endswith(".json")])
    n_ff  = len([f for f in os.listdir(args.freeform_cache_dir) if f.endswith(".json")])
    print(f"\nMCQ cache files     : {n_mcq}")
    print(f"Free-form cache files: {n_ff}")
    print("Ready for Phase 2.")


if __name__ == "__main__":
    main()
