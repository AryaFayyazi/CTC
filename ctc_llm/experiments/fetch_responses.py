"""
Phase 1 — Pre-fetch and cache all LLM responses.

For EACH model in --models, we cache:
  • 5 clean agents × all questions   (10 agents on the primary model)
  • 2 injection-attack personas × N agents × all questions

Then Phase 2 runs purely from cache with no GPU.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m ctc_llm.experiments.fetch_responses \\
        --models Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct \\
        --tasks mmlu truthfulqa arc --batch-size 64
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
    p.add_argument("--batch-size",    type=int, default=64)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--skip-injection", action="store_true")
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    models = args.models if args.models else MODELS_DEFAULT

    # Load all question data once
    all_questions: List[Question] = []
    if "mmlu" in args.tasks:
        subjects = args.subjects or MMLU_SUBSET
        print(f"Loading MMLU ({len(subjects)} subjects)…")
        qs = load_mmlu(split="test", subjects=subjects)
        print(f"  {len(qs)} questions")
        all_questions.extend(qs)
    if "truthfulqa" in args.tasks:
        print("Loading TruthfulQA…")
        qs = load_truthfulqa(max_questions=600)
        print(f"  {len(qs)} questions")
        all_questions.extend(qs)
    if "arc" in args.tasks:
        print("Loading ARC-Challenge…")
        qs = load_arc(max_questions=600)
        print(f"  {len(qs)} questions")
        all_questions.extend(qs)
    if "gpqa" in args.tasks:
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
    print(f"\nUnique questions   : {len(all_questions)}")
    print(f"Models             : {models}")
    print(f"Primary (ablations): {primary_model}")
    print(f"Clean personas     : {N_AGENTS_CLEAN_MAIN} per model "
          f"(+ {N_AGENTS_CLEAN_BIG - N_AGENTS_CLEAN_MAIN} extra on primary)")
    print(f"Inject personas    : {N_AGENTS_INJECT} × {len(INJECTION_ATTACKS)} (primary only)\n")

    for model_id in models:
        is_primary = (model_id == primary_model)
        n_clean    = N_AGENTS_CLEAN_BIG if is_primary else N_AGENTS_CLEAN_MAIN

        # Auto-pick batch size: large models (>10B) need smaller batches
        # to fit lm_head output on H100 (94 GB VRAM)
        if any(s in model_id for s in ["70B", "72B"]):
            bs = 2
        elif any(s in model_id for s in ["32B", "30B", "20B", "14B", "13B"]):
            bs = 4
        else:
            bs = args.batch_size

        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"MODEL: {model_id}  ({'PRIMARY' if is_primary else 'aux'})  bs={bs}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
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

        # Free GPU memory between models
        if model_id != models[-1]:
            import torch, gc
            from ctc_llm.agents.local_llm_agent import LocalLLMAgent
            if model_id in LocalLLMAgent._loaded:
                del LocalLLMAgent._loaded[model_id]
            gc.collect()
            torch.cuda.empty_cache()

        print(f"\n  Model {model_id} done in {time.time()-t0:.0f}s")

    n_cached = len([f for f in os.listdir(args.cache_dir) if f.endswith(".json")])
    print(f"\nTotal cache files: {n_cached}")
    print("Ready for Phase 2.")


if __name__ == "__main__":
    main()
