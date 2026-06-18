#!/bin/bash
# ============================================================================
# CTC-LLM: Conformal Trust Coordination for LLM Multi-Agent Safety
#
# Comprehensive ICLR-grade experiment suite
# ------------------------------------------
# Models : Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, Phi-3.5-mini-instruct
#          Qwen3-30B-A3B-Instruct-2507, OLMo-3-7B-Instruct
# Tasks  : MMLU (16 subjects), TruthfulQA, ARC-Challenge  [MCQ]
#          GSM8K (open-ended math generation)              [Free-form]
#          HellaSwag (full sequence log-prob scoring)      [Free-form]
# Attacks: overconfident, random, subtle, inject_sycophant, inject_deceptive
# Sizes  : N=5 (primary), N=10 (scaling ablation)
# Alphas : 0.05, 0.10, 0.20
# Methods: vanilla, average, majority, entropy, CTC-Global, CTC-Focal,
#          CTC-Agreement, CTC-Hybrid (main)
# Stats  : 20 seeds, 95% CI, Wilcoxon signed-rank tests
#
# Workflow
# --------
# Phase 1   : batched GPU inference → disk cache  (MCQ + free-form, ~2h on H100)
# Phase 2   : MCQ coordination experiments from cache (CPU only, ~4h)
# Phase 2b  : Free-form coordination experiments from cache (CPU only, ~1h)
# Phase 3   : LaTeX tables (text + .tex files)
# Phase 4   : PDF figures (corruption sweeps, cross-model, coverage, scaling)
# Phase 5   : Selective prediction analysis
#
# Usage:
#   sbatch run.slurm
# ============================================================================
#SBATCH --job-name=ctc_llm_iclr
#SBATCH --output=results/ctc_llm_%j.log
#SBATCH --error=results/ctc_llm_%j.err
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --partition=gpu

set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

REPO=/home/arya/kamal_new
cd "$REPO"

echo "Node   : $(hostname)"
echo "GPUs   : $CUDA_VISIBLE_DEVICES"
echo "CPUs   : $(nproc)"
echo "Date   : $(date)"
echo ""

pip install -q accelerate datasets scipy matplotlib tqdm transformers 2>/dev/null || true
mkdir -p results data/cache data/cache_freeform

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "Qwen/Qwen3-30B-A3B-Instruct-2507"
    "microsoft/Phi-3.5-mini-instruct"
    "allenai/Olmo-3-7B-Instruct"
)
PRIMARY_MODEL="Qwen/Qwen2.5-7B-Instruct"

# ── Phase 1: Pre-fetch (MCQ + free-form, GPU) ────────────────────────────────
CACHE_MCQ=$(find data/cache -name "*.json" 2>/dev/null | wc -l || echo 0)
CACHE_FF=$(find data/cache_freeform -name "*.json" 2>/dev/null | wc -l || echo 0)
echo "========================================"
echo "PHASE 1: FETCH LLM RESPONSES"
echo "MCQ cache files found    : $CACHE_MCQ"
echo "Free-form cache files    : $CACHE_FF"
echo "========================================"
python3 -u -m ctc_llm.experiments.fetch_responses \
    --tasks mmlu truthfulqa arc gsm8k hellaswag \
    --models "${MODELS[@]}" \
    --hf-cache-dir /data/models \
    --cache-dir data/cache \
    --freeform-cache-dir data/cache_freeform \
    --batch-size 64 \
    --device cuda

echo ""
echo "Phase 1 complete.  Starting Phase 2 (CPU only)…"
echo ""

# ── Phase 2: MCQ coordination experiments ───────────────────────────────────
echo "========================================"
echo "PHASE 2: MCQ COORDINATION EXPERIMENTS"
echo "========================================"
python3 -u -m ctc_llm.experiments.run_experiments \
    --tasks mmlu truthfulqa arc \
    --n-seeds 20 \
    --models "${MODELS[@]}" \
    --hf-cache-dir /data/models \
    --cache-dir data/cache \
    --output results/raw_results.json

# ── Phase 2b: Free-form coordination experiments ─────────────────────────────
echo ""
echo "========================================"
echo "PHASE 2b: FREE-FORM COORDINATION (GSM8K + HellaSwag)"
echo "========================================"
python3 -u -m ctc_llm.experiments.run_freeform \
    --tasks gsm8k hellaswag \
    --models "$PRIMARY_MODEL" \
    --hf-cache-dir /data/models \
    --cache-dir data/cache_freeform \
    --output results/raw_results_freeform.json \
    --max-questions 400

# ── Phase 3: Tables (text + LaTeX) ───────────────────────────────────────────
echo ""
echo "========================================"
echo "PHASE 3: TABLES"
echo "========================================"
python3 -u -m ctc_llm.paper.tables \
    --input results/raw_results.json \
    --tasks mmlu truthfulqa arc \
    --models "${MODELS[@]}" \
    --primary-model "$PRIMARY_MODEL" \
    | tee results/tables.txt

# Per-model LaTeX tables
for model in "${MODELS[@]}"; do
    safe=$(echo "$model" | tr '/' '_')
    python3 -u -m ctc_llm.paper.tables \
        --input results/raw_results.json \
        --tasks mmlu truthfulqa arc \
        --models "$model" \
        --latex > "results/table_${safe}.tex"
    echo "Saved results/table_${safe}.tex"
done

# Headline cross-model table
python3 -u -m ctc_llm.paper.tables \
    --input results/raw_results.json \
    --tasks mmlu truthfulqa arc \
    --models "${MODELS[@]}" \
    --latex > results/table_crossmodel.tex
echo "Saved results/table_crossmodel.tex"

# ── Phase 4: Figures ─────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "PHASE 4: FIGURES"
echo "========================================"
python3 -u -m ctc_llm.paper.plot_results \
    --input results/raw_results.json \
    --out-dir results \
    --tasks mmlu truthfulqa arc \
    --models "${MODELS[@]}" \
    --primary-model "$PRIMARY_MODEL"

# ── Phase 5: Selective Prediction Analysis (headline contribution) ───────────
echo ""
echo "========================================"
echo "PHASE 5: SELECTIVE PREDICTION"
echo "========================================"
for attack in overconfident overconfident_extreme coordinated_extreme \
              inject_sycophant inject_deceptive subtle; do
    python3 -u -m ctc_llm.paper.selective \
        --input results/raw_results.json \
        --out-dir results \
        --tasks mmlu truthfulqa arc \
        --attack "$attack" \
        --alpha 0.10 \
        --coverage-target 0.80 \
        | tee -a results/selective.txt
done

echo ""
echo "========================================"
echo "ALL DONE: $(date)"
echo "========================================"
echo "Artifacts in results/:"
echo "  raw_results.json            MCQ data (20 seeds × 5 models × 3 tasks)"
echo "  raw_results_freeform.json   Free-form data (GSM8K + HellaSwag)"
echo "  tables.txt                  human-readable summary tables"
echo "  table_crossmodel.tex        ICLR headline LaTeX table"
echo "  table_*.tex                 per-model LaTeX tables"
echo "  fig_crossmodel.pdf          headline figure"
echo "  fig_corruption_sweep_*.pdf  per-(model,task) corruption sweeps"
echo "  fig_coverage_*.pdf          empirical coverage verification"
echo "  fig_scaling.pdf             N=5 vs N=10"
echo "  fig_attack_ablation_*.pdf   attack-type ablation"
echo "  fig_alpha_coverage_*.pdf    α sweep"
