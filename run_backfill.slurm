#!/bin/bash
# Backfill SC, Debate, MoA, LLM-Judge on existing records (CPU-only).
#SBATCH --job-name=ctc_backfill
#SBATCH --output=results/backfill_%j.log
#SBATCH --error=results/backfill_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

set -euo pipefail
cd /home/arya/kamal_new

echo "Node: $(hostname)  Date: $(date)"
echo ""

python3 -u scripts_backfill_baselines.py results/raw_results.json
echo ""
echo "Backfill done: $(date)"
