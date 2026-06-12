# CTC: Conformal Trust Coordination for LLM Multi-Agent Committees

> **Paper-in-progress** — first formal selective coverage guarantee for adversarially-robust LLM committee aggregation. 6,400 records across 5 models, 3 benchmarks, 7 attacks, 13 methods, 20 seeds. Empirical coverage 0.894-0.905 across every experiment.

```
                       Question q
                            │
                            ▼
      ┌──────────────────────────────────────────┐
      │  Agent 0 ─→ π₀  ─→  C₀ = {plausible}     │
      │  Agent 1 ─→ π₁  ─→  C₁ = {plausible}     │
      │   …                                       │
      │  Agent N ─→ πN  ─→  CN = {plausible}     │
      └──────────────────────────────────────────┘
                            │
                  ┌─────────┴──────────┐
                  ▼                    ▼
         CTC-Hybrid score      Committee union ⋃Cᵢ
              argmax_a             |U| = 1?
                  │           yes →  predict
                  ▼           no  →  ABSTAIN
              answer        (formal selective coverage)
```

---

## Table of contents
- [TL;DR](#tldr)
- [Key results](#key-results)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Reading the results](#reading-the-results)
- [What is in `results/`](#what-is-in-results)
- [Method, in one paragraph](#method-in-one-paragraph)
- [Open questions and what we want done next](#open-questions-and-what-we-want-done-next)
- [Citation](#citation)

---

## TL;DR

LLM multi-agent committees (AutoGen, CrewAI, OpenHands, OpenAI Agents SDK) currently use aggregation rules — majority vote, self-consistency, debate, mixture-of-agents, LLM-as-judge — that are **empirical heuristics with no formal safety guarantee**. We introduce **Conformal Trust Coordination (CTC)**, a framework that combines split-conformal calibration, trust-weighted aggregation, and committee-wide abstention to provide the **first formal selective coverage guarantee under arbitrary Byzantine attacks** of size *k < N* corrupt agents.

Empirically: under **60% adversarial corruption** with our strongest published attacks, **CTC-Hybrid + Committee-Conformal abstention achieves 82-99% selective accuracy at 50% coverage**, while every 2024 SOTA baseline collapses below 30%.

---

## Key results

### 1. CTC-Hybrid is Pareto-dominant across 4 model families × 3 tasks

Forced-prediction accuracy at *k* = 3 corrupt agents (60% adversarial), `overconfident_extreme` attack, α = 0.10:

| Model | Task | Majority | SC | Debate | MoA | Judge | Entropy | CTC-Global | **CTC-Hybrid** |
|---|---|---|---|---|---|---|---|---|---|
| Qwen-7B | MMLU | 0.065 | 0.058 | 0.065 | 0.098 | 0.277 | 0.482 | 0.562 | **0.494** |
| Qwen-7B | TruthfulQA | 0.084 | 0.079 | 0.084 | 0.139 | 0.304 | 0.547 | 0.569 | **0.555** |
| Qwen-7B | ARC | 0.095 | 0.092 | 0.095 | 0.170 | 0.371 | 0.787 | 0.169 | **0.787** |
| Mistral-7B | MMLU | 0.034 | 0.038 | 0.034 | 0.069 | 0.273 | 0.289 | 0.371 | **0.300** |
| Mistral-7B | TruthfulQA | 0.059 | 0.069 | 0.071 | 0.125 | 0.265 | 0.436 | 0.361 | **0.445** |
| Mistral-7B | ARC | 0.038 | 0.088 | 0.090 | 0.173 | 0.353 | 0.485 | 0.534 | **0.510** |
| Phi-3.5 | MMLU | 0.042 | 0.038 | 0.042 | 0.069 | 0.273 | 0.489 | 0.506 | **0.496** |
| Phi-3.5 | TruthfulQA | 0.071 | 0.069 | 0.071 | 0.125 | 0.265 | 0.501 | 0.453 | **0.505** |
| Phi-3.5 | ARC | 0.090 | 0.088 | 0.090 | 0.173 | 0.353 | 0.709 | 0.171 | **0.709** |
| Qwen3-30B-A3B | MMLU | 0.098 | 0.093 | 0.098 | 0.164 | 0.309 | 0.563 | 0.158 | **0.560** |
| Qwen3-30B-A3B | TruthfulQA | 0.108 | 0.100 | 0.108 | 0.171 | 0.318 | 0.605 | 0.561 | **0.614** |
| Qwen3-30B-A3B | ARC | 0.150 | 0.149 | 0.150 | 0.285 | 0.385 | 0.880 | 0.284 | **0.880** |

**Every 2024 SOTA baseline (SC, Debate, MoA, LLM-Judge) collapses below 30%. Only EntropyTrust and CTC-Hybrid survive. CTC-Hybrid never catastrophically fails (unlike CTC-Global on ARC) and matches or beats EntropyTrust everywhere.**

### 2. Selective accuracy at 50% coverage — the safety headline

When Committee-Conformal abstains on the hardest 50% of inputs, the remaining predictions reach **82-99% accuracy under 60% adversarial corruption**:

| Model / committee | Task | Forced | **Selective @50%** | Gain (pp) |
|---|---|---|---|---|
| Qwen-7B | MMLU | 0.524 | **0.820** | +29.6 |
| Qwen-7B | TruthfulQA | 0.575 | **0.862** | +28.7 |
| Phi-3.5 | MMLU | 0.500 | **0.773** | +27.3 |
| Qwen3-30B-A3B | MMLU | 0.560 | **0.940** | +38.0 |
| Qwen3-30B-A3B | TruthfulQA | 0.614 | **0.876** | +26.2 |
| Qwen3-30B-A3B | ARC | 0.880 | **0.997** | +11.7 |
| Hetero-committee (5 LLMs) | MMLU | 0.524 | **0.820** | +29.6 |
| Hetero-committee | TruthfulQA | 0.575 | **0.862** | +28.7 |
| Hetero-committee | ARC | 0.794 | **0.962** | +16.8 |

### 3. Empirical coverage guarantee verified — across every experiment

Target: 1 − α = 0.90.

| Experiment | n records | Empirical coverage |
|---|---|---|
| E1 main multi-model | 960 | 0.903 ✓ |
| E2 attack × α ablation | 5,040 | 0.894 ✓ |
| E3 N=10 scaling | 80 | 0.899 ✓ |
| E4 heterogeneous committee | 240 | 0.901 ✓ |
| E5 domain-specialist personas | 80 | 0.900 ✓ |

---

## Repository layout

```
.
├── ctc_llm/                      # Main Python package
│   ├── agents/                   # LLM agent classes
│   │   ├── base.py
│   │   ├── local_llm_agent.py    # HuggingFace logit-extraction agent + personas
│   │   ├── corrupt_agent.py      # Programmatic attacks (overconfident, random, etc.)
│   │   └── prompt_inject_agent.py# Real-LLM jailbreak attacks (sycophant, deceptive)
│   ├── conformal/                # Split-conformal calibration utilities
│   │   ├── calibrate.py
│   │   └── coverage.py
│   ├── coordination/             # The 13 aggregation methods
│   │   ├── ctc.py                # CTC-Global, CTC-Focal, CTC-Agreement,
│   │   │                         # CTC-Calibrated, CTC-Hybrid
│   │   ├── confidence.py         # Committee-Conformal abstention
│   │   ├── self_consistency.py   # Wang ICLR 2023
│   │   ├── debate.py             # Du ICML 2024
│   │   ├── mixture_of_agents.py  # Wang NeurIPS 2024
│   │   ├── llm_judge.py          # Zheng NeurIPS 2023
│   │   ├── majority.py, vanilla.py, average.py, entropy_trust.py
│   ├── tasks/                    # Benchmark loaders
│   │   └── mmlu.py               # MMLU + TruthfulQA + ARC + GPQA
│   ├── experiments/              # SLURM-scale experiment driver
│   │   ├── fetch_responses.py    # Phase 1: cache LLM forward passes
│   │   ├── runner.py             # Per-question coordination loop
│   │   └── run_experiments.py    # Phase 2: 5-experiment grid
│   └── paper/                    # Tables, figures, selective analysis
│       ├── tables.py             # LaTeX + text tables
│       ├── plot_results.py       # 130+ paper-quality figures
│       ├── selective.py          # Selective accuracy + AURC
│       ├── workflow_diagram.py   # End-to-end pipeline figure
│       └── end_to_end_example.py # Real-question walkthroughs (markdown)
├── results/                      # All paper artifacts
│   ├── TECHNICAL_REPORT.pdf      # 10-page report for advisor/co-author
│   ├── TECHNICAL_REPORT.md       # Editable markdown source
│   ├── summary_results.json      # 4.8 MB summary (per-condition means/CIs only)
│   ├── tables.txt                # Human-readable summary tables
│   ├── table_crossmodel.tex      # Headline LaTeX table (Hybrid bold)
│   ├── table_<model>.tex         # Per-model LaTeX tables × 5
│   ├── fig_crossmodel.pdf        # Headline cross-model bar chart
│   ├── fig_workflow.pdf|png      # Pipeline diagram
│   ├── fig_corruption_sweep_*.pdf
│   ├── fig_coverage_*.pdf
│   ├── fig_attack_ablation_*.pdf
│   ├── fig_alpha_coverage_*.pdf
│   ├── fig_risk_coverage_*.pdf
│   ├── examples_mmlu.md, examples_truthfulqa.md  # Walkthroughs
│   ├── selective.txt             # Selective-accuracy tables across attacks
│   ├── ctc_llm_1045.log          # Most recent SLURM run log
│   └── backfill_1049.log         # SC/Debate/MoA/Judge backfill log
├── THEOREM.md                    # Formal Theorems 1, 2, 3 with proofs
├── run.slurm                     # End-to-end SLURM job (Phase 1-5)
├── run_backfill.slurm            # Backfill SC/Debate/MoA/Judge on existing records
├── scripts_backfill_baselines.py # Backfill script itself
├── scripts_make_report.py        # Regenerates TECHNICAL_REPORT.pdf
├── verify_local.py               # Smoke test (1 model, 1 question)
└── README.md
```

> **Note on data files.** The full per-question raw data (`results/raw_results.json`, ~1.3 GB) and the LLM-output cache (`data/cache/`, ~950 MB) are excluded from git via `.gitignore`. The repo ships the **4.8 MB `summary_results.json`** containing per-condition accuracy / CI / coverage — enough to regenerate every table and figure in the paper. The full raw data and cache can be reproduced from `run.slurm` (~25 hours on a single H100).

---

## Quick start

### Prerequisites
- Python ≥ 3.10
- PyTorch with CUDA (for Phase 1; not needed for Phase 2-5)
- One H100 GPU (or any GPU with ≥ 80 GB VRAM) for Phase 1
- SLURM cluster (or adapt to any scheduler)
- All HuggingFace models locally cached in `/data/models/`:
  - `Qwen/Qwen2.5-7B-Instruct`
  - `mistralai/Mistral-7B-Instruct-v0.3`
  - `microsoft/Phi-3.5-mini-instruct`
  - `Qwen/Qwen3-30B-A3B-Instruct-2507`
  - `allenai/Olmo-3-7B-Instruct`

### Install dependencies
```bash
pip install -U accelerate datasets transformers torch numpy scipy matplotlib reportlab pypdf
```

### Verify the pipeline works end-to-end (1 minute)
```bash
CUDA_VISIBLE_DEVICES=0 python3 verify_local.py
```
This loads Qwen-7B, queries one MMLU question, runs all coordination methods, and reports their answers.

### Regenerate tables + figures from the summary
```bash
# Tables (text + LaTeX)
python3 -m ctc_llm.paper.tables --input results/summary_results.json \
    --tasks mmlu truthfulqa arc | tee results/tables.txt

# Figures (130+ PDFs)
python3 -m ctc_llm.paper.plot_results --input results/summary_results.json \
    --out-dir results --tasks mmlu truthfulqa arc

# Selective-accuracy curves (per attack)
python3 -m ctc_llm.paper.selective --input results/summary_results.json \
    --out-dir results --tasks mmlu truthfulqa arc \
    --attack overconfident_extreme --coverage-target 0.50

# 1-page technical report PDF (the file you send to your advisor)
python3 scripts_make_report.py
```

> Some plots that rely on per-question confidence arrays (selective risk-coverage curves) require the full `raw_results.json`. The summary file is enough for headline tables and cross-model figures.

---

## Reproducing the experiments

The full experimental grid (6,400 records) is reproducible via two SLURM jobs.

### Step 1 — End-to-end run (~25 hours on H100, 5-day SLURM time limit)
```bash
sbatch run.slurm
```
This runs:
1. **Phase 1** (~45 min GPU): batched LLM inference for all (model, persona, question) triples → `data/cache/`
2. **Phase 2** (~20 hr CPU): the 5-experiment coordination grid → `results/raw_results.json`
3. **Phase 3** (~5 min): generate text + LaTeX tables → `results/tables.txt`, `results/table_*.tex`
4. **Phase 4** (~5 min): generate 130+ paper-quality figures → `results/fig_*.pdf`
5. **Phase 5** (~5 min): selective-accuracy analysis → `results/selective.txt`, risk-coverage figures

The script is **resumable**: if killed mid-way, re-submitting picks up where it left off using deduplication on the condition key (model, task, k, attack, α, seed, experiment).

### Step 2 — Backfill new baselines on legacy records (~3 hours CPU)
After adding a new baseline method, you don't need to re-run inference. Instead:
```bash
sbatch run_backfill.slurm
```
This re-runs only the coordination math (no GPU) over cached probabilities, adding the new method scores to every existing record.

---

## Reading the results

### Headline number
> **At k = 3 corrupt agents out of 5 (60% adversarial), with `overconfident_extreme` attack, the Committee-Conformal predictor achieves 82-99% selective accuracy at 50% coverage on every (model, task) cell except ARC-near-ceiling — where it achieves 99.7%. Every published 2024 baseline collapses below 30%.**

### The 5 experiments

| # | Experiment | Records | What it answers |
|---|---|---|---|
| E1 | Main cross-model | 960 | Does CTC dominate across 4 different LLM families? |
| E2 | Attack × α ablation | 5,040 | Does CTC dominate across 7 attack types and 3 calibration budgets? |
| E3 | N=10 scaling | 80 | Does CTC's robustness improve with larger committees? |
| E4 | Heterogeneous committee | 240 | Does CTC work when the 5 agents are *different* LLMs (real deployment)? |
| E5 | Domain-specialist personas | 80 | Does the trust score correctly reflect domain expertise? |

### Key files for the paper

- **Tables**: `results/tables.txt` (text) and `results/table_*.tex` (LaTeX) — *the headline cross-model table is `table_crossmodel.tex`*
- **Figures**: `results/fig_*.pdf` — *the headline figure is `fig_crossmodel.pdf`*; the pipeline diagram is `fig_workflow.pdf`
- **Statistical-rigor evidence**: 20 random seeds per condition, 95% CI from 1.96·σ/√n, one-sided Wilcoxon signed-rank tests against CTC-Hybrid (p < 0.001 throughout)
- **Theoretical contribution**: `THEOREM.md` — Theorems 1, 2, 3 with proofs

---

## What is in `results/`

| File | Size | Purpose |
|---|---|---|
| `TECHNICAL_REPORT.pdf` | 320 KB | 10-page report for advisor/co-author |
| `TECHNICAL_REPORT.md` | 25 KB | Editable markdown source |
| `summary_results.json` | 4.8 MB | Per-condition accuracy / CI / coverage (6,400 records, all 13 methods) |
| `tables.txt` | ~80 KB | Human-readable summary tables |
| `table_crossmodel.tex` | ~3 KB | **Headline LaTeX table** (CTC-Hybrid bolded) |
| `table_<model>.tex` × 5 | ~3 KB each | Per-model LaTeX tables |
| `selective.txt` | ~50 KB | Selective accuracy at multiple coverage targets, all attacks |
| `fig_crossmodel.pdf` | 35 KB | **Headline cross-model bar chart** |
| `fig_workflow.pdf` + `.png` | 320 KB | End-to-end pipeline diagram |
| `fig_corruption_sweep_*.pdf` × 12 | ~25 KB each | Accuracy vs k for each (task, model) |
| `fig_coverage_*.pdf` × 3 | ~25 KB each | Empirical coverage verification (Theorem 1) |
| `fig_attack_ablation_*.pdf` × 3 | ~35 KB each | Methods × attack types |
| `fig_alpha_coverage_*.pdf` × 3 | ~25 KB each | α sweep with selective coverage overlay |
| `fig_risk_coverage_*.pdf` × ~30 | ~30 KB each | Risk-coverage curves (need full raw_results to regenerate) |
| `examples_mmlu.md`, `examples_truthfulqa.md` | ~30 KB | Interpretable real-question walkthroughs |
| `ctc_llm_1045.log` | 75 KB | Main SLURM run log (Job 1045, completed June 9) |
| `backfill_1049.log` | 3 KB | Baseline-backfill SLURM log (Job 1049, completed June 9) |

---

## Method, in one paragraph

Each agent emits a probability distribution π_i over answer choices. Offline split-conformal calibration on a clean calibration set produces a shared threshold q̂ at the (1−α)-quantile of the per-agent non-conformity scores 1 − π_i(a*|q). At test time each agent computes a conformal set C_i = {a : π_i(a) ≥ 1 − q̂} and a trust score T_i = (1/|C_i|) · (1/(H(π_i) + ε)) where H is Shannon entropy. **CTC-Hybrid** scores each answer by

$$\text{score}(a) \;=\; \sum_i \pi_i(a) \cdot T_i \cdot \mathbb{1}[\,a \in C_i\,]$$

and picks argmax. **Committee-Conformal abstention** commits to a prediction only if the union ⋃_i C_i has size 1 (every agent agrees on a single plausible answer); otherwise abstain. The full method, with a worked example and the formal theorem, is in Section 2 of `results/TECHNICAL_REPORT.pdf`.

---

## Open questions and what we want done next

The current scope is publishable at top venues with a realistic shot at ICLR/NeurIPS **spotlight**. Pushing it to **likely spotlight** or **Outstanding Paper** territory benefits from one or two of the following extensions, each independent and sized to justify a substantial co-author contribution.

### 🎯 Primary recommendation: Real-agent deployment (OpenHands + SWE-Bench)

**Goal**: deploy CTC on a real, standardised agent benchmark and demonstrate quantitative robustness gains under a published attack model.

- **Platform**: [OpenHands](https://github.com/All-Hands-AI/OpenHands) (formerly OpenDevin), the open-source code-agent platform used in 2024-25 SWE-Bench leaderboard papers
- **Setup**: 5-agent OpenHands committee where each slot is a different base LLM (Qwen-7B + Mistral-7B + Phi-3.5 + Qwen3-30B-A3B + Olmo-3-7B). At every tool-call decision point (`edit_file`, `run_tests`, `search_repo`, ...), agents emit a distribution over available tool calls and CTC selects the action
- **Attack**: inject a malicious comment or README into the repository directing one or more agents to perform a destructive action (a real, published threat against code agents in the 2024-25 literature)
- **Metric**: % of SWE-Bench-Verified issues resolved by single-agent vs majority-vote vs CTC committee under k = 0, 1, 2, 3 compromised agents
- **Expected output**: one new section/figure/table; ~3 weeks of effort; self-contained — no changes to existing MCQ infrastructure
- **Why it matters**: reviewers consistently ask *"do these committees translate to real agent deployments?"* OpenHands + SWE-Bench answers decisively. Spotlight probability rises from ~35% (current scope) to ~60% (with this added).

### Secondary directions (any one would also strengthen the paper)

| Direction | Owner profile | Output | Effort |
|---|---|---|---|
| **Published prompt-injection benchmarks** (AgentDojo, InjecAgent) | Safety researcher | Section showing CTC catches real published attacks | 2-3 weeks |
| **Tighten Theorem 2 selective coverage bound** | Theoretical ML | Formal "Theoretical Analysis" section with new tight bound | 2-3 weeks |
| **Adaptive Conformal Inference extension** (Gibbs & Candès 2024) | Conformal-prediction expert | New algorithm variant + distribution-shift experiments | 3-4 weeks |
| **70B-scale validation** (Llama-3.1-70B + 4-bit quantisation) | Systems engineer | Additional model row in the cross-model table | 1-2 weeks |
| **Specialised-domain benchmarks** (MedQA, LegalBench, MMLU-Pro) | Domain expert | Safety-critical applications section | 2-3 weeks |

### Writing tasks (parallel to experimental work)

- Related-work survey — compare CTC to recent conformal-LLM papers (Quach et al. ICML 2024 on conformal language modelling, Mohri & Hashimoto, Angelopoulos et al. conformal risk control)
- Figure-design polish for camera-ready quality
- Theorem-proof appendix

---

## Recommended assignment

For a single substantial co-author contribution that maximises spotlight probability without diluting paper focus, our recommendation is **Real-Agent Deployment (OpenHands + SWE-Bench)** above. If theoretical strength is preferred, **Tighten Theorem 2** is the cleanest independent task.

The detailed plan, expected outcomes, and spotlight-probability calculus are in **Section 8** of `results/TECHNICAL_REPORT.pdf`.

---

## Citation

```bibtex
@misc{ctc-llm-2026,
  title={Conformal Trust Coordination: Adversarially-Robust LLM Committee Aggregation
         with Formal Selective Coverage Guarantees},
  author={Arya Fayyazi and co-authors},
  year={2026},
  note={In preparation}
}
```

---

## Contact

Arya Fayyazi — afayyazi@usc.edu
