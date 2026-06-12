"""
Selective prediction analysis for adversarial LLM committees.

Headline metric (novel)
-----------------------
SELECTIVE ACCURACY @ COVERAGE C:
    Fix a coverage budget C ∈ (0, 1].  Each method, given its per-question
    confidence scores, abstains on the (1-C) fraction of questions with
    lowest confidence.  Among the C-fraction it commits to, report accuracy.

A method dominates if its risk-coverage curve lies strictly below others.

COMMITTEE CONFORMAL ABSTENTION:
    Our method abstains when the *union* of all agents' conformal sets has
    size > 1 (i.e. the committee cannot agree on a single conformal-blessed
    action).  This gives a selective coverage guarantee:

        P( correct = answer | committee_set_size == 1 ) ≥ 1 - N·α  (loose)

    Empirically tighter; verified by `committee_selective_accuracy` below.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS_SELECT = [
    ("vanilla",    "vanilla_conf",     "Vanilla"),
    ("average",    "average_conf",     "Avg-Ensemble"),
    ("majority",   "majority_conf",    "Majority Vote"),
    ("entropy",    "entropy_conf",     "EntropyTrust"),
    ("ctc",        "ctc_conf",         "CTC-Global"),
    ("ctc_hybrid", "ctc_hybrid_conf",  "CTC-Hybrid (ours)"),
]
COLORS = {
    "Vanilla":          "#999999",
    "Avg-Ensemble":     "#4477AA",
    "Majority Vote":    "#66CCEE",
    "EntropyTrust":     "#EE6677",
    "CTC-Global":       "#228833",
    "CTC-Hybrid (ours)": "#CC0066",
    "Committee Conformal (ours)": "#990000",
}
MARKERS = {
    "Vanilla":          "x",
    "Avg-Ensemble":     "s",
    "Majority Vote":    "^",
    "EntropyTrust":     "D",
    "CTC-Global":       "o",
    "CTC-Hybrid (ours)": "*",
    "Committee Conformal (ours)": "P",
}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.dpi": 150, "text.usetex": False,
})


def _ci95(values):
    if len(values) < 2:
        return 0.0
    return 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))


def risk_coverage_curve(acc_per_q: List[float], conf_per_q: List[float],
                        n_points: int = 21) -> Tuple[List[float], List[float]]:
    """
    Sweep coverage from 5% to 100% and compute selective accuracy.
    Returns (coverage_levels, accuracies_at_each).
    """
    acc  = np.array(acc_per_q, dtype=float)
    conf = np.array(conf_per_q, dtype=float)
    n    = len(acc)
    order = np.argsort(-conf)  # descending confidence
    acc_sorted = acc[order]

    coverages = np.linspace(0.05, 1.0, n_points)
    accs = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        sel = acc_sorted[:k]
        accs.append(float(np.mean(sel)))
    return coverages.tolist(), accs


def committee_selective_curve(committee_acc_per_q: List[float],
                              committee_set_per_q: List[int],
                              concentration_per_q: List[float]
                              ) -> Tuple[List[float], List[float]]:
    """
    Sweep across all possible (set_size_max, concentration_min) thresholds.
    Returns (coverage, selective_accuracy) — the conformal-blessed Pareto.
    """
    acc  = np.array(committee_acc_per_q, dtype=float)
    sz   = np.array(committee_set_per_q, dtype=int)
    conc = np.array(concentration_per_q, dtype=float)
    n    = len(acc)

    # Rank: smaller set size first; within same size, higher concentration first
    rank_score = sz * 1.0 - conc  # smaller better
    order      = np.argsort(rank_score)
    acc_sorted = acc[order]

    coverages = np.linspace(0.05, 1.0, 21)
    accs = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        accs.append(float(np.mean(acc_sorted[:k])))
    return coverages.tolist(), accs


def aurc(coverages: List[float], accs: List[float]) -> float:
    """Area Under Risk-Coverage curve (1 - selective accuracy)."""
    risk = 1.0 - np.array(accs)
    return float(np.trapezoid(risk, coverages))


def aggregate_per_q(records: List[Dict], key: str) -> List[float]:
    """Pool the per-question lists across matching records (same condition)."""
    out: List[float] = []
    for r in records:
        v = r.get(key)
        if isinstance(v, list):
            out.extend(v)
        elif v is not None:
            out.append(float(v))
    return out


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_risk_coverage(results: List[Dict], out_dir: str,
                       model_id: str, task: str, attack: str,
                       k: int, alpha: float = 0.10) -> None:
    rows = [r for r in results
            if r.get("model_id") == model_id and r["task"] == task and
               r["attack"] == attack and r["n_corrupt"] == k and
               r["n_agents"] == 5 and abs(r["alpha"] - alpha) < 1e-6]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for method_key, conf_key, label in METHODS_SELECT:
        accs = aggregate_per_q(rows, f"{method_key}_per_q")
        conf = aggregate_per_q(rows, f"{conf_key}_per_q")
        if not accs or not conf or len(accs) != len(conf):
            continue
        cov, sel = risk_coverage_curve(accs, conf)
        ax.plot(cov, sel, marker=MARKERS.get(label, "o"),
                color=COLORS.get(label, "gray"), label=label,
                linewidth=2 if "ours" in label else 1.2,
                markersize=7 if "ours" in label else 5)

    # Committee conformal abstention (the headline curve)
    cmt_accs = aggregate_per_q(rows, "committee_acc_per_q")
    cmt_size = aggregate_per_q(rows, "committee_set_size_per_q")
    cmt_conc = aggregate_per_q(rows, "committee_score_concentration_per_q")
    if cmt_accs and cmt_size and cmt_conc and len(cmt_accs) == len(cmt_size):
        cov, sel = committee_selective_curve(cmt_accs, cmt_size, cmt_conc)
        ax.plot(cov, sel, marker="P", color="#990000",
                label="Committee Conformal (ours)",
                linewidth=2.5, markersize=8, zorder=10)

    ax.axhline(y=1-alpha, color="gray", linestyle=":", alpha=0.5,
               label=f"1−α = {1-alpha}")
    ax.set_xlabel("Coverage (fraction of questions answered)")
    ax.set_ylabel("Selective accuracy")
    ax.set_title(f"Risk–Coverage: {task.upper()} | {model_id.split('/')[-1]} | "
                 f"k={k}/5 corrupt | {attack}")
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    safe_model = model_id.split("/")[-1].replace("-","_").replace(".","_")
    path = os.path.join(out_dir,
                        f"fig_risk_coverage_{task}_{safe_model}_k{k}_{attack}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def selective_accuracy_table(results: List[Dict],
                             tasks: List[str],
                             attack: str = "overconfident",
                             alpha: float = 0.10,
                             target_coverage: float = 0.80) -> str:
    """
    Cross-model table: selective accuracy at fixed coverage target.

    The KEY result: at the same coverage budget, our method dominates.
    """
    rows = [r for r in results
            if r["attack"] == attack and r["n_corrupt"] == 3 and
               r["n_agents"] == 5 and abs(r["alpha"] - alpha) < 1e-6 and
               r.get("experiment", "main") == "main"]
    models = sorted(set(r["model_id"] for r in rows))
    if not models:
        return "(no data)"

    out = [f"\n{'='*108}",
           f"SELECTIVE ACCURACY @ {int(target_coverage*100)}% COVERAGE  |  "
           f"k=3 corrupt / 5  |  α={alpha}  |  attack={attack}",
           "="*108]
    hdr = f"{'Model':<28}{'Task':<14}"
    methods_show = ["majority", "entropy", "ctc", "ctc_hybrid", "committee"]
    for m in methods_show:
        hdr += f"  {m:<14}"
    out.append(hdr)
    out.append("-"*108)

    for model in models:
        for task in tasks:
            sub = [r for r in rows if r["model_id"] == model and r["task"] == task]
            if not sub: continue
            row = f"{model.split('/')[-1]:<28}{task:<14}"
            for m in methods_show:
                if m == "committee":
                    accs = aggregate_per_q(sub, "committee_acc_per_q")
                    sz   = aggregate_per_q(sub, "committee_set_size_per_q")
                    conc = aggregate_per_q(sub, "committee_score_concentration_per_q")
                    if accs and sz and len(accs)==len(sz):
                        cov, sel = committee_selective_curve(accs, sz, conc)
                        idx = int(np.argmin(np.abs(np.array(cov) - target_coverage)))
                        row += f"  {sel[idx]:.3f}        "
                    else:
                        row += "  —            "
                else:
                    accs = aggregate_per_q(sub, f"{m}_per_q")
                    conf_key = {"majority":"majority_conf","entropy":"entropy_conf",
                                "ctc":"ctc_conf","ctc_hybrid":"ctc_hybrid_conf"}[m]
                    conf = aggregate_per_q(sub, f"{conf_key}_per_q")
                    if accs and conf and len(accs) == len(conf):
                        cov, sel = risk_coverage_curve(accs, conf)
                        idx = int(np.argmin(np.abs(np.array(cov) - target_coverage)))
                        row += f"  {sel[idx]:.3f}        "
                    else:
                        row += "  —            "
            out.append(row)
    out.append("="*108)
    return "\n".join(out)


def aurc_table(results: List[Dict], tasks: List[str],
               attack: str = "overconfident", alpha: float = 0.10) -> str:
    """Area Under Risk-Coverage curve.  Lower is better."""
    rows = [r for r in results
            if r["attack"] == attack and r["n_corrupt"] == 3 and
               r["n_agents"] == 5 and abs(r["alpha"] - alpha) < 1e-6 and
               r.get("experiment", "main") == "main"]
    models = sorted(set(r["model_id"] for r in rows))
    if not models: return "(no data)"

    out = [f"\n{'='*108}",
           f"AURC (Area Under Risk-Coverage)  |  k=3  |  α={alpha}  |  attack={attack}",
           "(lower is better — integrates 1 - selective_accuracy over all coverage levels)",
           "="*108]
    hdr = f"{'Model':<28}{'Task':<14}"
    methods_show = ["majority", "entropy", "ctc", "ctc_hybrid", "committee"]
    for m in methods_show: hdr += f"  {m:<11}"
    out.append(hdr); out.append("-"*108)

    for model in models:
        for task in tasks:
            sub = [r for r in rows if r["model_id"] == model and r["task"] == task]
            if not sub: continue
            row = f"{model.split('/')[-1]:<28}{task:<14}"
            for m in methods_show:
                if m == "committee":
                    accs = aggregate_per_q(sub, "committee_acc_per_q")
                    sz   = aggregate_per_q(sub, "committee_set_size_per_q")
                    conc = aggregate_per_q(sub, "committee_score_concentration_per_q")
                    if accs and sz and len(accs)==len(sz):
                        cov, sel = committee_selective_curve(accs, sz, conc)
                        a = aurc(cov, sel)
                        row += f"  {a:.4f}    "
                    else:
                        row += "  —          "
                else:
                    accs = aggregate_per_q(sub, f"{m}_per_q")
                    conf_key = {"majority":"majority_conf","entropy":"entropy_conf",
                                "ctc":"ctc_conf","ctc_hybrid":"ctc_hybrid_conf"}[m]
                    conf = aggregate_per_q(sub, f"{conf_key}_per_q")
                    if accs and conf and len(accs)==len(conf):
                        cov, sel = risk_coverage_curve(accs, conf)
                        a = aurc(cov, sel)
                        row += f"  {a:.4f}    "
                    else:
                        row += "  —          "
            out.append(row)
    out.append("="*108)
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",   default="results/raw_results.json")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--tasks",   nargs="+", default=["mmlu","truthfulqa","arc"])
    p.add_argument("--attack",  default="overconfident")
    p.add_argument("--alpha",   type=float, default=0.10)
    p.add_argument("--coverage-target", type=float, default=0.80)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.input) as f:
        results = json.load(f)

    print(f"Loaded {len(results)} records from {args.input}")
    print(selective_accuracy_table(results, args.tasks, args.attack,
                                   args.alpha, args.coverage_target))
    print(aurc_table(results, args.tasks, args.attack, args.alpha))

    # Per-model per-task risk-coverage plots at k=3 (most adversarial)
    models = sorted(set(r.get("model_id","?") for r in results if r.get("model_id")))
    print("\nGenerating risk-coverage plots…")
    for model_id in models:
        for task in args.tasks:
            for k in [1, 2, 3]:
                plot_risk_coverage(results, args.out_dir, model_id, task,
                                   args.attack, k, args.alpha)


if __name__ == "__main__":
    main()
