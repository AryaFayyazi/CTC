"""
Paper figures.

fig_corruption_sweep_{task}.pdf — accuracy vs k for one (model, task)
fig_crossmodel.pdf              — k=3 accuracy across all (model, task) cells
fig_coverage_{task}.pdf         — empirical coverage vs 1-α (verifies guarantee)
fig_scaling.pdf                 — N=5 vs N=10 comparison
fig_attack_ablation_{task}.pdf  — methods × attack types
fig_alpha_ablation_{task}.pdf   — α sweep with coverage overlay
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["vanilla", "average", "majority", "entropy",
           "ctc_focal", "ctc_agreement", "ctc",
           "ctc_calibrated", "ctc_hybrid"]
COLORS = {
    "vanilla":        "#999999",
    "average":        "#4477AA",
    "majority":       "#66CCEE",
    "entropy":        "#EE6677",
    "ctc_focal":      "#AA3377",
    "ctc_agreement":  "#CCBB44",
    "ctc":            "#228833",
    "ctc_calibrated": "#999933",
    "ctc_hybrid":     "#CC0066",
}
LABELS = {
    "vanilla":        "Vanilla",
    "average":        "Avg-Ensemble",
    "majority":       "Majority Vote",
    "entropy":        "EntropyTrust",
    "ctc_focal":      "CTC-Focal",
    "ctc_agreement":  "CTC-Agreement",
    "ctc":            "CTC-Global",
    "ctc_calibrated": "CTC-Calibrated",
    "ctc_hybrid":     "CTC-Hybrid (ours)",
}
MARKERS = {
    "vanilla":        "x",
    "average":        "s",
    "majority":       "^",
    "entropy":        "D",
    "ctc_focal":      "v",
    "ctc_agreement":  "P",
    "ctc":            "o",
    "ctc_calibrated": "+",
    "ctc_hybrid":     "*",
}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 8, "figure.dpi": 150, "text.usetex": False,
})


def _ci95(values):
    if len(values) < 2:
        return 0.0
    return 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))


# ── Per-(model,task) corruption sweep ────────────────────────────────────────

def plot_corruption_sweep(results, out_dir, model_id, task,
                          attack="overconfident", alpha=0.10):
    rows = [r for r in results
            if r.get("model_id") == model_id and r["task"] == task and
               r["attack"] == attack and r["n_agents"] == 5 and
               abs(r["alpha"] - alpha) < 1e-6 and
               r.get("experiment", "main") == "main"]
    if not rows:
        return
    ks = sorted(set(r["n_corrupt"] for r in rows))
    n_agents = rows[0]["n_agents"]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m in METHODS:
        means, cis = [], []
        for k in ks:
            accs = [r[f"{m}_accuracy"] for r in rows if r["n_corrupt"] == k and f"{m}_accuracy" in r]
            if not accs:
                means.append(np.nan); cis.append(0)
                continue
            means.append(np.mean(accs)); cis.append(_ci95(accs))
        fracs = [k / n_agents for k in ks]
        is_ctc = (m == "ctc_hybrid")
        ax.errorbar(fracs, means, yerr=cis, label=LABELS[m],
                    color=COLORS[m], marker=MARKERS[m],
                    linewidth=2.4 if is_ctc else 1.2,
                    markersize=8 if is_ctc else 5,
                    capsize=3, zorder=5 if is_ctc else 3,
                    alpha=0.95 if is_ctc else 0.7)
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5,
               label="Majority fails (>50%)")
    ax.set_xlabel("Fraction of corrupt agents (k/N)")
    ax.set_ylabel("Team accuracy")
    ax.set_title(f"{task.upper()} — {model_id.split('/')[-1]}  ({attack}, α={alpha})")
    ax.legend(loc="lower left", framealpha=0.9, ncol=2)
    ax.set_ylim([0, 1.05]); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    short = model_id.split("/")[-1].replace("-","_").replace(".","_")
    path = os.path.join(out_dir, f"fig_corruption_sweep_{task}_{short}.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {path}")


# ── Cross-model headline figure (k=3 bar chart) ──────────────────────────────

def plot_crossmodel(results, out_dir, tasks, attack="overconfident",
                    alpha=0.10):
    rows = [r for r in results
            if r["attack"] == attack and r["n_corrupt"] == 3 and
               r["n_agents"] == 5 and abs(r["alpha"] - alpha) < 1e-6 and
               r.get("experiment", "main") == "main"]
    if not rows:
        return
    models = sorted(set(r["model_id"] for r in rows))
    methods_show = ["majority", "entropy", "ctc", "ctc_calibrated", "ctc_hybrid"]

    fig, axes = plt.subplots(1, len(tasks), figsize=(4.5*len(tasks), 4),
                             sharey=True)
    if len(tasks) == 1: axes = [axes]
    width = 0.18
    for ax, task in zip(axes, tasks):
        x = np.arange(len(models))
        for i, m in enumerate(methods_show):
            means, cis = [], []
            for model in models:
                accs = [r.get(f"{m}_accuracy") for r in rows
                        if r["model_id"] == model and r["task"] == task
                        and f"{m}_accuracy" in r]
                accs = [a for a in accs if a is not None]
                means.append(np.mean(accs) if accs else 0)
                cis.append(_ci95(accs) if len(accs) > 1 else 0)
            ax.bar(x + (i - 2.0)*width, means, width, yerr=cis,
                   capsize=3, label=LABELS[m], color=COLORS[m],
                   edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([m.split("/")[-1].replace("-Instruct","").replace("-instruct","")
                            for m in models], rotation=20, ha="right", fontsize=8)
        ax.set_title(task.upper())
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 1.0])
    axes[0].set_ylabel("Team accuracy at k=3 (60% corrupt)")
    axes[-1].legend(loc="upper right", framealpha=0.9)
    fig.suptitle("Cross-model robustness under prompt-injection attack (k=3/N=5)",
                 y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_crossmodel.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {path}")


# ── Coverage figure (verifies conformal guarantee) ───────────────────────────

def plot_coverage(results, out_dir, model_id, task):
    rows = [r for r in results
            if r.get("model_id") == model_id and r["task"] == task and
               r["attack"] == "overconfident" and
               r.get("experiment") == "ablation"]
    if not rows:
        return
    alphas = sorted(set(r["alpha"] for r in rows))
    ks     = sorted(set(r["n_corrupt"] for r in rows))

    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(ks)))
    for color, k in zip(cmap, ks):
        means, cis = [], []
        for alpha in alphas:
            sub = [r for r in rows if r["n_corrupt"]==k
                   and abs(r["alpha"]-alpha) < 1e-6]
            covs = [r["ctc_coverage"] for r in sub]
            means.append(np.mean(covs) if covs else 0)
            cis.append(_ci95(covs) if len(covs) > 1 else 0)
        ax.errorbar(alphas, means, yerr=cis, label=f"k={k}",
                    color=color, marker="o", capsize=3)

    # Theoretical guarantee 1-α
    xs = np.linspace(min(alphas)-0.01, max(alphas)+0.01, 50)
    ax.plot(xs, 1-xs, "k--", linewidth=1.5, label="Guarantee 1−α", alpha=0.7)

    ax.set_xlabel("Conformal level α")
    ax.set_ylabel("Empirical coverage P(a* ∈ C)")
    ax.set_title(f"Coverage guarantee — {task.upper()} ({model_id.split('/')[-1]})")
    ax.legend(title="Corrupt count", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([min(0.6, 1-max(alphas)-0.1), 1.02])
    fig.tight_layout()
    path = os.path.join(out_dir, f"fig_coverage_{task}.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {path}")


# ── Scaling figure (N=5 vs N=10) ─────────────────────────────────────────────

def plot_scaling(results, out_dir, model_id):
    main = [r for r in results
            if r.get("model_id") == model_id and r["task"] == "mmlu" and
               r["attack"] == "overconfident" and r["n_agents"] == 5 and
               abs(r["alpha"] - 0.10) < 1e-6 and
               r.get("experiment", "main") == "main"]
    sca  = [r for r in results
            if r.get("model_id") == model_id and r["task"] == "mmlu" and
               r.get("experiment") == "scaling"]
    if not main or not sca:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for data, n, ls in [(main, 5, "-"), (sca, 10, "--")]:
        ks = sorted(set(r["n_corrupt"] for r in data))
        for m in ["majority", "entropy", "ctc", "ctc_hybrid"]:
            means, cis = [], []
            for k in ks:
                accs = [r[f"{m}_accuracy"] for r in data if r["n_corrupt"]==k and f"{m}_accuracy" in r]
                means.append(np.mean(accs) if accs else 0)
                cis.append(_ci95(accs) if len(accs) > 1 else 0)
            fracs = [k / n for k in ks]
            ax.errorbar(fracs, means, yerr=cis,
                        label=f"{LABELS[m]} (N={n})",
                        color=COLORS[m], linestyle=ls,
                        marker=MARKERS[m],
                        linewidth=2 if m == "ctc_hybrid" else 1.2,
                        markersize=7 if m == "ctc_hybrid" else 5,
                        capsize=3)
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Fraction of corrupt agents (k/N)")
    ax.set_ylabel("Team accuracy")
    ax.set_title(f"Scaling: N=5 vs N=10 — MMLU ({model_id.split('/')[-1]})")
    ax.legend(loc="lower left", ncol=2, framealpha=0.9, fontsize=8)
    ax.set_ylim([0, 1.05]); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_scaling.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {path}")


# ── Attack ablation (CTC vs majority across attack types) ────────────────────

def plot_attack_ablation(results, out_dir, model_id, task, alpha=0.10):
    rows = [r for r in results
            if r.get("model_id") == model_id and r["task"] == task and
               abs(r["alpha"] - alpha) < 1e-6 and
               r.get("experiment") == "ablation"]
    if not rows:
        return
    attacks = sorted(set(r["attack"] for r in rows))
    fig, axes = plt.subplots(1, len(attacks), figsize=(3.8*len(attacks), 3.8),
                             sharey=True)
    if len(attacks) == 1: axes = [axes]
    for ax, attack in zip(axes, attacks):
        sub = [r for r in rows if r["attack"] == attack]
        ks = sorted(set(r["n_corrupt"] for r in sub))
        n_agents = sub[0]["n_agents"] if sub else 5
        for m in ["majority", "entropy", "ctc", "ctc_hybrid"]:
            means, cis = [], []
            for k in ks:
                accs = [r[f"{m}_accuracy"] for r in sub if r["n_corrupt"]==k and f"{m}_accuracy" in r]
                means.append(np.mean(accs) if accs else 0)
                cis.append(_ci95(accs) if len(accs) > 1 else 0)
            fracs = [k/n_agents for k in ks]
            ax.errorbar(fracs, means, yerr=cis, label=LABELS[m],
                        color=COLORS[m], marker=MARKERS[m],
                        linewidth=2 if m=="ctc_hybrid" else 1.2, capsize=3)
        ax.set_title(attack, fontsize=10)
        ax.set_xlabel("Corrupt fraction")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    axes[0].set_ylabel("Team accuracy")
    axes[-1].legend(loc="lower left", framealpha=0.9, fontsize=8)
    fig.suptitle(f"Attack ablation — {task.upper()} (α={alpha})", y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, f"fig_attack_ablation_{task}.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {path}")


# ── Alpha ablation (accuracy + coverage together) ────────────────────────────

def plot_alpha_ablation(results, out_dir, model_id, task):
    rows = [r for r in results
            if r.get("model_id") == model_id and r["task"] == task and
               r["attack"] == "overconfident" and
               r.get("experiment") == "ablation"]
    if not rows:
        return
    alphas = sorted(set(r["alpha"] for r in rows))
    ks     = sorted(set(r["n_corrupt"] for r in rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(ks)))
    for color, k in zip(cmap, ks):
        accs_m, accs_c, covs_m = [], [], []
        for a in alphas:
            sub = [r for r in rows if r["n_corrupt"]==k
                   and abs(r["alpha"]-a)<1e-6]
            acc = [r["ctc_accuracy"]  for r in sub]
            cov = [r["ctc_coverage"]  for r in sub]
            accs_m.append(np.mean(acc) if acc else 0)
            accs_c.append(_ci95(acc) if len(acc)>1 else 0)
            covs_m.append(np.mean(cov) if cov else 0)
        ax1.errorbar(alphas, accs_m, yerr=accs_c, label=f"k={k}",
                     color=color, marker="o", capsize=3)
        ax2.plot(alphas, covs_m, label=f"k={k}", color=color, marker="o")
    xs = np.linspace(min(alphas)-0.01, max(alphas)+0.01, 50)
    ax2.plot(xs, 1-xs, "k--", linewidth=1.5, label="1−α", alpha=0.7)
    ax1.set_xlabel("Conformal level α"); ax1.set_ylabel("CTC-Global accuracy")
    ax1.set_title(f"Accuracy vs α — {task.upper()}")
    ax1.legend(title="k"); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Conformal level α"); ax2.set_ylabel("Empirical coverage")
    ax2.set_title(f"Coverage vs α — {task.upper()}")
    ax2.legend(title="k"); ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.6, 1.02])
    fig.tight_layout()
    path = os.path.join(out_dir, f"fig_alpha_coverage_{task}.pdf")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",   default="results/raw_results.json")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--tasks",   nargs="+", default=["mmlu","truthfulqa","arc"])
    p.add_argument("--models",  nargs="+", default=None)
    p.add_argument("--primary-model", default="Qwen/Qwen2.5-7B-Instruct")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.input) as f:
        results = json.load(f)
    models = args.models or sorted(set(r.get("model_id","?") for r in results
                                       if r.get("model_id")))

    print("Generating figures…")
    plot_crossmodel(results, args.out_dir, args.tasks)
    for model_id in models:
        for task in args.tasks:
            plot_corruption_sweep(results, args.out_dir, model_id, task)
    for task in args.tasks:
        plot_coverage(results, args.out_dir, args.primary_model, task)
        plot_attack_ablation(results, args.out_dir, args.primary_model, task)
        plot_alpha_ablation(results, args.out_dir, args.primary_model, task)
    plot_scaling(results, args.out_dir, args.primary_model)
    print("Done.")


if __name__ == "__main__":
    main()
