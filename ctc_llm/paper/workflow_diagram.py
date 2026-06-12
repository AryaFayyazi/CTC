"""End-to-end workflow figure for CTC (Conformal Trust Coordination)."""

from __future__ import annotations
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


def _box(ax, x, y, w, h, text, fc="#ffffff", ec="#222222", lw=1.2,
         fontsize=9, fontweight="normal"):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                         fc=fc, ec=ec, linewidth=lw)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight)


def _arrow(ax, x1, y1, x2, y2, color="#444", lw=1.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                shrinkA=2, shrinkB=2))


def _bar_dist(ax, x, y, w, h, probs, highlight_in_set=None):
    n = len(probs)
    bar_w = w / (n * 1.3)
    gap   = bar_w * 0.3
    for i, p in enumerate(probs):
        bx = x + gap/2 + i * (bar_w + gap)
        bar_h = max(0.005, float(p)) * (h - 0.06)
        color = "#228833" if (highlight_in_set is not None and i in highlight_in_set) else "#bbbbbb"
        ax.add_patch(Rectangle((bx, y + 0.03), bar_w, bar_h,
                               facecolor=color, edgecolor="#333", linewidth=0.4))
        ax.text(bx + bar_w/2, y + 0.005, "ABCD"[i],
                ha="center", va="bottom", fontsize=6.5)


def make_workflow(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 15); ax.set_ylim(0, 10)
    ax.axis("off")

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(7.5, 9.55, "Conformal Trust Coordination (CTC) — End-to-End Pipeline",
            ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(7.5, 9.15,
            "Decentralised LLM committee with provable selective coverage under Byzantine attack",
            ha="center", va="center", fontsize=10.5, color="#444")

    # ── Section labels (row above each column) ───────────────────────────────
    ax.text(2.0,  8.50, "Input", ha="center", fontsize=11, fontweight="bold")
    ax.text(5.3,  8.50, "1.  Each agent emits πᵢ(·|q)",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(9.0,  8.50, "2.  Per-agent conformal set + trust",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(12.0, 8.50, "3.  Committee aggregation",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(14.0, 8.50, "4.  Decision", ha="center", fontsize=11, fontweight="bold")

    # ── Input column ─────────────────────────────────────────────────────────
    _box(ax, 0.5, 7.2, 3.0, 1.0,
         "Question q\n(4-choice MCQ)", fc="#fff4e0", fontweight="bold",
         fontsize=10)

    _box(ax, 0.5, 5.2, 3.0, 1.5,
         "Offline Calibration\n\n"
         "  • Pool clean cal data\n"
         "  • Compute non-conformity scores\n"
         "  • q̂ = ⌈(n+1)(1−α)/n⌉ quantile\n"
         "  • Broadcast same q̂",
         fc="#e9f0ff", fontsize=9)
    ax.text(2.0, 6.85, "(once)", ha="center", fontsize=8,
            color="#3355aa", style="italic")

    # ── Column 1: Agents (5 boxes) ───────────────────────────────────────────
    agent_x, agent_w, agent_h = 4.0, 2.6, 0.85
    n_agents = 5
    spacing = 0.10
    y_top = 8.0
    sample_dists = [
        ([0.05, 0.70, 0.20, 0.05], [1],     "Agent 0  (clean)", "#e9f5e9"),
        ([0.10, 0.50, 0.30, 0.10], [1, 2],  "Agent 1  (clean)", "#e9f5e9"),
        ([0.05, 0.05, 0.85, 0.05], [2],     "Agent 2  (clean)", "#e9f5e9"),
        ([0.97, 0.01, 0.01, 0.01], [0],     "Agent 3  CORRUPT", "#ffe9e9"),
        ([0.97, 0.01, 0.01, 0.01], [0],     "Agent 4  CORRUPT", "#ffe9e9"),
    ]
    agent_ys = []
    for i, (probs, in_set, lbl, fc) in enumerate(sample_dists):
        y = y_top - i * (agent_h + spacing)
        agent_ys.append(y)
        _box(ax, agent_x, y, agent_w, agent_h, "", fc=fc, lw=0.8)
        ax.text(agent_x + 0.08, y + agent_h - 0.13, lbl,
                ha="left", va="top", fontsize=8, fontweight="bold")
        _bar_dist(ax, agent_x + 1.20, y + 0.04, 1.30, agent_h - 0.18,
                  probs, highlight_in_set=in_set)

    # arrow from question → agents
    _arrow(ax, 3.5, 7.7, agent_x, agent_ys[2] + agent_h/2 + 0.5)
    # arrow from calibration → conformal-set column (going through agents)
    _arrow(ax, 3.5, 5.95, 8.05, 5.0, color="#3355aa")

    # ── Column 2: Conformal sets ─────────────────────────────────────────────
    set_x, set_w, set_h = 8.05, 2.0, agent_h
    sample_sets = [
        ("C₀ = {B}",     "T = 1.00"),
        ("C₁ = {B, C}",  "T = 0.50"),
        ("C₂ = {C}",     "T = 1.00"),
        ("C₃ = {A}",     "T = 1.00"),
        ("C₄ = {A}",     "T = 1.00"),
    ]
    for i, (text, T_str) in enumerate(sample_sets):
        y = agent_ys[i]
        _box(ax, set_x, y, set_w, set_h, f"{text}\n{T_str}",
             fc="#fbfbfb", fontsize=10)
        _arrow(ax, agent_x + agent_w + 0.05, y + agent_h/2,
               set_x - 0.05, y + agent_h/2)

    # ── Column 3: Committee aggregation ──────────────────────────────────────
    cmt_x, cmt_w = 10.5, 3.0
    _box(ax, cmt_x, 6.6, cmt_w, 1.4,
         "Committee union  ⋃ᵢ Cᵢ\n\n"
         "Here:  {A, B, C}  →  size 3\n"
         "(size > 1  ⇒  uncertain)",
         fc="#fff0f0", fontsize=9)

    _box(ax, cmt_x, 4.6, cmt_w, 1.7,
         "CTC-Global score\n\n"
         "score(a) = Σᵢ πᵢ(a) · Tᵢ · 𝟙[a∈Cᵢ]\n\n"
         "→ argmax = B  (correct ✓)",
         fc="#e8f7ec", fontsize=9)

    # arrows from set column → committee
    _arrow(ax, set_x + set_w + 0.05, agent_ys[1] + agent_h/2,
           cmt_x, 7.30, color="#aa4444")
    _arrow(ax, set_x + set_w + 0.05, agent_ys[3] + agent_h/2,
           cmt_x, 5.45, color="#228833")

    # ── Column 4: Decision ───────────────────────────────────────────────────
    dec_x, dec_w = 13.7, 1.2
    _box(ax, dec_x, 6.6, dec_w, 1.4,
         "Decide\n\n"
         "size = 1\n→ PREDICT\n\nsize > 1\n→ ABSTAIN\n(escalate)",
         fc="#fef9e7", fontsize=8, fontweight="bold")
    _arrow(ax, cmt_x + cmt_w, 7.30, dec_x, 7.30, color="#cc8800")

    # ── Bottom row: Coverage guarantee + headline result ─────────────────────
    _box(ax, 0.5, 1.5, 4.5, 2.7,
         "Coverage guarantee\n\n"
         "• Single agent (split conformal):\n"
         "    P(a* ∈ Cᵢ) ≥ 1 − α\n\n"
         "• Committee with size-1 union:\n"
         "    P(a* = answer) ≥ 1 − N·α (loose)\n"
         "                       ≥ 1 − αᴺ (independence)\n\n"
         "• Empirical (this work):  ≥ 0.90\n"
         "  across all corruption levels k",
         fc="#f4eaff", fontsize=9.5)

    _box(ax, 5.3, 1.5, 9.5, 2.7,
         "Headline empirical result\n"
         "Setup: k = 3 corrupt of N = 5,  α = 0.10,  overconfident_extreme attack\n\n"
         "Forced prediction (no abstention) — all methods near 0.46\n\n"
         "Selective accuracy @ 50% coverage:\n\n"
         "    Majority Vote    →   0.000   (collapses)\n"
         "    EntropyTrust     →   0.689\n"
         "    CTC-Hybrid       →   0.767\n"
         "    Committee (ours) →   0.767  (+7.7 pp vs Entropy)\n\n"
         "Means: with abstention on ½ of inputs, CTC commits to 77% accurate predictions\n"
         "even with 60% of agents adversarially corrupted (vs. 46% without abstention).",
         fc="#fff7e6", fontsize=9.5)

    # ── Footnote ─────────────────────────────────────────────────────────────
    ax.text(7.5, 0.4,
            "α = miscoverage budget    T = 1/|C| = trust score    "
            "𝟙[·] = indicator    'CORRUPT' = prompt-injected adversarial agent",
            ha="center", va="center", fontsize=9, style="italic", color="#555")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/fig_workflow.pdf")
    args = p.parse_args()
    make_workflow(args.out)
    if args.out.endswith(".pdf"):
        make_workflow(args.out.replace(".pdf", ".png"))


if __name__ == "__main__":
    main()
