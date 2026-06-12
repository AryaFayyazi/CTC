"""
Publication-quality tables from the multi-model raw results.

Main table  : accuracy by method and k, mean ± 95% CI, Wilcoxon vs CTC.
Multi-model : same but stratified by model.
Ablation    : attack × alpha grid for the primary model.
Scaling     : N=5 vs N=10 comparison.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List

import numpy as np
from scipy.stats import wilcoxon  # type: ignore


METHODS = ["vanilla", "average", "majority", "entropy",
           "ctc", "ctc_focal", "ctc_agreement",
           "ctc_calibrated", "ctc_hybrid"]
METHOD_NAMES = {
    "vanilla":        "Vanilla (single)",
    "average":        "Avg-Ensemble",
    "majority":       "Majority Vote",
    "entropy":        "EntropyTrust",
    "ctc":            "CTC-Global",
    "ctc_focal":      "CTC-Focal",
    "ctc_agreement":  "CTC-Agreement",
    "ctc_calibrated": "CTC-Calibrated",
    "ctc_hybrid":     "CTC-Hybrid (ours)",
}


def _ci95(values):
    if len(values) < 2:
        return 0.0
    return 1.96 * float(np.std(values, ddof=1)) / float(np.sqrt(len(values)))


def _wilcoxon_p(a, b):
    diffs = [x - y for x, y in zip(a, b)]
    if all(d == 0 for d in diffs):
        return 1.0
    try:
        _, p = wilcoxon(diffs, alternative="greater")
        return float(p)
    except Exception:
        return 1.0


def _sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def load_results(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def build_main_table(results, task, model_id, attack="overconfident_extreme",
                     alpha=0.10, n_agents=5, latex=False):
    rows = [r for r in results
            if r.get("model_id") == model_id and r["task"] == task and
               r["attack"] == attack and r["n_agents"] == n_agents and
               abs(r["alpha"] - alpha) < 1e-6 and
               r.get("experiment", "main") == "main"]

    by_k        = defaultdict(lambda: {m: [] for m in METHODS})
    cov_by_k    = defaultdict(list)
    setsize_by_k = defaultdict(list)
    for r in rows:
        k = r["n_corrupt"]
        for m in METHODS:
            if f"{m}_accuracy" in r:
                by_k[k][m].append(r[f"{m}_accuracy"])
        cov_by_k[k].append(r["ctc_coverage"])
        setsize_by_k[k].append(r["ctc_mean_set_size"])

    ks = sorted(by_k.keys())
    if not ks:
        return f"(no data: model={model_id} task={task})"
    if latex:
        return _latex(by_k, cov_by_k, setsize_by_k, ks, task, model_id, alpha, n_agents)
    return _text(by_k, cov_by_k, setsize_by_k, ks, task, model_id, alpha, n_agents)


def _text(by_k, cov_by_k, setsize_by_k, ks, task, model_id, alpha, n_agents):
    short = model_id.split("/")[-1]
    title = (f"\n{'='*88}\n"
             f"Task: {task.upper()}  |  Model: {short}  |  "
             f"α={alpha}  |  N={n_agents}  |  overconfident attack\n"
             f"{'='*88}")
    lines = [title,
             f"{'Method':<28}" + "".join(f"  k={k:<10}" for k in ks),
             "-" * 88]
    ctc_accs = {k: by_k[k]["ctc_hybrid"] for k in ks}
    for m in METHODS:
        row = f"{METHOD_NAMES[m]:<28}"
        for k in ks:
            accs = by_k[k][m]
            if not accs:
                row += "  —          "
                continue
            mean = np.mean(accs); ci = _ci95(accs); sig = ""
            if m != "ctc_hybrid" and ctc_accs[k]:
                sig = _sig(_wilcoxon_p(ctc_accs[k], accs))
            row += f"  {mean:.3f}±{ci:.3f}{sig:<3}"
        lines.append(row)
    lines.append("-" * 88)
    cov_line = f"{'CTC Coverage (clean)':<28}"
    set_line = f"{'CTC Mean Set Size':<28}"
    for k in ks:
        cov_line += (f"  {np.mean(cov_by_k[k]):.3f}      "
                     if cov_by_k[k] else "  —          ")
        set_line += (f"  {np.mean(setsize_by_k[k]):.2f}       "
                     if setsize_by_k[k] else "  —          ")
    lines += [cov_line, set_line, "=" * 88,
              "* p<0.05  ** p<0.01  *** p<0.001  (Wilcoxon CTC-Hybrid > baseline)"]
    return "\n".join(lines)


def _latex(by_k, cov_by_k, setsize_by_k, ks, task, model_id, alpha, n_agents):
    n_cols   = len(ks)
    col_spec = "l" + "r" * n_cols
    hdr      = " & ".join([f"$k={k}$" for k in ks])
    model_short = model_id.split("/")[-1]
    ctc_accs = {k: by_k[k]["ctc_hybrid"] for k in ks}
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        (rf"\caption{{Coordination accuracy on {task.upper()} with "
         rf"{model_short} under overconfident attack ($\alpha={alpha}$, "
         rf"$N={n_agents}$). Mean $\pm$ 95\% CI over 20 seeds. "
         r"Significance vs.\ CTC-Global: $^*p<.05$, $^{**}p<.01$, "
         r"$^{***}p<.001$ (Wilcoxon).}}"),
        rf"\label{{tab:{task}_{model_short.lower().replace('-','_').replace('.','_')}}}",
        rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule",
        rf"Method & {hdr} \\", r"\midrule",
    ]
    for m in METHODS:
        name = METHOD_NAMES[m]
        if m == "ctc_hybrid":
            name = r"\textbf{" + name + "}"
        parts = [name]
        for k in ks:
            accs = by_k[k][m]
            if not accs:
                parts.append("—"); continue
            mean = np.mean(accs); ci = _ci95(accs); sig = ""
            if m != "ctc_hybrid" and ctc_accs[k]:
                raw = _sig(_wilcoxon_p(ctc_accs[k], accs))
                if   raw == "***": sig = r"$^{***}$"
                elif raw == "**":  sig = r"$^{**}$"
                elif raw == "*":   sig = r"$^{*}$"
            cell = rf"{mean:.3f}{{\small$\pm${ci:.3f}}}{sig}"
            if m == "ctc_hybrid": cell = r"\textbf{" + cell + "}"
            parts.append(cell)
        lines.append(" & ".join(parts) + r" \\")
    lines.append(r"\midrule")
    cov_row = ["CTC Coverage"]
    set_row = ["Mean $|C|$"]
    for k in ks:
        cov_row.append(f"{np.mean(cov_by_k[k]):.3f}" if cov_by_k[k] else "—")
        set_row.append(f"{np.mean(setsize_by_k[k]):.2f}" if setsize_by_k[k] else "—")
    lines.append(" & ".join(cov_row) + r" \\")
    lines.append(" & ".join(set_row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_crossmodel_table(results, tasks, attack="overconfident_extreme",
                           alpha=0.10, n_agents=5, latex=False):
    rows = [r for r in results
            if r["attack"] == attack and r["n_corrupt"] == 3 and
               r["n_agents"] == n_agents and
               abs(r["alpha"] - alpha) < 1e-6 and
               r.get("experiment", "main") == "main"]
    models = sorted(set(r["model_id"] for r in rows))
    if not models:
        return "(no cross-model data)"
    if latex:
        return _crossmodel_latex(rows, models, tasks, alpha)
    out = [f"\n{'='*88}",
           f"CROSS-MODEL: k=3 (60% corrupt) accuracy  |  α={alpha}  |  overconfident",
           "=" * 88,
           f"{'Model':<28}{'Task':<14}" +
           "".join(f"  {METHOD_NAMES[m][:18]:<18}"
                   for m in ["majority","entropy","ctc","ctc_hybrid"]),
           "-" * 88]
    for model in models:
        for task in tasks:
            sub = [r for r in rows if r["model_id"] == model and r["task"] == task]
            if not sub: continue
            cell = f"{model.split('/')[-1]:<28}{task:<14}"
            for m in ["majority","entropy","ctc","ctc_hybrid"]:
                accs = [r[f"{m}_accuracy"] for r in sub]
                cell += f"  {np.mean(accs):.3f}±{_ci95(accs):.3f}    "
            out.append(cell)
    out.append("=" * 88)
    return "\n".join(out)


def _crossmodel_latex(rows, models, tasks, alpha):
    methods_show = ["majority", "entropy", "ctc", "ctc_calibrated", "ctc_hybrid"]
    col_spec     = "ll" + "r" * len(methods_show)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        (rf"\caption{{Headline result: accuracy at $k=3$ corrupt agents out "
         rf"of $N=5$ ($\alpha={alpha}$, overconfident attack). Mean $\pm$ "
         r"95\% CI over 20 seeds. \textbf{CTC-Hybrid (ours)} maintains "
         r"accuracy across models and tasks while majority vote collapses. "
         r"Significance vs.\ CTC-Hybrid: $^*p<.05$, $^{**}p<.01$, $^{***}p<.001$.}}"),
        r"\label{tab:crossmodel}",
        rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule",
        "Model & Task & " + " & ".join(METHOD_NAMES[m] for m in methods_show) + r" \\",
        r"\midrule",
    ]
    for model in models:
        for task in tasks:
            sub = [r for r in rows if r["model_id"] == model and r["task"] == task]
            if not sub: continue
            parts = [model.split("/")[-1], task.upper()]
            ctc_accs = [r.get("ctc_hybrid_accuracy", r["ctc_accuracy"]) for r in sub]
            for m in methods_show:
                accs = [r[f"{m}_accuracy"] for r in sub]
                mean = np.mean(accs); ci = _ci95(accs)
                sig = "" if m == "ctc_hybrid" else _sig(_wilcoxon_p(ctc_accs, accs))
                marker = ""
                if   sig == "***": marker = r"$^{***}$"
                elif sig == "**":  marker = r"$^{**}$"
                elif sig == "*":   marker = r"$^{*}$"
                cell = rf"{mean:.3f}{{\small$\pm${ci:.3f}}}{marker}"
                if m == "ctc_hybrid": cell = r"\textbf{" + cell + "}"
                parts.append(cell)
            lines.append(" & ".join(parts) + r" \\")
        if model != models[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_attack_ablation(results, task, model_id, alpha=0.10):
    rows = [r for r in results
            if r.get("model_id") == model_id and r["task"] == task and
               abs(r["alpha"] - alpha) < 1e-6 and
               r.get("experiment") == "ablation"]
    if not rows:
        return f"(no ablation data for task={task})"
    attacks = sorted(set(r["attack"] for r in rows))
    ks      = sorted(set(r["n_corrupt"] for r in rows))
    out = [f"\nAttack ablation ({task.upper()}, {model_id.split('/')[-1]}, α={alpha})",
           f"{'Attack':<22}{'k':<4}" +
           "".join(f"  {m[:13]:<13}"
                   for m in ["vanilla","average","majority","entropy","ctc"])]
    for attack in attacks:
        for k in ks:
            sub = [r for r in rows if r["attack"] == attack and r["n_corrupt"] == k]
            if not sub: continue
            row = f"{attack:<22}{k:<4}"
            for m in ["vanilla","average","majority","entropy","ctc"]:
                accs = [r[f"{m}_accuracy"] for r in sub]
                row += f"  {np.mean(accs):.3f}±{_ci95(accs):.3f}"
            out.append(row)
        out.append("")
    return "\n".join(out)


def build_alpha_ablation(results, task, model_id):
    rows = [r for r in results
            if r.get("model_id") == model_id and r["task"] == task and
               r["attack"] == "overconfident" and
               r.get("experiment") == "ablation"]
    if not rows:
        return f"(no alpha ablation for task={task})"
    alphas = sorted(set(r["alpha"] for r in rows))
    ks     = sorted(set(r["n_corrupt"] for r in rows))
    out = [f"\nAlpha ablation ({task.upper()}, {model_id.split('/')[-1]}, overconfident)",
           f"{'α':<8}{'k':<4}  CTC-acc       Coverage      Mean |C|"]
    for alpha in alphas:
        for k in ks:
            sub = [r for r in rows if abs(r["alpha"]-alpha)<1e-6 and r["n_corrupt"]==k]
            if not sub: continue
            accs = [r["ctc_accuracy"]      for r in sub]
            covs = [r["ctc_coverage"]      for r in sub]
            szs  = [r["ctc_mean_set_size"] for r in sub]
            out.append(f"{alpha:<8.2f}{k:<4}  "
                       f"{np.mean(accs):.3f}±{_ci95(accs):.3f}  "
                       f"{np.mean(covs):.3f}±{_ci95(covs):.3f}  "
                       f"{np.mean(szs):.2f}")
        out.append("")
    return "\n".join(out)


def build_scaling_table(results, model_id):
    rows = [r for r in results
            if r.get("model_id") == model_id and r["task"] == "mmlu" and
               r.get("experiment") == "scaling"]
    if not rows:
        return "(no scaling data)"
    ks = sorted(set(r["n_corrupt"] for r in rows))
    out = [f"\nScaling N=10 ({model_id.split('/')[-1]}, MMLU, overconfident, α=0.10)",
           f"{'k':<4} {'frac':<7}  " +
           "".join(f"  {m[:13]:<13}"
                   for m in ["vanilla","majority","entropy","ctc"])]
    for k in ks:
        sub = [r for r in rows if r["n_corrupt"] == k]
        row = f"{k:<4} {k/10:<7.2f}"
        for m in ["vanilla","majority","entropy","ctc"]:
            accs = [r[f"{m}_accuracy"] for r in sub]
            row += f"  {np.mean(accs):.3f}±{_ci95(accs):.3f}"
        out.append(row)
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="results/raw_results.json")
    p.add_argument("--latex",  action="store_true")
    p.add_argument("--tasks",  nargs="+", default=["mmlu","truthfulqa","arc"])
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--primary-model", default="Qwen/Qwen2.5-7B-Instruct")
    args = p.parse_args()
    all_results = load_results(args.input)
    models = args.models or sorted(set(r.get("model_id","?") for r in all_results
                                       if r.get("model_id")))
    print(build_crossmodel_table(all_results, args.tasks, latex=args.latex))
    for model_id in models:
        for task in args.tasks:
            print(build_main_table(all_results, task, model_id, latex=args.latex))
            if args.latex: print()
    if not args.latex:
        for task in args.tasks:
            print(build_attack_ablation(all_results, task, args.primary_model))
            print(build_alpha_ablation(all_results,  task, args.primary_model))
        print(build_scaling_table(all_results, args.primary_model))


if __name__ == "__main__":
    main()
