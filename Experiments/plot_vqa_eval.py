#!/usr/bin/env python3
"""
plot_vqa_eval.py

Plots for vqa_eval.json produced by your evaluator.

Works for:
- a single file: logs/.../vqa_eval.json
- multiple runs: a glob like "logs/*/vqa_eval.json" to compare runs

Outputs (PNG) into --outdir:
1) per_preprocessor_accuracy.png
2) per_preprocessor_accuracy_with_ci.png
3) score_distribution_by_preprocessor.png
4) accuracy_histogram_overall.png
5) run_comparison_overall_accuracy.png          (only if multiple input files)
6) run_comparison_per_preprocessor.png          (only if multiple input files)

Notes:
- Uses matplotlib only (no seaborn).
- Does not set explicit colors.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import matplotlib.pyplot as plt


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class DetailRow:
    preprocessor: str
    sample_id: str
    question: str
    accuracy: float


@dataclass(frozen=True)
class RunEval:
    path: Path
    run_name: str
    overall_accuracy: float
    per_preprocessor: Dict[str, Dict[str, float]]  # {"n": int, "accuracy": float}
    details: List[DetailRow]


# ----------------------------
# Utilities
# ----------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_run_name(path: Path) -> str:
    # logs/<run_folder>/vqa_eval.json -> <run_folder>
    if path.parent.name:
        return path.parent.name
    return path.stem


def _load_run(path: Path) -> RunEval:
    obj = _read_json(path)
    summary = obj.get("summary", {})
    details_raw = obj.get("details", [])

    details: List[DetailRow] = []
    for r in details_raw:
        details.append(
            DetailRow(
                preprocessor=str(r.get("preprocessor", "UNKNOWN")),
                sample_id=str(r.get("sample_id", "")),
                question=str(r.get("question", "")),
                accuracy=float(r.get("accuracy_official_vqa", 0.0)),
            )
        )

    return RunEval(
        path=path,
        run_name=_infer_run_name(path),
        overall_accuracy=float(summary.get("overall_accuracy_official_vqa", 0.0)),
        per_preprocessor=summary.get("per_preprocessor", {}),
        details=details,
    )


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, outpath: Path) -> None:
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def _group_details_by_preprocessor(details: List[DetailRow]) -> Dict[str, List[float]]:
    grouped: Dict[str, List[float]] = {}
    for d in details:
        grouped.setdefault(d.preprocessor, []).append(d.accuracy)
    return grouped


def _bin_score(a: float) -> str:
    # Official VQAEval yields averages of {0, 0.333..., 0.666..., 1.0} but sometimes shows as 0.3/0.6 in your JSON.
    # Bin robustly by nearest.
    bins = [(0.0, "0"), (1.0 / 3.0, "0.33"), (2.0 / 3.0, "0.66"), (1.0, "1.0")]
    best = min(bins, key=lambda t: abs(a - t[0]))
    return best[1]


def _bootstrap_ci_mean(values: List[float], iters: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    # Simple bootstrap with replacement, using Python's random (kept deterministic via seed).
    import random
    if not values:
        return (0.0, 0.0)
    rnd = random.Random(1337)
    n = len(values)
    means = []
    for _ in range(iters):
        samp = [values[rnd.randrange(n)] for _ in range(n)]
        means.append(sum(samp) / n)
    means.sort()
    lo_idx = int((alpha / 2.0) * iters)
    hi_idx = int((1.0 - alpha / 2.0) * iters) - 1
    return means[lo_idx], means[hi_idx]


# ----------------------------
# Plot functions (single run)
# ----------------------------

def plot_per_preprocessor_accuracy(run: RunEval, outdir: Path) -> None:
    keys = sorted(run.per_preprocessor.keys())
    accs = [float(run.per_preprocessor[k]["accuracy"]) for k in keys]
    ns = [int(run.per_preprocessor[k]["n"]) for k in keys]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.bar(keys, accs)
    ax.set_title(f"Accuracy per preprocessor ({run.run_name})")
    ax.set_ylabel("Accuracy (official VQA)")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=35)

    # annotate n on top
    for i, (a, n) in enumerate(zip(accs, ns)):
        ax.text(i, min(a + 0.02, 1.02), f"n={n}", ha="center", va="bottom", fontsize=9)

    _save(fig, outdir / "per_preprocessor_accuracy.png")


def plot_per_preprocessor_accuracy_with_ci(run: RunEval, outdir: Path) -> None:
    grouped = _group_details_by_preprocessor(run.details)
    keys = sorted(grouped.keys())
    means = []
    err_lo = []
    err_hi = []

    for k in keys:
        vals = grouped[k]
        m = sum(vals) / len(vals) if vals else 0.0
        lo, hi = _bootstrap_ci_mean(vals)
        means.append(m)
        err_lo.append(m - lo)
        err_hi.append(hi - m)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.bar(keys, means, yerr=[err_lo, err_hi], capsize=4)
    ax.set_title(f"Accuracy per preprocessor with 95% bootstrap CI ({run.run_name})")
    ax.set_ylabel("Accuracy (official VQA)")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=35)

    _save(fig, outdir / "per_preprocessor_accuracy_with_ci.png")


def plot_score_distribution_by_preprocessor(run: RunEval, outdir: Path) -> None:
    grouped = _group_details_by_preprocessor(run.details)
    keys = sorted(grouped.keys())
    bins = ["0", "0.33", "0.66", "1.0"]

    # counts[bin][preproc_index]
    counts = {b: [] for b in bins}
    for k in keys:
        vals = grouped[k]
        c = {b: 0 for b in bins}
        for a in vals:
            c[_bin_score(a)] += 1
        for b in bins:
            counts[b].append(c[b])

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    bottom = [0] * len(keys)
    for b in bins:
        ax.bar(keys, counts[b], bottom=bottom, label=b)
        bottom = [bottom[i] + counts[b][i] for i in range(len(keys))]

    ax.set_title(f"Discrete score distribution per preprocessor ({run.run_name})")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Binned score")

    _save(fig, outdir / "score_distribution_by_preprocessor.png")


def plot_accuracy_histogram_overall(run: RunEval, outdir: Path) -> None:
    vals = [d.accuracy for d in run.details]
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=[-0.01, 0.17, 0.5, 0.83, 1.01])  # roughly separates 0, 0.33, 0.66, 1.0
    ax.set_title(f"Overall per-sample accuracy histogram ({run.run_name})")
    ax.set_xlabel("Accuracy (official VQA)")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1.0)

    _save(fig, outdir / "accuracy_histogram_overall.png")


# ----------------------------
# Plot functions (multi-run)
# ----------------------------

def plot_run_comparison_overall_accuracy(runs: List[RunEval], outdir: Path) -> None:
    runs_sorted = sorted(runs, key=lambda r: r.run_name)
    labels = [r.run_name for r in runs_sorted]
    accs = [r.overall_accuracy for r in runs_sorted]

    fig = plt.figure(figsize=(max(8, 0.45 * len(labels)), 5))
    ax = fig.add_subplot(111)
    ax.bar(labels, accs)
    ax.set_title("Overall accuracy per run")
    ax.set_ylabel("Accuracy (official VQA)")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=35)

    _save(fig, outdir / "run_comparison_overall_accuracy.png")


def plot_run_comparison_per_preprocessor(runs: List[RunEval], outdir: Path) -> None:
    # Collect union of preprocessors
    all_p = sorted({p for r in runs for p in r.per_preprocessor.keys()})
    runs_sorted = sorted(runs, key=lambda r: r.run_name)

    # For each preprocessor, make a line plot across runs
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    x = list(range(len(runs_sorted)))
    x_labels = [r.run_name for r in runs_sorted]

    for p in all_p:
        ys = []
        for r in runs_sorted:
            if p in r.per_preprocessor:
                ys.append(float(r.per_preprocessor[p]["accuracy"]))
            else:
                ys.append(float("nan"))
        ax.plot(x, ys, marker="o", label=p)

    ax.set_title("Per-preprocessor accuracy across runs")
    ax.set_ylabel("Accuracy (official VQA)")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, x_labels, rotation=35, ha="right")
    ax.legend(loc="best", fontsize=8)

    _save(fig, outdir / "run_comparison_per_preprocessor.png")


# ----------------------------
# Entry point
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default=r"logs\2026-02-10_16-20-05_GPT_Realtime\vqa_eval.json",
        help="Path to vqa_eval.json OR glob like logs/*/vqa_eval.json",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default=r"outputs\vqa_eval_plots",
        help="Output directory for plots",
    )
    ap.add_argument(
        "--bootstrap-iters",
        type=int,
        default=2000,
        help="Bootstrap iterations for CI plot",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    _ensure_outdir(outdir)

    paths = sorted(Path().glob(args.input)) if any(ch in args.input for ch in ["*", "?", "["]) else [Path(args.input)]
    paths = [p for p in paths if p.exists()]

    if not paths:
        print(f"Error: no files found for input={args.input}", file=sys.stderr)
        return 2

    runs = [_load_run(p) for p in paths]

    # Single-run plots for each run (into subfolders)
    for r in runs:
        run_out = outdir / r.run_name
        _ensure_outdir(run_out)
        plot_per_preprocessor_accuracy(r, run_out)
        plot_per_preprocessor_accuracy_with_ci(r, run_out)
        plot_score_distribution_by_preprocessor(r, run_out)
        plot_accuracy_histogram_overall(r, run_out)

    # Multi-run comparison plots (only if multiple)
    if len(runs) > 1:
        plot_run_comparison_overall_accuracy(runs, outdir)
        plot_run_comparison_per_preprocessor(runs, outdir)

    print(f"Wrote plots to: {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
