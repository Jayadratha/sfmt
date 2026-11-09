"""Collect the results of the experiments into results/ (tables + figures).

usage: python tools/make_report_assets.py [runs_dir] [out_dir]        (defaults: ../runs  ./results)
needs:  <runs_dir>/logs/<run>.log (VALIDATION_DONE lines)   and   <runs_dir>/metrics_all.json (tools/extra_metrics.py)
writes: results/validation_curves.csv, results/summary.csv, results/metrics_per_clip.json,
        results/figures/validation_curves.png, results/figures/metrics_by_run.png
"""
import csv
import json
import os
import re
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "runs"))
OUT = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "results"))
ORDER = ["baseline", "baseline_s1", "avg_w1", "avg_w3", "weighted_w3", "avg_w5", "avg_w5_s1", "weighted_w5"]
LABEL = {"baseline": "baseline (no prior)", "avg_w1": "window 1, avg", "avg_w3": "window 3, avg", "weighted_w3": "window 3, weighted",
         "avg_w5": "window 5, avg", "weighted_w5": "window 5, weighted"}

# palette validated with the dataviz skill's validate_palette.js (categorical slots 1-2, light surface)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dd"
BLUE, ORANGE = "#2a78d6", "#eb6834"

os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

# ---- validation curves ---------------------------------------------------------------------------------------------
rows = []
for run in ORDER:
    for line in open(os.path.join(RUNS, "logs", f"{run}.log")):
        m = re.match(r"VALIDATION_DONE iter (\d+) frames (\d+) nonfinite (\d+) dist ([\d.]+) loss ([\d.]+)", line)
        if m:
            rows.append((run, int(m[1]), int(m[2]), int(m[3]), float(m[4]), float(m[5])))
with open(os.path.join(OUT, "validation_curves.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "iteration", "frames", "nonfinite_frames", "dist", "loss"])
    w.writerows(rows)

# ---- per-run summary ---------------------------------------------------------------------------------------------------
shutil.copy(os.path.join(RUNS, "metrics_all.json"), os.path.join(OUT, "metrics_per_clip.json"))
M = json.load(open(os.path.join(OUT, "metrics_per_clip.json")))
KEYS = ["dist", "p95", "acc_err", "jerk_err", "hf_err", "edge_dist"]
mean = lambda run, k: float(np.mean([c[k] for c in M[run]]))
with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "final_val_dist", "final_val_loss", "mean_last3_val_dist", "nonfinite_total"] + KEYS[1:])
    for run in ORDER:
        v = [r for r in rows if r[0] == run]
        w.writerow([run, v[-1][4], v[-1][5], round(float(np.mean([r[4] for r in v[-3:]])), 6), sum(r[3] for r in v)]
                   + [round(mean(run, k), 6) for k in KEYS[1:]])

# ---- figures ----------------------------------------------------------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# 1) validation distance over training: baseline vs window 5, two seeds each
fig, ax = plt.subplots(figsize=(7.2, 3.9))
spec = [("baseline", BLUE, "-", "baseline, seed 0"), ("baseline_s1", BLUE, "--", "baseline, seed 1"),
        ("avg_w5", ORANGE, "-", "window 5, seed 0"), ("avg_w5_s1", ORANGE, "--", "window 5, seed 1")]
for run, col, ls, lab in spec:
    v = [r for r in rows if r[0] == run]
    ax.plot([r[1] for r in v], [r[4] for r in v], color=col, ls=ls, lw=2, marker="o", ms=4.5, label=lab)
ax.set_ylim(0.03, 0.17)
ax.set_xlabel("training iteration")
ax.set_ylabel("validation distance (mean vertex error)")
ax.set_title("Validation distance: baseline vs. window 5 (two seeds each)", loc="left", color=INK, fontsize=11)
ax.legend(frameon=False, ncol=2, loc="upper right", labelcolor=INK2)
style(ax)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "figures", "validation_curves.png"), dpi=160)
plt.close(fig)

# 2) final-checkpoint metrics per run (one panel per metric, never a shared axis); replicate seeds shown as separate dots
panels = [("dist", "distance to ground truth"), ("acc_err", "acceleration error (jitter)"),
          ("hf_err", "fast flicker (period <= 4 frames)"), ("edge_dist", "edge-length distortion")]
groups = ["baseline", "avg_w1", "avg_w3", "weighted_w3", "avg_w5", "weighted_w5"]
seed1 = {"baseline": "baseline_s1", "avg_w5": "avg_w5_s1"}
fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.6), sharey=True)
for ax, (k, title) in zip(axes, panels):
    for y, g in enumerate(groups):
        col = BLUE if g == "baseline" else ORANGE
        ax.plot([mean(g, k)], [y], "o", color=col, ms=9, mec=SURFACE, mew=1.5, zorder=3)
        if g in seed1:
            ax.plot([mean(seed1[g], k)], [y], "o", color=col, ms=9, mfc=SURFACE, mew=2, zorder=3)
    ax.axvline(mean("baseline", k), color=BLUE, lw=1, alpha=.35, zorder=1)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1.12 * max(mean(r, k) for r in groups + list(seed1.values())))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
axes[0].set_yticks(range(len(groups)))
axes[0].set_yticklabels([LABEL[g] for g in groups])
axes[0].invert_yaxis()
fig.text(0.01, 0.01, "lower is better.  filled dot = seed 0, hollow dot = seed 1 (only baseline and window 5 were repeated).  thin line = baseline seed 0.",
         color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(OUT, "figures", "metrics_by_run.png"), dpi=160)
plt.close(fig)
print("wrote", OUT)
