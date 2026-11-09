"""Tables from extra_metrics.py output.   usage: python tools/compare_runs.py metrics.json
1. mean of every metric per run
2. each windowed run vs the baseline, paired over the 10 test clips (bootstrap 95% CI, #clips where it is worse)
3. if replicate runs exist (<name>_s1): seed-to-seed spread of the baseline / window-5 versus the gap between them
"""
import json
import sys

import numpy as np

R = json.load(open(sys.argv[1]))
KEYS = ["dist", "p95", "err_early", "err_late", "acc_err", "jerk_err", "hf_err", "edge_dist"]
rng = np.random.RandomState(0)
col = lambda n, k: np.array([c[k] for c in R[n]])
mean = lambda n, k: col(n, k).mean()

print("1) mean over the 10 test clips (lower is better for every column)")
print(f"{'run':14s}" + "".join(f"{k:>11s}" for k in KEYS) + f"{'late/early':>11s}")
for n in R:
    print(f"{n:14s}" + "".join(f"{mean(n, k):11.5f}" for k in KEYS) + f"{mean(n, 'err_late') / mean(n, 'err_early'):11.3f}")

print("\n2) paired vs baseline over the clips: mean difference [95% bootstrap CI]  (clips where the run is worse)")
for n in R:
    if n in ("baseline", "baseline_s1") or n.endswith("_s1"):
        continue
    print(f"  {n}")
    for k in KEYS:
        d = col(n, k) - col("baseline", k)
        bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        sig = "  *" if lo > 0 or hi < 0 else ""
        print(f"    {k:10s}{d.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]  worse on {(d > 0).sum():2d}/{len(d)}{sig}")

pairs = [(a, b) for a, b in (("baseline", "baseline_s1"), ("avg_w5", "avg_w5_s1")) if a in R and b in R]
if len(pairs) == 2:
    print("\n3) seed replicates (seed 0 = first six runs, seed 1 = replicate; model init and data schedule both change)")
    print(f"{'metric':10s}{'base s0':>9s}{'base s1':>9s}{'w5 s0':>9s}{'w5 s1':>9s}{'seed spread base':>18s}{'seed spread w5':>16s}{'gap w5-base (s0, s1)':>24s}")
    for k in KEYS:
        b0, b1, w0, w1 = mean("baseline", k), mean("baseline_s1", k), mean("avg_w5", k), mean("avg_w5_s1", k)
        print(f"{k:10s}{b0:9.5f}{b1:9.5f}{w0:9.5f}{w1:9.5f}{abs(b0 - b1):18.5f}{abs(w0 - w1):16.5f}   {w0 - b0:+.5f}, {w1 - b1:+.5f}")
    print("\n   read: a window-5 effect is only believable if the gap has the same sign in both seeds and is larger than the seed spread.")
