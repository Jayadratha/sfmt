# Performance and reliability notes

Measured on a GTX 1650 (4 GB), see below.

Measured on a GTX 1650 (4 GB), cool GPU, per training iteration (batch of 2 pose pairs; an epoch is 36,000 iterations):

| | ms / iteration | peak GPU memory |
|---|---|---|
| original code (anomaly detection on, checkpoint written every step) | ~850 | 3.4 GB |
| phase-timed baseline after removing the debug overhead | 301 | 2.4 GB |
| + sync-free per-graph ops, GPU farthest-point sampling, no dead `data4` forward, `data3` forward without autograd, no `retain_graph`, foreach AdamW | 221 | 1.0 GB |
| + EdgeConv first layer computed per vertex instead of per edge (exact reformulation) | 199 | 0.9 GB |
| + stateless data schedule with 4 background loader workers | **181** (~5.5 it/s) | 0.9 GB |

Profile any configuration with `python train.py --window 5 --agg weighted_avg --profile 40 [--profile_ops]`
(per-phase wall time with GPU synchronisation; `--device cpu` works too).

Data schedule: sample `2k`/`2k+1` of an epoch form iteration `k`; the pose index and the characters (re-drawn every 60 iterations
from a RandomState seeded with `(seed, block)`) are pure functions of `k`. So loader workers can run in parallel, resumed runs see the
same data, and all experiments see the same character schedule.

Reliability: `latest.pth` is written every `--ckpt_every` iterations (multiple of 60) and an interrupted run resumes from there.
`run_overnight.sh` keeps outputs in `/dev/shm` and mirrors them to disk, because this machine's btrfs root stalls for up to a minute.

## Changes vs. upstream / the original experiment folders

- one `train.py` replaces the copies `sfpose_transfer_wind{1,3,5}`, `weighted_wind{3,5}`, `sfpose_transfer_wgted_wind5`
- window/aggregation/epochs/validation interval are CLI args; cluster-specific paths -> env vars
- debug `exit()` / shape prints removed
