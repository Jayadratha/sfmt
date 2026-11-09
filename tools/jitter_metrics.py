"""Temporal-jitter metrics from the final-validation meshes of each run.

Per run: val_0_23999/mixamo_{i}_{input,pred}.ply, i = 0..599; i // 60 is the clip
(the validation loop resets its window / feeds ground truth at i % 60 == 0).
  input = ground-truth target mesh (data2.v1), pred = predicted mesh.
"""
import json
import os
import sys

import numpy as np

RUNS = ["baseline", "avg_w1", "avg_w3", "weighted_w3", "avg_w5", "weighted_w5"]
ROOT = os.environ.get("SFMT_LOG_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs")
CLIP = 60
N = 600


def read_ply_vertices(path):
    with open(path, "rb") as f:
        raw = f.read()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode().splitlines()
    nv = None
    props = []
    in_vertex = False
    for line in header:
        t = line.split()
        if t[:2] == ["element", "vertex"]:
            nv = int(t[2]); in_vertex = True
        elif t[:1] == ["element"]:
            in_vertex = False
        elif t[:1] == ["property"] and in_vertex:
            props.append((t[2], {"double": "<f8", "float": "<f4", "uchar": "u1"}[t[1]]))
    dt = np.dtype(props)
    arr = np.frombuffer(raw, dtype=dt, count=nv, offset=end)
    return np.stack([arr["x"], arr["y"], arr["z"]], -1).astype(np.float64)


def load_run(name):
    d = os.path.join(ROOT, name, "val_0_23999")
    gt, pr = [], []
    for i in range(N):
        gt.append(read_ply_vertices(os.path.join(d, f"mixamo_{i}_input.ply")))
        pr.append(read_ply_vertices(os.path.join(d, f"mixamo_{i}_pred.ply")))
    return gt, pr


def clip_stack(frames, k):
    return np.stack(frames[k * CLIP:(k + 1) * CLIP])           # (60, V, 3)


def metrics(gt, pr):
    """gt, pr: (T, V, 3). Returns dict of scalars for one clip."""
    vel_p, vel_g = np.diff(pr, axis=0), np.diff(gt, axis=0)
    acc_p, acc_g = vel_p[1:] - vel_p[:-1], vel_g[1:] - vel_g[:-1]
    n = lambda x: np.linalg.norm(x, axis=-1).mean()
    return dict(
        dist=n(pr - gt),                    # mean per-vertex error (what validation reports)
        acc_pred=n(acc_p),                  # raw jitter of the output
        acc_gt=n(acc_g),                    # acceleration the ground-truth motion itself has
        acc_err=n(acc_p - acc_g),           # deviation of output acceleration from the GT's
        vel_err=n(vel_p - vel_g),           # frame-to-frame motion error
        # error that is NOT explained by per-frame error: how much the error itself flickers
        err_flicker=n(np.diff(pr - gt, axis=0)),
    )


if __name__ == "__main__":
    out = {}
    for name in RUNS:
        gt, pr = load_run(name)
        clips = []
        for k in range(N // CLIP):
            g, p = clip_stack(gt, k), clip_stack(pr, k)
            assert g.shape == p.shape, (name, k, g.shape, p.shape)
            clips.append(metrics(g, p))
        out[name] = clips
        print(name, "ok", flush=True)
    json.dump(out, open(sys.argv[1], "w"))
