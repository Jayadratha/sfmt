"""Extended per-clip metrics from the final-validation meshes (val_0_23999) of each run.

usage: python tools/extra_metrics.py out.json [run ...]     (default: all runs found in $SFMT_LOG_DIR / ../runs)
All distances are in mesh units (characters are normalised); "gt" = ground-truth target mesh, "pred" = prediction.
A clip is 60 consecutive validation frames; the first frame of a clip is predicted from ground truth, the rest autoregressively.

per clip:
  dist           mean per-vertex error
  dist_norm      same, divided by the ground-truth bounding-box diagonal (scale free)
  dist_centered  error after removing each frame's centroid offset (articulation error without global drift)
  p95            per-frame 95th-percentile vertex error (tail), averaged over frames
  err_early/mid/late   mean error of frames 0-9 / 10-29 / 30-59 (does the error grow along the rollout?)
  acc_pred, acc_gt, acc_err   see jitter_metrics.py (acceleration of output / of ground truth / their difference)
  jerk_err       |third difference of (pred - gt)|, a stricter smoothness measure
  hf_err         part of the error signal that flickers fast: error trajectory per vertex, components with period <= 4 frames
  edge_dist      mean |edge_pred - edge_gt| / mean edge_gt   (local shape distortion of the output)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jitter_metrics import CLIP, N, ROOT, clip_stack, metrics, read_ply_vertices   # noqa: E402

HF_FROM = CLIP // 4          # rfft bin index; bin k <-> period CLIP/k frames, so k >= 15 <-> period <= 4 frames


def read_ply_faces(path):
    raw = open(path, "rb").read()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    nv = nf = None
    stride = 0
    in_vertex = False
    for line in raw[:end].decode().splitlines():
        t = line.split()
        if t[:2] == ["element", "vertex"]:
            nv, in_vertex = int(t[2]), True
        elif t[:2] == ["element", "face"]:
            nf, in_vertex = int(t[2]), False
        elif t[:1] == ["property"] and in_vertex:
            stride += {"double": 8, "float": 4, "uchar": 1}[t[1]]
    f = np.frombuffer(raw, dtype=np.dtype([("n", "u1"), ("i", "<u4", 3)]), count=nf, offset=end + nv * stride)
    assert (f["n"] == 3).all()
    return f["i"].astype(np.int64)


def run_dir(name):
    return os.path.join(ROOT, name, "val_0_23999")


def clip_metrics(g, p, faces):
    """g, p: (T, V, 3)."""
    m = metrics(g, p)
    m.pop("err_flicker")                                   # identical to vel_err
    nrm = lambda x: np.linalg.norm(x, axis=-1)
    err = nrm(p - g)                                       # (T, V)
    diag = np.linalg.norm(g.max(1) - g.min(1), axis=-1)    # (T,)
    m["dist_norm"] = float((err.mean(1) / diag).mean())
    m["dist_centered"] = float(nrm((p - p.mean(1, keepdims=True)) - (g - g.mean(1, keepdims=True))).mean())
    m["p95"] = float(np.percentile(err, 95, axis=1).mean())
    e_t = err.mean(1)
    m["err_early"], m["err_mid"], m["err_late"] = float(e_t[:10].mean()), float(e_t[10:30].mean()), float(e_t[30:].mean())
    r = p - g
    m["jerk_err"] = float(nrm(np.diff(r, n=3, axis=0)).mean())
    R = np.fft.rfft(r - r.mean(0, keepdims=True), axis=0)
    R[:HF_FROM] = 0
    m["hf_err"] = float(nrm(np.fft.irfft(R, n=r.shape[0], axis=0)).mean())
    e = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1), axis=0)
    lp = nrm(p[:, e[:, 0]] - p[:, e[:, 1]])
    lg = nrm(g[:, e[:, 0]] - g[:, e[:, 1]])
    m["edge_dist"] = float((np.abs(lp - lg).mean(1) / lg.mean(1)).mean())
    return m


def run_metrics(name):
    d = run_dir(name)
    out = []
    for k in range(N // CLIP):
        idx = range(k * CLIP, (k + 1) * CLIP)
        g = np.stack([read_ply_vertices(os.path.join(d, f"mixamo_{i}_input.ply")) for i in idx])
        p = np.stack([read_ply_vertices(os.path.join(d, f"mixamo_{i}_pred.ply")) for i in idx])
        faces = read_ply_faces(os.path.join(d, f"mixamo_{k * CLIP}_input.ply"))
        out.append(clip_metrics(g, p, faces))
    return out


if __name__ == "__main__":
    runs = sys.argv[2:] or sorted(n for n in os.listdir(ROOT) if os.path.isdir(run_dir(n)))
    res = {}
    for n in runs:
        res[n] = run_metrics(n)
        print(n, "ok", flush=True)
    json.dump(res, open(sys.argv[1], "w"))
