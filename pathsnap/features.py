"""Candidate generation and feature engineering.

For every GPS observation we propose the K nearest road arcs (directed) as
candidates. Each (observation, candidate) pair becomes one row; the learning
task is binary: "is this candidate the reference arc?". The classifier's
probabilities later become the HMM emission scores.

No absolute coordinates or ids are used as features, so the model transfers
between cities/countries (every trip is a different place).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .data import Trip, point_segment

K_CAND = 12          # candidates per observation (directed arcs)
SAMPLE_STEP = 8.0    # metres between densified sample points on each arc
KNN_SAMPLES = 160    # sample points fetched from the KD-tree per observation
WINDOW_OFFSETS = (-15, -8, -4, -2, -1, 1, 2, 4, 8, 15)

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio_min",
    "heading_cos", "heading_sin_abs", "heading_valid",
    "speed", "arc_len", "frac", "dt_prev", "dt_next",
    "n_within_30", "n_within_60",
    "ctx_mean_near", "ctx_mean_far", "ctx_max_near", "ctx_rank_mean",
    "ctx_heading_cos",
    "out_deg_head", "in_deg_tail", "has_reverse",
    "fwd_progress",
]


class ArcIndex:
    """KD-tree over points sampled along every (undirected) road segment near the track."""

    def __init__(self, trip: Trip, margin: float = 400.0):
        A, B = trip.arc_a, trip.arc_b
        lo = trip.track_xy.min(0) - margin
        hi = trip.track_xy.max(0) + margin
        smin = np.minimum(A, B)
        smax = np.maximum(A, B)
        near = np.all(smax >= lo, axis=1) & np.all(smin <= hi, axis=1)
        arc_ids = np.nonzero(near)[0]
        # undirected de-duplication: both directions share geometry
        u = trip.arcs[arc_ids, 0]
        v = trip.arcs[arc_ids, 1]
        key = np.minimum(u, v).astype(np.int64) * 10_000_000 + np.maximum(u, v)
        _, first = np.unique(key, return_index=True)
        seg_arc = arc_ids[first]  # one representative arc per undirected segment
        self.seg_key = key[first]
        a, b = A[seg_arc], B[seg_arc]
        L = np.linalg.norm(b - a, axis=1)
        n = np.maximum(1, np.ceil(L / SAMPLE_STEP).astype(int)) + 1
        seg_of_sample = np.repeat(np.arange(len(seg_arc)), n)
        ts = np.concatenate([np.linspace(0, 1, k) for k in n])
        pts = a[seg_of_sample] + ts[:, None] * (b - a)[seg_of_sample]
        self.tree = cKDTree(pts)
        self.seg_of_sample = seg_of_sample
        # map undirected segment -> directed arcs (1 or 2)
        key_all = key
        order = np.argsort(key_all)
        ks = key_all[order]
        self.seg_dir_arcs = []
        starts = np.searchsorted(ks, self.seg_key, "left")
        ends = np.searchsorted(ks, self.seg_key, "right")
        for s, e in zip(starts, ends):
            self.seg_dir_arcs.append(arc_ids[order[s:e]])
        self.n_seg = len(seg_arc)

    def candidates(self, P: np.ndarray, k: int = K_CAND) -> np.ndarray:
        """Return (T, k) array of directed arc ids (-1 padded)."""
        kk = min(KNN_SAMPLES, len(self.seg_of_sample))
        _, idx = self.tree.query(P, k=kk)
        idx = np.atleast_2d(idx)
        out = np.full((len(P), k), -1, dtype=np.int64)
        for i in range(len(P)):
            segs = pd.unique(self.seg_of_sample[idx[i]])  # ordered by nearest sample
            arcs = []
            for s in segs:
                arcs.extend(self.seg_dir_arcs[s])
                if len(arcs) >= k:
                    break
            arcs = arcs[:k]
            out[i, : len(arcs)] = arcs
        return out


def gps_heading(xy: np.ndarray, times: np.ndarray, half: int = 2):
    """Heading (unit vector) and speed from a centred window; heading invalid when
    displacement is too small (vehicle stopped -> GPS jitter dominates)."""
    T = len(xy)
    i0 = np.clip(np.arange(T) - half, 0, T - 1)
    i1 = np.clip(np.arange(T) + half, 0, T - 1)
    disp = xy[i1] - xy[i0]
    dist = np.linalg.norm(disp, axis=1)
    dt = np.maximum(times[i1] - times[i0], 1e-6)
    speed = dist / dt
    valid = dist > 3.0
    unit = np.where(valid[:, None], disp / np.maximum(dist, 1e-9)[:, None], 0.0)
    return unit, speed, valid


def build_features(trip: Trip, index: ArcIndex | None = None, labels: np.ndarray | None = None):
    """Return DataFrame with one row per (observation, candidate)."""
    if index is None:
        index = ArcIndex(trip)
    P = trip.track_xy
    T = len(P)
    cand = index.candidates(P)
    valid_c = cand >= 0
    t_idx = np.repeat(np.arange(T), K_CAND)[valid_c.ravel()]
    arc = cand.ravel()[valid_c.ravel()]

    A_all, B_all = trip.arc_a, trip.arc_b
    A, B = A_all[arc], B_all[arc]
    d, frac, _ = point_segment(P[t_idx], A, B)
    seg = B - A
    L = np.linalg.norm(seg, axis=1)
    arc_unit = seg / np.maximum(L, 1e-9)[:, None]

    df = pd.DataFrame({"t": t_idx, "arc": arc, "dist": d, "frac": frac, "arc_len": L})
    g = df.groupby("t")["dist"]
    df["dist_rank"] = g.rank(method="first")
    dmin = g.transform("min")
    df["dist_minus_min"] = df["dist"] - dmin
    df["dist_ratio_min"] = df["dist"] / (dmin + 1.0)
    df["n_within_30"] = (df["dist"] < 30).groupby(df["t"]).transform("sum")
    df["n_within_60"] = (df["dist"] < 60).groupby(df["t"]).transform("sum")

    unit, speed, hvalid = gps_heading(P, trip.times)
    hc = (unit[t_idx] * arc_unit).sum(1)
    cross = unit[t_idx, 0] * arc_unit[:, 1] - unit[t_idx, 1] * arc_unit[:, 0]
    df["heading_cos"] = np.where(hvalid[t_idx], hc, 0.0)
    df["heading_sin_abs"] = np.where(hvalid[t_idx], np.abs(cross), 0.0)
    df["heading_valid"] = hvalid[t_idx].astype(float)
    df["speed"] = speed[t_idx]
    dtp = np.r_[1.0, np.diff(trip.times)]
    dtn = np.r_[np.diff(trip.times), 1.0]
    df["dt_prev"] = np.minimum(dtp[t_idx], 600)
    df["dt_next"] = np.minimum(dtn[t_idx], 600)

    # Temporal context: how close do NEIGHBOURING observations lie to this arc?
    # A wrong parallel road may be close to one noisy fix but not to the
    # whole window of fixes around it.
    ctx = []
    for off in WINDOW_OFFSETS:
        j = np.clip(t_idx + off, 0, T - 1)
        dj, _, _ = point_segment(P[j], A, B)
        ctx.append(dj)
    ctx = np.vstack(ctx)  # (W, n)
    near = np.isin(np.array(WINDOW_OFFSETS), [-4, -2, -1, 1, 2, 4])
    df["ctx_mean_near"] = ctx[near].mean(0)
    df["ctx_mean_far"] = ctx[~near].mean(0)
    df["ctx_max_near"] = ctx[near].max(0)
    df["ctx_rank_mean"] = (df["ctx_mean_near"] + df["dist"]).groupby(df["t"]).rank(method="first")
    # heading agreement with a longer window
    unit_l, _, hv_l = gps_heading(P, trip.times, half=6)
    df["ctx_heading_cos"] = np.where(hv_l[t_idx], (unit_l[t_idx] * arc_unit).sum(1), 0.0)

    # Topology
    n_nodes = len(trip.nodes_xy)
    out_deg = np.bincount(trip.arcs[:, 0], minlength=n_nodes)
    in_deg = np.bincount(trip.arcs[:, 1], minlength=n_nodes)
    df["out_deg_head"] = out_deg[trip.arcs[arc, 1]]
    df["in_deg_tail"] = in_deg[trip.arcs[arc, 0]]
    rev_key = set(zip(trip.arcs[:, 0].tolist(), trip.arcs[:, 1].tolist()))
    df["has_reverse"] = np.fromiter(
        ((int(v), int(u)) in rev_key for u, v in trip.arcs[arc]), dtype=float, count=len(arc)
    )
    # progress along arc between t-2 and t+2 (positive = moving with the arc direction)
    jm = np.clip(t_idx - 2, 0, T - 1)
    jp = np.clip(t_idx + 2, 0, T - 1)
    _, fm, _ = point_segment(P[jm], A, B)
    _, fp, _ = point_segment(P[jp], A, B)
    df["fwd_progress"] = (fp - fm) * L

    df["trip"] = trip.trip_id
    if labels is not None:
        df["y"] = (df["arc"].values == labels[df["t"].values]).astype(np.int8)
    return df, cand
