"""Evaluation metrics.

Candidate-level (binary) metrics are computed on rows (observation, candidate).
The *decision* for an observation is its top-scoring candidate, so the predicted
binary label is 1 for the argmax candidate of each observation and 0 otherwise.
  - Top-1 accuracy : share of reliable observations whose argmax candidate is the
                     reference arc (a candidate miss counts as wrong).
  - Macro F1       : mean of F1(class 0) and F1(class 1) over candidate rows,
                     using the argmax decision above.
  - Log loss       : binary cross-entropy of the raw probabilities.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

RELIABLE_LABEL_DIST = 60.0  # metres; farther "labels" come from track/route mismatch


def load_split(root, split):
    import os
    c = pd.read_pickle(os.path.join(root, "data", "processed", f"{split}_candidates.pkl"))
    p = pd.read_csv(os.path.join(root, "data", "processed", f"{split}_points.csv"),
                    dtype={"trip": str})
    p["trip"] = p["trip"].str.zfill(8)
    rel = p.loc[p.label_dist <= RELIABLE_LABEL_DIST, ["trip", "t"]]
    c = c.merge(rel, on=["trip", "t"], how="inner")
    return c, p


def argmax_flags(df: pd.DataFrame, score_col="p") -> np.ndarray:
    key = df["trip"].astype(str) + ":" + df["t"].astype(str)
    idx = df.groupby(key, sort=False)[score_col].idxmax()
    flag = np.zeros(len(df), dtype=np.int8)
    flag[df.index.get_indexer(idx.values)] = 1
    return flag


def candidate_metrics(df: pd.DataFrame, p: np.ndarray) -> dict:
    df = df[["trip", "t", "y"]].copy()
    df["p"] = p
    df = df.reset_index(drop=True)
    pred = argmax_flags(df)
    n_obs = df.groupby(["trip", "t"]).ngroups
    top1 = (pred & df["y"].values).sum() / n_obs
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "top1": float(top1),
        "macro_f1": float(f1_score(df["y"], pred, average="macro")),
        "precision": float(precision_score(df["y"], pred, zero_division=0)),
        "recall": float(recall_score(df["y"], pred, zero_division=0)),
        "log_loss": float(log_loss(df["y"], pc, labels=[0, 1])),
        "n_obs": int(n_obs),
    }


def route_continuity(arcs_seq, graph, gps_xy) -> float:
    """Share of consecutive observation pairs whose predicted arcs form a feasible
    move: same arc, or the next arc is reachable on the directed road graph within
    max(300 m, 3 x GPS step + 150 m). 1.0 = a drivable route with no teleports."""
    ok = 0
    n = len(arcs_seq) - 1
    for t in range(n):
        a, b = int(arcs_seq[t]), int(arcs_seq[t + 1])
        if a == b:
            ok += 1
            continue
        gc = float(np.linalg.norm(gps_xy[t + 1] - gps_xy[t]))
        dist, _ = graph.dijkstra(int(graph.arcs[a, 1]), max(300.0, 3.0 * gc + 150.0))
        ok += int(graph.arcs[b, 0]) in dist
    return ok / n if n > 0 else 1.0


def sequence_metrics(pred_route: list, ref_route: np.ndarray, arc_len: np.ndarray) -> dict:
    """Route-level comparison of the reconstructed arc sequence with the reference."""
    P, R = set(map(int, pred_route)), set(map(int, ref_route))
    inter = P & R
    prec = len(inter) / max(len(P), 1)
    rec = len(inter) / max(len(R), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    # length-weighted (Newson & Krumm style): (d_missed + d_added) / d_reference
    Lr = arc_len[list(R)].sum()
    missed = arc_len[list(R - P)].sum()
    added = arc_len[list(P - R)].sum()
    return {"arc_precision": prec, "arc_recall": rec, "arc_f1": f1,
            "length_error": float((missed + added) / max(Lr, 1e-9))}
