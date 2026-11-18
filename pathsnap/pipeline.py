"""End-to-end inference for one trip: features -> classifier -> HMM/Viterbi.

predict_trip() never sees the reference route. evaluate_prediction() is the
only place the reference is used, and the API calls it server-side.
"""
from __future__ import annotations

import time

import numpy as np

from .features import FEATURES, ArcIndex, build_features
from .hmm import HMMConfig, RoadGraph, stitch_route, viterbi
from .labels import align_route
from .metrics import RELIABLE_LABEL_DIST, route_continuity, sequence_metrics


def score_candidates(trip, model):
    """Features + classifier probability for every (observation, candidate)."""
    timings = {}
    t0 = time.perf_counter()
    df, _ = build_features(trip, ArcIndex(trip), labels=None)
    timings["features_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    df["p"] = model.predict_proba(df[FEATURES].values)[:, 1]
    timings["classifier_s"] = time.perf_counter() - t0
    return df, timings


def decode(trip, df, cfg: HMMConfig, graph: RoadGraph | None = None):
    """Classifier-only argmax and HMM/Viterbi decoding from scored candidates."""
    t0 = time.perf_counter()
    df = df.sort_values(["t", "p"], ascending=[True, False])
    T = len(trip.track_xy)
    top = df.groupby("t", sort=True).head(cfg.k_hmm)
    cands, probs, offs = [], [], []
    clf_arcs = np.empty(T, dtype=np.int64)
    for t, g in top.groupby("t", sort=True):
        a = g["arc"].values
        p = g["p"].values
        cands.append(a)
        probs.append(p / max(p.sum(), 1e-12))
        offs.append(g["frac"].values * g["arc_len"].values)
        clf_arcs[t] = a[0]
    graph = graph or RoadGraph(trip.arcs, trip.nodes_xy)
    hmm_arcs, info = viterbi(cands, probs, offs, trip.track_xy, trip.times, graph, cfg)
    route = stitch_route(hmm_arcs, graph)
    return {"classifier_arcs": clf_arcs, "hmm_arcs": hmm_arcs, "route": route,
            "hmm_info": info, "graph": graph, "hmm_s": time.perf_counter() - t0}


def predict_trip(trip, model, cfg: HMMConfig):
    df, timings = score_candidates(trip, model)
    pred = decode(trip, df, cfg)
    timings["hmm_s"] = pred.pop("hmm_s")
    pred["timings"] = timings
    return pred


def evaluate_prediction(trip, pred) -> dict:
    """Server-side scoring against the reference route (never sent to the UI)."""
    labels, _, dlab = align_route(trip)
    rel = dlab <= RELIABLE_LABEL_DIST
    g = pred["graph"]
    out = {"n_obs": int(len(labels)), "n_reliable_obs": int(rel.sum())}
    for key in ("classifier", "hmm"):
        arcs = pred[f"{key}_arcs"]
        out[f"{key}_top1"] = float((arcs[rel] == labels[rel]).mean())
        out[f"{key}_continuity"] = route_continuity(arcs, g, trip.track_xy)
    clf_route = stitch_route(pred["classifier_arcs"], g)
    for key, r in (("classifier", clf_route), ("hmm", pred["route"])):
        for m, v in sequence_metrics(r, trip.route, g.length).items():
            out[f"{key}_{m}"] = float(v)
    return out
