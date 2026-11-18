"""Decoder tuning and classifier-only vs classifier+HMM evaluation.

1. Decoder grid search (beta, w_emit, k_hmm) on training trips, using
   OUT-OF-FOLD probabilities (each trip scored by the fold model that never saw
   it), so the decoder is tuned on realistic probabilities. Student-test is
   reported for the chosen config as a development check.
2. Final model + chosen decoder on the 10 student-test trips: per-trip table.
3. Every baseline model through the same decoder on student-test
   ("compatibility with the sequential decoder" + route continuity column).

Outputs: results/hmm/*.csv, artifacts/hmm_config.json
"""
import itertools
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.data import list_trips, load_trip  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.hmm import HMMConfig, RoadGraph  # noqa: E402
from pathsnap.pipeline import decode, evaluate_prediction, score_candidates  # noqa: E402

OUT = os.path.join(ROOT, "results", "hmm")
N_TUNE_TRIPS = 24
GRID = {"beta": [3.0, 6.0, 10.0, 20.0], "w_emit": [0.5, 1.0, 2.0], "k_hmm": [6]}
K_GRID = [3, 4, 6, 8]


def run(trip, df, cfg, graph):
    pred = decode(trip, df, cfg, graph)
    e = evaluate_prediction(trip, pred)
    e["hmm_s"] = pred["hmm_s"]
    e.update(pred["hmm_info"])
    return e


def main():
    os.makedirs(OUT, exist_ok=True)
    fold_of_trip = json.load(open(os.path.join(ROOT, "artifacts", "oof_fold_of_trip.json")))
    fold_models = [joblib.load(os.path.join(ROOT, "artifacts", f"oof_fold{k}.joblib")) for k in range(5)]
    rng = np.random.default_rng(42)
    train_dirs = list_trips(os.path.join(ROOT, "data", "train"))
    # skip the very long trips to keep the grid affordable
    sizes = {d: sum(1 for _ in open(os.path.join(d, os.path.basename(d) + ".track"))) for d in train_dirs}
    pool = [d for d in train_dirs if sizes[d] <= 6000]
    tune_dirs = sorted(rng.choice(pool, N_TUNE_TRIPS, replace=False))

    cache = []
    for d in tune_dirs:
        trip = load_trip(d)
        m = fold_models[fold_of_trip[trip.trip_id]]
        df, _ = score_candidates(trip, m)
        cache.append((trip, df, RoadGraph(trip.arcs, trip.nodes_xy)))
        print("scored", trip.trip_id, flush=True)

    rows = []
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    for c in combos:
        cfg = HMMConfig(**c)
        res = pd.DataFrame([run(t, df, cfg, g) for t, df, g in cache])
        rows.append({**c, "hmm_top1": res.hmm_top1.mean(), "clf_top1": res.classifier_top1.mean(),
                     "hmm_continuity": res.hmm_continuity.mean(), "hmm_arc_f1": res.hmm_arc_f1.mean(),
                     "hmm_length_error": res.hmm_length_error.mean(), "hmm_s": res.hmm_s.sum()})
        print(rows[-1], flush=True)
    grid = pd.DataFrame(rows)
    best = grid.sort_values("hmm_top1", ascending=False).iloc[0]
    base = {"beta": float(best.beta), "w_emit": float(best.w_emit)}
    for k in K_GRID:
        if k == 6:
            continue
        cfg = HMMConfig(k_hmm=k, **base)
        res = pd.DataFrame([run(t, df, cfg, g) for t, df, g in cache])
        rows.append({**base, "k_hmm": k, "hmm_top1": res.hmm_top1.mean(), "clf_top1": res.classifier_top1.mean(),
                     "hmm_continuity": res.hmm_continuity.mean(), "hmm_arc_f1": res.hmm_arc_f1.mean(),
                     "hmm_length_error": res.hmm_length_error.mean(), "hmm_s": res.hmm_s.sum()})
        print(rows[-1], flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(OUT, "decoder_grid_train_oof.csv"), index=False)
    best = grid.sort_values(["hmm_top1", "hmm_s"], ascending=[False, True]).iloc[0]
    cfg = HMMConfig(k_hmm=int(best.k_hmm), beta=float(best.beta), w_emit=float(best.w_emit))
    json.dump(cfg.__dict__, open(os.path.join(ROOT, "artifacts", "hmm_config.json"), "w"), indent=2)
    print("CHOSEN", cfg, flush=True)

    # final model on student test
    final = joblib.load(os.path.join(ROOT, "artifacts", "model.joblib"))
    per_trip = []
    test_dirs = list_trips(os.path.join(ROOT, "data", "student_test"))
    test_cache = []
    for d in test_dirs:
        trip = load_trip(d)
        t0 = time.perf_counter()
        df, tm = score_candidates(trip, final)
        g = RoadGraph(trip.arcs, trip.nodes_xy)
        e = run(trip, df, cfg, g)
        e.update(trip=trip.trip_id, total_s=time.perf_counter() - t0, **tm)
        per_trip.append(e)
        test_cache.append(trip)
        print("test", e["trip"], round(e["classifier_top1"], 4), round(e["hmm_top1"], 4), flush=True)
    pt = pd.DataFrame(per_trip)
    pt.to_csv(os.path.join(OUT, "student_test_per_trip.csv"), index=False)
    w = pt.n_reliable_obs
    summary = {c: float(np.average(pt[c], weights=w)) for c in ("classifier_top1", "hmm_top1")}
    summary.update({c: float(pt[c].mean()) for c in pt.columns if c.startswith(("classifier_", "hmm_"))
                    and c not in summary and c != "hmm_s"})
    json.dump(summary, open(os.path.join(OUT, "student_test_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))

    # every baseline model through the decoder (student test)
    rows = []
    bdir = os.path.join(ROOT, "artifacts", "baseline_models")
    for f in sorted(os.listdir(bdir)):
        if not f.endswith(".joblib"):
            continue
        m = joblib.load(os.path.join(bdir, f))
        res = []
        for trip in test_cache:
            df, _ = score_candidates(trip, m)
            res.append(run(trip, df, cfg, RoadGraph(trip.arcs, trip.nodes_xy)))
        r = pd.DataFrame(res)
        ww = r.n_reliable_obs
        rows.append({"model": f[:-7],
                     "clf_top1": np.average(r.classifier_top1, weights=ww),
                     "clf_continuity": r.classifier_continuity.mean(),
                     "hmm_top1": np.average(r.hmm_top1, weights=ww),
                     "hmm_continuity": r.hmm_continuity.mean(),
                     "clf_arc_f1": r.classifier_arc_f1.mean(), "hmm_arc_f1": r.hmm_arc_f1.mean()})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "all_models_with_decoder.csv"), index=False)


if __name__ == "__main__":
    main()
