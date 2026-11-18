"""Score the final system (tuned LightGBM + HMM, frozen) on a held-out folder of trips,
e.g. the evaluator_test set released on review day. Nothing is retrained or retuned.

  python scripts/08_eval_holdout.py data/evaluator_test
Writes results/<folder name>/per_trip.csv and summary.json.
"""
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
from pathsnap.hmm import HMMConfig  # noqa: E402
from pathsnap.pipeline import evaluate_prediction, predict_trip  # noqa: E402


def main(folder):
    name = os.path.basename(os.path.normpath(folder))
    out_dir = os.path.join(ROOT, "results", name)
    os.makedirs(out_dir, exist_ok=True)
    model = joblib.load(os.path.join(ROOT, "artifacts", "model.joblib"))
    cfg = HMMConfig(**json.load(open(os.path.join(ROOT, "artifacts", "hmm_config.json"))))
    rows = []
    for path in list_trips(folder):
        tid = os.path.basename(path)
        t0 = time.perf_counter()
        trip = load_trip(path, with_route=False)            # predict without the reference
        pred = predict_trip(trip, model, cfg)
        secs = time.perf_counter() - t0
        trip.route = np.loadtxt(os.path.join(path, f"{tid}.route"), dtype=np.int64, ndmin=1)
        m = evaluate_prediction(trip, pred)                  # reference used only here
        m.update(trip=tid, n_arcs=len(trip.arcs), seconds=round(secs, 2))
        rows.append(m)
        print(f"{tid}  fixes {m['n_obs']:>5}  arcs {m['n_arcs']:>6}  clf {m['classifier_top1']:.4f}  "
              f"hmm {m['hmm_top1']:.4f}  arcF1 {m['classifier_arc_f1']:.3f}->{m['hmm_arc_f1']:.3f}  "
              f"lenErr {m['hmm_length_error']:.3f}  {secs:.1f}s", flush=True)
    df = pd.DataFrame(rows).set_index("trip")
    df.to_csv(os.path.join(out_dir, "per_trip.csv"))
    w = df.n_reliable_obs
    summ = {"n_trips": len(df), "n_fixes": int(df.n_obs.sum()), "n_reliable_fixes": int(w.sum())}
    for k in ("classifier", "hmm"):
        summ[f"{k}_top1_pooled"] = float((df[f"{k}_top1"] * w).sum() / w.sum())
        for m in ("top1", "continuity", "arc_precision", "arc_recall", "arc_f1", "length_error"):
            summ[f"{k}_{m}_mean_over_trips"] = float(df[f"{k}_{m}"].mean())
    summ["seconds_mean"] = float(df.seconds.mean())
    summ["seconds_max"] = float(df.seconds.max())
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "evaluator_test"))
