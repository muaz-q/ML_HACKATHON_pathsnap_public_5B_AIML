"""Summarise the prepared (processed) dataset and save small samples that fit in git.

The full candidate tables (data/processed/*.pkl, ~500 MB) are rebuilt by
02_build_dataset.py; this script records what they contain.

Outputs (results/data_prep/):
  data_prep_summary.json         rows, fixes, positives, reliability, candidate miss rate per split
  per_trip_prep.csv              per-trip fixes, reliable share, miss rate, rows
  train_candidates_sample.csv    every candidate row of 40 consecutive fixes from 3 training trips
  feature_stats_train.csv        describe() of every feature, split by label y
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.features import FEATURES  # noqa: E402

RELIABLE = 60.0


def main():
    out = os.path.join(ROOT, "results", "data_prep")
    os.makedirs(out, exist_ok=True)
    proc = os.path.join(ROOT, "data", "processed")
    summary, per_trip = {}, []
    for split in ("train", "student_test"):
        c = pd.read_pickle(os.path.join(proc, f"{split}_candidates.pkl"))
        p = pd.read_csv(os.path.join(proc, f"{split}_points.csv"), dtype={"trip": str})
        p["trip"] = p["trip"].str.zfill(8)
        p["reliable"] = p.label_dist <= RELIABLE
        rel = p[p.reliable]
        cr = c.merge(rel[["trip", "t"]], on=["trip", "t"])
        summary[split] = {
            "trips": int(p.trip.nunique()),
            "gps_fixes": int(len(p)),
            "reliable_fixes": int(len(rel)),
            "unreliable_fixes_gt_60m": int((~p.reliable).sum()),
            "candidate_rows_all": int(len(c)),
            "candidate_rows_reliable": int(len(cr)),
            "candidates_per_fix_mean": round(len(c) / len(p), 2),
            "positive_rows": int(cr.y.sum()),
            "positive_share": round(float(cr.y.mean()), 4),
            "candidate_miss_rate_reliable": round(float(1 - rel.in_candidates.mean()), 5),
            "label_dist_median_m": round(float(p.label_dist.median()), 2),
            "n_features": len(FEATURES),
            "features": FEATURES,
        }
        g = p.groupby("trip")
        rows = c.groupby("trip").size()
        per_trip.append(pd.DataFrame({
            "split": split, "fixes": g.size(), "reliable_share": g.reliable.mean().round(4),
            "candidate_miss_rate": (1 - g.in_candidates.mean()).round(4),
            "label_dist_median_m": g.label_dist.median().round(2), "candidate_rows": rows}))
        if split == "train":
            sample = []
            for tid in sorted(cr.trip.unique())[:3]:
                s = cr[(cr.trip == tid) & (cr.t >= 100) & (cr.t < 140)]
                sample.append(s)
            pd.concat(sample)[["trip", "t", "arc", "y"] + FEATURES].round(4).to_csv(
                os.path.join(out, "train_candidates_sample.csv"), index=False)
            cr.groupby("y")[FEATURES].describe().T.round(4).to_csv(os.path.join(out, "feature_stats_train.csv"))
        print(split, json.dumps({k: v for k, v in summary[split].items() if k != "features"}))
    pd.concat(per_trip).reset_index().to_csv(os.path.join(out, "per_trip_prep.csv"), index=False)
    json.dump(summary, open(os.path.join(out, "data_prep_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
