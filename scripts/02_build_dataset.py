"""Build the candidate-level training tables for the train and student_test partitions.

Outputs (in data/processed/):
  <split>_candidates.pkl   one row per (GPS observation, candidate arc) + label y
  <split>_points.csv       one row per GPS observation: reference arc, label distance,
                           whether the reference arc is in the candidate set
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.data import list_trips, load_trip  # noqa: E402
from pathsnap.features import ArcIndex, build_features  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402


def build(split, limit=None):
    out = os.path.join(ROOT, "data", "processed")
    os.makedirs(out, exist_ok=True)
    frames, pts = [], []
    trips = list_trips(os.path.join(ROOT, "data", split))[:limit]
    for k, d in enumerate(trips):
        t0 = time.time()
        trip = load_trip(d)
        labels, pos, dlab = align_route(trip)
        idx = ArcIndex(trip)
        df, cand = build_features(trip, idx, labels)
        hit = (cand == labels[:, None]).any(1)
        frames.append(df)
        pts.append(pd.DataFrame({
            "trip": trip.trip_id, "t": np.arange(len(labels)), "ref_arc": labels,
            "route_pos": pos, "label_dist": dlab, "in_candidates": hit,
            "time": trip.times,
        }))
        print(f"[{split} {k+1}/{len(trips)}] {trip.trip_id} T={len(labels)} rows={len(df)} "
              f"miss={1-hit.mean():.3f} med_label_dist={np.median(dlab):.1f}m "
              f"({time.time()-t0:.1f}s)", flush=True)
    if limit is None:
        pd.concat(frames, ignore_index=True).to_pickle(os.path.join(out, f"{split}_candidates.pkl"))
        pd.concat(pts, ignore_index=True).to_csv(os.path.join(out, f"{split}_points.csv"), index=False)


if __name__ == "__main__":
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    for split in sys.argv[1].split(","):
        build(split, limit)
