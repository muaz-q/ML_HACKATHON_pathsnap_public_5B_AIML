"""Step-by-step dry run of the PathSnap pipeline on ONE trip, printing what each
stage produces. Made for live demos / viva.

  python scripts/dry_run.py                         # phase 1 (data prep), student-test trip 00000065
  python scripts/dry_run.py --trip 00000009         # another trip
  python scripts/dry_run.py --full                  # + classifier + HMM + metrics (phase 2)
  python scripts/dry_run.py --fix 250               # inspect a different GPS fix
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.data import load_trip  # noqa: E402
from pathsnap.features import FEATURES, K_CAND, ArcIndex, build_features  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def step(n, title):
    print("\n" + "=" * 78 + f"\nSTEP {n}: {title}\n" + "=" * 78)


def pause(args):
    if args.pause:
        input("   [press Enter for the next step] ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trip", default="00000065")
    ap.add_argument("--split", default="student_test")
    ap.add_argument("--fix", type=int, default=120, help="GPS fix index to inspect")
    ap.add_argument("--full", action="store_true", help="also run classifier + HMM (phase 2)")
    ap.add_argument("--pause", action="store_true", help="wait for Enter between steps")
    args = ap.parse_args()
    d = os.path.join(ROOT, "data", args.split, args.trip)
    tid = args.trip

    step(1, "Raw files (headerless, whitespace-separated)")
    for ext, cols in (("track", "lon lat time"), ("nodes", "lon lat"), ("arcs", "from_node to_node"),
                      ("route", "arc_id")):
        p = os.path.join(d, f"{tid}.{ext}")
        with open(p) as f:
            head = [next(f).strip() for _ in range(3)]
        n = sum(1 for _ in open(p))
        print(f"  {tid}.{ext:<6} {n:>7} rows   columns: {cols:<18} first rows: {head}")
    pause(args)

    step(2, "Load + clean + project to metres  (pathsnap/data.py)")
    t0 = time.perf_counter()
    trip = load_trip(d)
    print(f"  loaded in {time.perf_counter() - t0:.2f}s: {len(trip.track_xy)} fixes, {len(trip.nodes_xy)} nodes, "
          f"{len(trip.arcs)} arcs, route of {len(trip.route)} arcs")
    dt = np.diff(trip.times)
    print(f"  time steps: {np.mean(dt == 1):.1%} are exactly 1 s, largest gap {dt.max():.0f} s")
    i = 0
    print(f"  fix 0 lon/lat = ({trip.track_ll[i, 0]:.6f}, {trip.track_ll[i, 1]:.6f})  ->  "
          f"x/y = ({trip.track_xy[i, 0]:.1f} m, {trip.track_xy[i, 1]:.1f} m)  [AEQD centred on trip]")
    deg = np.linalg.norm(trip.track_ll[1] - trip.track_ll[0])
    met = np.linalg.norm(trip.track_xy[1] - trip.track_xy[0])
    print(f"  fix0->fix1: {deg:.6f} 'degrees' (meaningless)  vs  {met:.1f} m after projection")
    pause(args)

    step(3, "Route -> one label per GPS fix  (pathsnap/labels.py, monotonic DP)")
    t0 = time.perf_counter()
    labels, pos, dlab = align_route(trip)
    print(f"  aligned {len(labels)} fixes to {len(trip.route)} route arcs in {time.perf_counter() - t0:.2f}s")
    print(f"  route position never decreases: {bool(np.all(np.diff(pos) >= 0))}")
    print(f"  fix -> route position for fixes 0..14: {pos[:15].tolist()}")
    print(f"  distance fix->its label arc: median {np.median(dlab):.1f} m, 95th pct {np.percentile(dlab, 95):.1f} m")
    print(f"  unreliable fixes (> 60 m from label, excluded): {(dlab > 60).sum()}")
    pause(args)

    step(4, f"Candidates: {K_CAND} nearest directed arcs per fix  (pathsnap/features.py: ArcIndex)")
    t0 = time.perf_counter()
    idx = ArcIndex(trip)
    cand = idx.candidates(trip.track_xy)
    hit = (cand == labels[:, None]).any(1)
    print(f"  KD-tree over {len(idx.seg_of_sample):,} points sampled along {idx.n_seg:,} road segments "
          f"near the track; candidates for all fixes in {time.perf_counter() - t0:.2f}s")
    print(f"  reference arc is among the candidates for {hit.mean():.2%} of fixes (miss rate {1 - hit.mean():.2%})")
    f = min(args.fix, len(labels) - 1)
    print(f"  fix {f}: candidates {cand[f].tolist()}   reference arc = {labels[f]}")
    pause(args)

    step(5, f"Features: one row per (fix, candidate), {len(FEATURES)} features + label y")
    t0 = time.perf_counter()
    df, _ = build_features(trip, idx, labels)
    print(f"  built {len(df):,} rows x {len(FEATURES)} features in {time.perf_counter() - t0:.2f}s; "
          f"positives (y=1): {df.y.mean():.1%}")
    show = ["arc", "dist", "dist_rank", "heading_cos", "ctx_mean_near", "fwd_progress", "frac", "y"]
    rows = df[df.t == f][show].round(2)
    print(f"\n  the {len(rows)} candidate rows for fix {f}:")
    print(rows.to_string(index=False))
    print("\n  read it: the y=1 row should be close (dist), pointing the same way (heading_cos ~ +1),"
          "\n  with neighbouring fixes also close (ctx_mean_near) and moving forward along it (fwd_progress > 0).")
    pause(args)

    if not args.full:
        print("\n(phase 1 dry run done; add --full for classifier + HMM)")
        return

    import joblib
    import json
    from pathsnap.hmm import HMMConfig, RoadGraph, viterbi, stitch_route
    from pathsnap.metrics import route_continuity, sequence_metrics

    step(6, "Classifier: P(candidate is the reference)  (artifacts/model.joblib, tuned LightGBM)")
    model = joblib.load(os.path.join(ROOT, "artifacts", "model.joblib"))
    t0 = time.perf_counter()
    df["p"] = model.predict_proba(df[FEATURES].values)[:, 1]
    print(f"  scored {len(df):,} rows in {time.perf_counter() - t0:.2f}s")
    r = df[df.t == f][["arc", "dist", "heading_cos", "p", "y"]].sort_values("p", ascending=False).round(3)
    print(f"\n  fix {f}, candidates ranked by probability:")
    print(r.to_string(index=False))
    pause(args)

    step(7, "HMM + Viterbi: choose a connected sequence  (pathsnap/hmm.py)")
    cfg = HMMConfig(**json.load(open(os.path.join(ROOT, "artifacts", "hmm_config.json"))))
    print(f"  config: {cfg}")
    df = df.sort_values(["t", "p"], ascending=[True, False])
    top = df.groupby("t").head(cfg.k_hmm)
    cands, probs, offs = [], [], []
    clf = np.empty(len(labels), dtype=np.int64)
    for t, g in top.groupby("t"):
        cands.append(g.arc.values)
        probs.append(g.p.values / g.p.values.sum())
        offs.append(g.frac.values * g.arc_len.values)
        clf[t] = g.arc.values[0]
    graph = RoadGraph(trip.arcs, trip.nodes_xy)
    t0 = time.perf_counter()
    hmm, info = viterbi(cands, probs, offs, trip.track_xy, trip.times, graph, cfg)
    route = stitch_route(hmm, graph)
    print(f"  Viterbi over {len(labels)} fixes x {cfg.k_hmm} states in {time.perf_counter() - t0:.2f}s; {info}")
    diff = np.nonzero(clf != hmm)[0]
    fixed = np.nonzero((clf != labels) & (hmm == labels))[0]
    broke = np.nonzero((clf == labels) & (hmm != labels))[0]
    print(f"  the HMM changed the classifier's choice at {len(diff)} fixes: "
          f"{len(fixed)} corrected, {len(broke)} made worse")
    show_at = fixed if len(fixed) else diff
    if len(show_at):
        s = max(0, show_at[0] - 3)
        w = pd.DataFrame({"fix": range(s, s + 10), "classifier": clf[s:s + 10], "hmm": hmm[s:s + 10],
                          "reference": labels[s:s + 10]})
        w["clf_ok"] = w.classifier == w.reference
        w["hmm_ok"] = w.hmm == w.reference
        print(f"\n  around the first fix the HMM corrected (fix {show_at[0]}):")
        print(w.to_string(index=False))
    pause(args)

    step(8, "Scores (reference used ONLY here, after prediction)")
    rel = dlab <= 60
    for name, arcs, rt in (("classifier only", clf, stitch_route(clf, graph)), ("classifier + HMM", hmm, route)):
        sm = sequence_metrics(rt, trip.route, graph.length)
        print(f"  {name:<17} Top-1 {np.mean(arcs[rel] == labels[rel]):.2%} | continuity "
              f"{route_continuity(arcs, graph, trip.track_xy):.2%} | route arc F1 {sm['arc_f1']:.3f} | "
              f"length error {sm['length_error']:.3f}")


if __name__ == "__main__":
    main()
