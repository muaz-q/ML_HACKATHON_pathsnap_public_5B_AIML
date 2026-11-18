"""Exploratory data analysis on the train and student_test partitions.

Outputs in results/eda/:
  inventory.csv        per-trip counts, sizes, gaps, extent, connectivity, label quality
  eda_summary.json     partition-level summary
  *.png                plots
  map_trip_<id>.html   Folium map: GPS + road network + reference route
"""
import json
import os
import sys

import folium
import matplotlib
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"  # keyless basemap (CARTO now needs a key off-site)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.data import list_trips, load_trip  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402

OUT = os.path.join(ROOT, "results", "eda")


def raw_checks(trip_dir):
    """Checks on the raw files before any cleaning."""
    tid = os.path.basename(trip_dir)
    tr = np.loadtxt(os.path.join(trip_dir, f"{tid}.track"), ndmin=2)
    sizes = {ext: os.path.getsize(os.path.join(trip_dir, f"{tid}.{ext}")) for ext in
             ("track", "nodes", "arcs", "route")}
    return {
        "raw_rows": len(tr), "n_cols_track": tr.shape[1],
        "nan_rows": int((~np.isfinite(tr)).any(1).sum()),
        "bad_lon": int(((tr[:, 0] < -180) | (tr[:, 0] > 180)).sum()),
        "bad_lat": int(((tr[:, 1] < -90) | (tr[:, 1] > 90)).sum()),
        "non_increasing_t": int((np.diff(tr[:, 2]) <= 0).sum()),
        **{f"bytes_{k}": v for k, v in sizes.items()},
    }


def trip_stats(trip_dir, split):
    r = raw_checks(trip_dir)
    trip = load_trip(trip_dir)
    xy, t = trip.track_xy, trip.times
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    n = len(trip.nodes_xy)
    A = coo_matrix((np.ones(len(trip.arcs)), (trip.arcs[:, 0], trip.arcs[:, 1])), shape=(n, n))
    n_comp, comp = connected_components(A, directed=True, connection="weak")
    n_scc, _ = connected_components(A, directed=True, connection="strong")
    deg = np.bincount(trip.arcs.ravel(), minlength=n)
    labels, _, dlab = align_route(trip)
    arc_len = np.linalg.norm(trip.arc_b - trip.arc_a, axis=1)
    route = trip.route
    route_conn = (trip.arcs[route[:-1], 1] == trip.arcs[route[1:], 0]).mean()
    return {
        "trip": trip.trip_id, "split": split, **r,
        "n_obs": len(t), "duration_s": t[-1] - t[0],
        "dt_median": np.median(dt), "share_dt_1s": (dt == 1).mean(), "dt_max": dt.max(),
        "n_gaps_gt_10s": int((dt > 10).sum()), "n_gaps_gt_60s": int((dt > 60).sum()),
        "speed_median_mps": np.median(step / np.maximum(dt, 1e-6)),
        "track_length_km": step.sum() / 1000,
        "lon_c": trip.track_ll[:, 0].mean(), "lat_c": trip.track_ll[:, 1].mean(),
        "extent_x_km": np.ptp(xy[:, 0]) / 1000, "extent_y_km": np.ptp(xy[:, 1]) / 1000,
        "n_nodes": n, "n_arcs": len(trip.arcs),
        "isolated_nodes": int((deg == 0).sum()),
        "weak_components": n_comp, "strong_components": n_scc,
        "largest_wcc_share": np.bincount(comp).max() / n,
        "arc_len_median_m": np.median(arc_len),
        "route_len": len(route), "route_km": arc_len[route].sum() / 1000,
        "route_consecutive_connected": route_conn,
        "route_unique_arcs": len(np.unique(route)),
        "label_dist_median": np.median(dlab), "label_dist_p95": np.percentile(dlab, 95),
        "share_label_gt_60m": (dlab > 60).mean(),
    }


def folium_map(trip, path):
    labels, _, _ = align_route(trip)
    c = trip.track_ll.mean(0)
    m = folium.Map(location=[c[1], c[0]], zoom_start=14, tiles=None)
    folium.TileLayer(TILES, attr="Tiles &copy; Esri", name="Basemap (Esri light grey)").add_to(m)
    net = folium.FeatureGroup(name="Road network (near track)", show=True)
    lo, hi = trip.track_xy.min(0) - 300, trip.track_xy.max(0) + 300
    A, B = trip.arc_a, trip.arc_b
    near = np.nonzero(np.all(np.minimum(A, B) <= hi, 1) & np.all(np.maximum(A, B) >= lo, 1))[0]
    seen = set()
    for a in near:
        u, v = trip.arcs[a]
        k = (min(u, v), max(u, v))
        if k in seen:
            continue
        seen.add(k)
        folium.PolyLine([(trip.nodes_ll[u, 1], trip.nodes_ll[u, 0]),
                         (trip.nodes_ll[v, 1], trip.nodes_ll[v, 0])],
                        color="#9e9e9e", weight=1, opacity=0.6).add_to(net)
    net.add_to(m)
    ref = folium.FeatureGroup(name="Reference route (.route)")
    for a in trip.route:
        u, v = trip.arcs[a]
        folium.PolyLine([(trip.nodes_ll[u, 1], trip.nodes_ll[u, 0]),
                         (trip.nodes_ll[v, 1], trip.nodes_ll[v, 0])],
                        color="#2e7d32", weight=5, opacity=0.8).add_to(ref)
    ref.add_to(m)
    gps = folium.FeatureGroup(name="Raw GPS observations")
    for lon, lat in trip.track_ll[::2]:
        folium.CircleMarker((lat, lon), radius=2, color="#1565c0", fill=True, weight=0).add_to(gps)
    gps.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[trip.track_ll[:, 1].min(), trip.track_ll[:, 0].min()],
                  [trip.track_ll[:, 1].max(), trip.track_ll[:, 0].max()]])
    m.save(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for split in ("train", "student_test"):
        for d in list_trips(os.path.join(ROOT, "data", split)):
            rows.append(trip_stats(d, split))
            print(split, rows[-1]["trip"], flush=True)
    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(OUT, "inventory.csv"), index=False)

    summ = {}
    for split, g in inv.groupby("split"):
        summ[split] = {
            "trips": len(g), "observations": int(g.n_obs.sum()),
            "obs_per_trip_min_median_max": [int(g.n_obs.min()), int(g.n_obs.median()), int(g.n_obs.max())],
            "nan_rows": int(g.nan_rows.sum()), "invalid_coords": int(g.bad_lon.sum() + g.bad_lat.sum()),
            "non_increasing_timestamps": int(g.non_increasing_t.sum()),
            "share_dt_1s_mean": float(g.share_dt_1s.mean()),
            "trips_with_gap_gt_60s": int((g.n_gaps_gt_60s > 0).sum()),
            "nodes_total": int(g.n_nodes.sum()), "arcs_total": int(g.n_arcs.sum()),
            "isolated_nodes_total": int(g.isolated_nodes.sum()),
            "trips_with_multiple_weak_components": int((g.weak_components > 1).sum()),
            "route_consecutive_connected_mean": float(g.route_consecutive_connected.mean()),
            "label_dist_median_mean": float(g.label_dist_median.mean()),
            "obs_with_label_gt_60m": float((g.share_label_gt_60m * g.n_obs).sum()),
        }
    with open(os.path.join(OUT, "eda_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps(summ, indent=2))

    tr = inv[inv.split == "train"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].hist(tr.n_obs, bins=30, color="#1565c0"); ax[0, 0].set_title("GPS observations per trip (train)")
    ax[0, 0].set_xlabel("observations"); ax[0, 0].set_ylabel("trips")
    ax[0, 1].hist(np.log10(tr.n_arcs), bins=30, color="#6a1b9a")
    ax[0, 1].set_title("Road-network size per trip (train)"); ax[0, 1].set_xlabel("log10(#arcs)")
    ax[1, 0].hist(tr.share_dt_1s, bins=30, color="#2e7d32")
    ax[1, 0].set_title("Share of 1-second sampling steps (train)"); ax[1, 0].set_xlabel("share")
    ax[1, 1].hist(tr.label_dist_median, bins=30, color="#ef6c00")
    ax[1, 1].set_title("Median GPS-to-reference-arc distance (train)"); ax[1, 1].set_xlabel("metres")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "eda_histograms.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for split, col in (("train", "#1565c0"), ("student_test", "#d32f2f")):
        g = inv[inv.split == split]
        ax.scatter(g.lon_c, g.lat_c, s=25, c=col, label=split)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_xlim(-180, 180); ax.set_ylim(-60, 75)
    ax.set_title("Trip locations (GPS centroid)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "trip_locations.png"), dpi=130); plt.close(fig)

    # pooled sampling-interval and label distance distributions (train)
    dts, dl = [], []
    for d in list_trips(os.path.join(ROOT, "data", "train")):
        trip = load_trip(d)
        dts.append(np.diff(trip.times))
        dl.append(align_route(trip)[2])
    dts, dl = np.concatenate(dts), np.concatenate(dl)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    vals, cnt = np.unique(np.minimum(dts, 11), return_counts=True)
    ax[0].bar([str(int(v)) if v < 11 else ">10" for v in vals], cnt, color="#2e7d32")
    ax[0].set_yscale("log"); ax[0].set_title("Sampling interval between fixes (train)"); ax[0].set_xlabel("seconds")
    ax[1].hist(np.minimum(dl, 100), bins=100, color="#ef6c00")
    ax[1].set_yscale("log"); ax[1].axvline(60, color="k", ls="--", label="60 m reliability cut")
    ax[1].set_title("GPS-to-reference-arc distance (train, clipped 100 m)"); ax[1].set_xlabel("metres"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "dt_and_label_distance.png"), dpi=130); plt.close(fig)

    # one trajectory + network plot (matplotlib) and Folium map
    trip = load_trip(os.path.join(ROOT, "data", "train", "00000005"))
    fig, ax = plt.subplots(figsize=(7, 7))
    for a in range(len(trip.arcs)):
        p, q = trip.arc_a[a], trip.arc_b[a]
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#bdbdbd", lw=.6)
    for a in trip.route:
        p, q = trip.arc_a[a], trip.arc_b[a]
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#2e7d32", lw=2.5)
    ax.scatter(trip.track_xy[:, 0], trip.track_xy[:, 1], s=2, c="#1565c0")
    ax.set_aspect("equal"); ax.set_title("Train trip 00000005: network (grey), reference (green), GPS (blue)")
    ax.set_xlabel("x (m, local AEQD)"); ax.set_ylabel("y (m)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "trip_00000005_network.png"), dpi=130); plt.close(fig)
    folium_map(trip, os.path.join(OUT, "map_trip_00000005.html"))


if __name__ == "__main__":
    main()
