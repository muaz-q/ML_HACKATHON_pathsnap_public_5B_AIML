"""Report figures for tuning/HMM + Folium maps of predictions (model-comparison plots: 03b).

Reads results/*.csv produced by scripts 03-05; writes results/figures/*.png and
results/maps/*.html. Nothing here is hand-entered: every bar/line is a saved result.
"""
import json
import os
import sys

import folium
import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"  # keyless basemap (CARTO now needs a key off-site)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.data import load_trip  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.hmm import HMMConfig  # noqa: E402
from pathsnap.labels import align_route  # noqa: E402
from pathsnap.metrics import argmax_flags, load_split  # noqa: E402
from pathsnap.pipeline import predict_trip  # noqa: E402

FIG = os.path.join(ROOT, "results", "figures")
MAPS = os.path.join(ROOT, "results", "maps")
C1, C2, C3 = "#1565c0", "#d32f2f", "#2e7d32"


def bar(ax, names, vals, err=None, color=C1, fmt="{:.3f}"):
    y = np.arange(len(names))
    ax.barh(y, vals, xerr=err, color=color, alpha=.85, capsize=3)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8)


def bayes_plots():
    d = os.path.join(ROOT, "results", "bayes_opt")
    tr = pd.read_csv(os.path.join(d, "optuna_trials.csv"))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tr.number, tr.value, "o", c=C1, label="trial CV Top-1")
    ax.plot(tr.number, tr.value.cummax(), "-", c=C2, label="best so far")
    ax.axvline(7.5, color="grey", ls=":", label="end of random start-up trials")
    ax.set_xlabel("trial"); ax.set_ylabel("mean CV Top-1 (5 folds)"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("Bayesian optimisation (Optuna TPE) history — LightGBM")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "optuna_history.png"), dpi=140); plt.close(fig)

    params = [c for c in tr.columns if c.startswith("params_")]
    fig, axs = plt.subplots(2, 4, figsize=(14, 6))
    for ax, p in zip(axs.ravel(), params):
        ax.scatter(tr[p], tr.value, c=tr.number, cmap="viridis", s=18)
        ax.set_title(p.replace("params_", ""), fontsize=9)
        if p.replace("params_", "") in ("learning_rate", "num_leaves", "min_child_samples", "reg_lambda", "reg_alpha"):
            ax.set_xscale("log")
    fig.suptitle("CV Top-1 vs each hyperparameter (colour = trial number)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "optuna_params.png"), dpi=140); plt.close(fig)

    tu = pd.read_csv(os.path.join(d, "tuned_vs_untuned.csv"))
    metrics = ["cv_top1", "cv_macro_f1", "test_top1", "test_macro_f1"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics))
    for i, (_, r) in enumerate(tu.iterrows()):
        ax.bar(x + (i - .5) * .38, [r[m] for m in metrics], .38, label=r.model, color=[C1, C2][i])
        for j, m in enumerate(metrics):
            ax.text(x[j] + (i - .5) * .38, r[m], f"{r[m]:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, metrics); ax.set_ylim(.85, 1); ax.legend(); ax.set_title("Untuned vs tuned LightGBM")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "tuned_vs_untuned.png"), dpi=140); plt.close(fig)

    imp = pd.read_csv(os.path.join(d, "feature_importance.csv")).head(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    bar(ax, imp.feature, imp.gain / imp.gain.sum(), color=C3, fmt="{:.2f}")
    ax.set_xlabel("share of total split gain"); ax.set_title("Tuned LightGBM feature importance (top 15)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "feature_importance.png"), dpi=140); plt.close(fig)


def confusion():
    test, _ = load_split(ROOT, "student_test")
    m = joblib.load(os.path.join(ROOT, "artifacts", "model.joblib"))
    test = test.reset_index(drop=True)
    test["p"] = m.predict_proba(test[FEATURES].values)[:, 1]
    pred = argmax_flags(test)
    cm = confusion_matrix(test.y, pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black" if i != j or i == 1 else "white")
    ax.set_xticks([0, 1], ["pred: not ref", "pred: ref (top-1)"]); ax.set_yticks([0, 1], ["true: not ref", "true: ref"])
    ax.set_title("Tuned LightGBM, student-test\ncandidate-level confusion matrix")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "confusion_tuned_lgbm.png"), dpi=140); plt.close(fig)


def hmm_plots():
    pt = pd.read_csv(os.path.join(ROOT, "results", "hmm", "student_test_per_trip.csv"), dtype={"trip": str})
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(pt))
    for k, (col, lab) in enumerate((("top1", "Top-1 accuracy"), ("arc_f1", "Route arc F1"))):
        ax[k].bar(x - .2, pt[f"classifier_{col}"], .4, label="classifier only", color="#ef6c00")
        ax[k].bar(x + .2, pt[f"hmm_{col}"], .4, label="classifier + HMM", color=C2)
        ax[k].set_xticks(x, pt.trip, rotation=45); ax[k].set_title(f"Student-test {lab} per trip")
        ax[k].set_ylim(min(.5, pt[[f"classifier_{col}", f"hmm_{col}"]].values.min() - .02), 1); ax[k].legend(loc="upper center", bbox_to_anchor=(.5, -.28), ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "hmm_vs_classifier.png"), dpi=140); plt.close(fig)

    g = pd.read_csv(os.path.join(ROOT, "results", "hmm", "decoder_grid_train_oof.csv"))
    g6 = g[g.k_hmm == 6].pivot(index="w_emit", columns="beta", values="hmm_top1")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(g6.values, cmap="viridis")
    ax.set_xticks(range(len(g6.columns)), g6.columns); ax.set_yticks(range(len(g6.index)), g6.index)
    ax.set_xlabel("beta (m)"); ax.set_ylabel("emission weight w_e")
    for i in range(g6.shape[0]):
        for j in range(g6.shape[1]):
            ax.text(j, i, f"{g6.values[i, j]:.4f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_title("Decoder grid (train OOF): HMM Top-1, K=6"); fig.colorbar(im)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "decoder_grid.png"), dpi=140); plt.close(fig)


def prediction_map(trip_dir, out_html):
    trip = load_trip(trip_dir)
    model = joblib.load(os.path.join(ROOT, "artifacts", "model.joblib"))
    cfg = HMMConfig(**json.load(open(os.path.join(ROOT, "artifacts", "hmm_config.json"))))
    pred = predict_trip(trip, model, cfg)
    labels, _, _ = align_route(trip)
    ll = lambda n: (trip.nodes_ll[n, 1], trip.nodes_ll[n, 0])
    c = trip.track_ll.mean(0)
    m = folium.Map(location=[c[1], c[0]], zoom_start=14, tiles=None)
    folium.TileLayer(TILES, attr="Tiles &copy; Esri", name="Basemap (Esri light grey)").add_to(m)
    lo, hi = trip.track_xy.min(0) - 250, trip.track_xy.max(0) + 250
    A, B = trip.arc_a, trip.arc_b
    near = np.nonzero(np.all(np.minimum(A, B) <= hi, 1) & np.all(np.maximum(A, B) >= lo, 1))[0]
    net = folium.FeatureGroup(name="Road network")
    for a in near:
        u, v = trip.arcs[a]
        folium.PolyLine([ll(u), ll(v)], color="#9e9e9e", weight=1, opacity=.5).add_to(net)
    layers = [(net, True)]
    for name, arcs, col, w, show in (("Reference route", trip.route, C3, 7, True),
                                     ("Classifier-only (argmax per fix)", np.unique(pred["classifier_arcs"]), "#ef6c00", 3, False),
                                     ("Predicted route (HMM + Viterbi)", pred["route"], C2, 3, True)):
        fg = folium.FeatureGroup(name=name, show=show)
        for a in arcs:
            u, v = trip.arcs[a]
            folium.PolyLine([ll(u), ll(v)], color=col, weight=w, opacity=.85).add_to(fg)
        layers.append((fg, show))
    gps = folium.FeatureGroup(name="Raw GPS observations")
    for i, (lon, lat) in enumerate(trip.track_ll):
        if i % 2 == 0:
            folium.CircleMarker((lat, lon), radius=2, color=C1, fill=True, weight=0, fill_opacity=.9).add_to(gps)
    wrong = folium.FeatureGroup(name="Fixes where HMM arc != reference arc", show=False)
    for i in np.nonzero(pred["hmm_arcs"] != labels)[0]:
        lon, lat = trip.track_ll[i]
        folium.CircleMarker((lat, lon), radius=4, color="black", fill=False, weight=1).add_to(wrong)
    for fg, _ in layers:
        fg.add_to(m)
    gps.add_to(m); wrong.add_to(m)
    legend = ('<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:#fff;padding:8px 10px;'
              'border-radius:6px;font:12px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.3)">'
              f'<b>Trip {trip.trip_id}</b><br>'
              '<span style="color:#9e9e9e">&#9473;</span> road network<br>'
              '<span style="color:#1565c0">&#9679;</span> raw GPS<br>'
              '<span style="color:#2e7d32">&#9473;&#9473;</span> reference route<br>'
              '<span style="color:#ef6c00">&#9473;</span> classifier only<br>'
              '<span style="color:#d32f2f">&#9473;</span> HMM + Viterbi</div>')
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[trip.track_ll[:, 1].min(), trip.track_ll[:, 0].min()],
                  [trip.track_ll[:, 1].max(), trip.track_ll[:, 0].max()]])
    m.save(out_html)


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(MAPS, exist_ok=True)
    bayes_plots()
    confusion()
    hmm_plots()
    for tid in sys.argv[1:] or ["00000009", "00000065"]:
        prediction_map(os.path.join(ROOT, "data", "student_test", tid), os.path.join(MAPS, f"prediction_{tid}.html"))
        print("map", tid)


if __name__ == "__main__":
    main()
