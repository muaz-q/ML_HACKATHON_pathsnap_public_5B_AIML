"""Train and compare candidate classifiers.

Protocol
  * 80 training trips -> GroupKFold(5) with trip as the group (no trip is ever
    split between fit and validation folds).
  * For fairness and to keep slow learners (KNN, MLP, forests) tractable, every
    model is fitted on the same sub-sample of WHOLE observations from the
    training folds (CAP_POINTS observations => ~CAP_POINTS*12 rows). Validation
    always uses every row of the held-out fold.
  * After CV each model is refitted on a CAP_POINTS sample of all 80 trips and
    scored on the 10 student-test trips.

Outputs: results/model_cv_folds.csv, results/model_comparison.csv,
         results/mlp_loss_curve.csv, artifacts/baseline_models/*.joblib
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.metrics import candidate_metrics, load_split  # noqa: E402

SEED = 42
CAP_POINTS = 30_000
N_JOBS = 12
VAL_CAP_MODELS = {"KNN"}   # validated on a 15k-observation sample of each held-out fold
VAL_CAP_POINTS = 15_000


def models():
    return {
        "LogisticRegression": ("Linear", make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)), "C=1.0, L2, standardised"),
        "GaussianNB": ("Probabilistic", GaussianNB(var_smoothing=1e-9), "var_smoothing=1e-9"),
        "KNN": ("Instance-based", make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=50, weights="distance", algorithm="brute", n_jobs=N_JOBS)),
            "k=50, distance-weighted, brute force, standardised"),
        "DecisionTree": ("Tree", DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=SEED),
                         "max_depth=12, min_leaf=20"),
        "RandomForest": ("Ensemble (bagging)", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, n_jobs=N_JOBS, random_state=SEED),
            "200 trees, min_leaf=5"),
        "ExtraTrees": ("Ensemble (bagging)", ExtraTreesClassifier(
            n_estimators=200, min_samples_leaf=5, n_jobs=N_JOBS, random_state=SEED),
            "200 trees, min_leaf=5"),
        "HistGradientBoosting": ("Boosting", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, random_state=SEED), "300 iters, lr=0.1"),
        "LightGBM": ("Boosting", LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED,
            n_jobs=N_JOBS, verbose=-1), "300 trees, lr=0.05, 31 leaves"),
        "XGBoost": ("Boosting", XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6, tree_method="hist",
            random_state=SEED, n_jobs=N_JOBS, eval_metric="logloss"), "300 trees, lr=0.1, depth=6"),
        "MLP": ("Neural network", make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3, max_iter=60,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=8, random_state=SEED)),
            "64-32 ReLU, Adam lr=1e-3, early stopping"),
    }


def sample_points(df, n_points, seed):
    keys = df[["trip", "t"]].drop_duplicates()
    if len(keys) > n_points:
        keys = keys.sample(n_points, random_state=seed)
    return df.merge(keys, on=["trip", "t"], how="inner")


def timed_predict(model, X):
    t0 = time.perf_counter()
    p = model.predict_proba(X)[:, 1]
    return p, time.perf_counter() - t0


def main():
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "artifacts", "baseline_models"), exist_ok=True)
    train, _ = load_split(ROOT, "train")
    test, _ = load_split(ROOT, "student_test")
    print("train rows", len(train), "test rows", len(test), flush=True)
    groups = train["trip"].values
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(train, groups=groups))
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None

    fold_rows, summary = [], []
    # non-learned reference baseline: nearest arc
    base = candidate_metrics(test, 1.0 / test["dist_rank"].values)  # rank 1 scores highest
    print("nearest-arc baseline (student test):", base, flush=True)

    for name, (family, est, cfg) in models().items():
        if only and name not in only:
            continue
        cv = []
        for k, (tr_i, va_i) in enumerate(folds):
            tr = sample_points(train.iloc[tr_i], CAP_POINTS, SEED + k)
            va = train.iloc[va_i]
            if name in VAL_CAP_MODELS:  # KNN: O(train x val) prediction cost
                va = sample_points(va, VAL_CAP_POINTS, SEED + 100 + k)
            m = clone(est)
            t0 = time.perf_counter()
            m.fit(tr[FEATURES].values, tr["y"].values)
            fit_t = time.perf_counter() - t0
            p, _ = timed_predict(m, va[FEATURES].values)
            r = candidate_metrics(va, p)
            r.update(model=name, fold=k, fit_s=fit_t)
            fold_rows.append(r)
            cv.append(r)
            print(f"  {name} fold {k}: top1={r['top1']:.4f} f1={r['macro_f1']:.4f} "
                  f"ll={r['log_loss']:.4f} fit={fit_t:.1f}s", flush=True)
        # refit on all training trips (same cap) and score student test
        tr = sample_points(train, CAP_POINTS, SEED)
        m = clone(est)
        t0 = time.perf_counter()
        m.fit(tr[FEATURES].values, tr["y"].values)
        fit_t = time.perf_counter() - t0
        p, inf_t = timed_predict(m, test[FEATURES].values)
        r = candidate_metrics(test, p)
        joblib.dump(m, os.path.join(ROOT, "artifacts", "baseline_models", f"{name}.joblib"))
        test_pred = test[["trip", "t", "arc", "y"]].copy()
        test_pred["p"] = p
        test_pred.to_pickle(os.path.join(ROOT, "artifacts", "baseline_models", f"{name}_test_pred.pkl"))
        if name == "MLP":
            mlp = m[-1]
            pd.DataFrame({"epoch": np.arange(1, len(mlp.loss_curve_) + 1),
                          "train_loss": mlp.loss_curve_,
                          "val_accuracy": mlp.validation_scores_}).to_csv(
                os.path.join(ROOT, "results", "mlp_loss_curve.csv"), index=False)
        cvdf = pd.DataFrame(cv)
        summary.append({
            "model": name, "family": family, "config": cfg,
            "cv_top1_mean": cvdf.top1.mean(), "cv_top1_std": cvdf.top1.std(),
            "cv_macro_f1_mean": cvdf.macro_f1.mean(), "cv_macro_f1_std": cvdf.macro_f1.std(),
            "cv_log_loss_mean": cvdf.log_loss.mean(),
            "test_top1": r["top1"], "test_macro_f1": r["macro_f1"],
            "test_precision": r["precision"], "test_recall": r["recall"],
            "test_log_loss": r["log_loss"], "fit_s": fit_t,
            "infer_ms_per_1k_obs": 1e6 * inf_t / r["n_obs"],
        })
        print(f"== {name}: CV top1={cvdf.top1.mean():.4f}±{cvdf.top1.std():.4f} "
              f"TEST top1={r['top1']:.4f} f1={r['macro_f1']:.4f} ll={r['log_loss']:.4f}", flush=True)

    suffix = "" if not only else "_partial"
    pd.DataFrame(fold_rows).to_csv(os.path.join(ROOT, "results", f"model_cv_folds{suffix}.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(ROOT, "results", f"model_comparison{suffix}.csv"), index=False)
    with open(os.path.join(ROOT, "results", "nearest_arc_baseline.json"), "w") as f:
        json.dump(base, f, indent=2)


if __name__ == "__main__":
    main()
