"""Bayesian hyperparameter optimisation of the selected model (LightGBM) with
Optuna's TPE sampler (Tree-structured Parzen Estimator).

Objective : maximise mean observation-level Top-1 accuracy over GroupKFold(5)
            on the 80 training trips (trip = group). Top-1 is the decision the
            route decoder consumes, so it is the primary selection metric.
Budget    : N_TRIALS trials; each trial fits on a sample of TUNE_POINTS whole
            observations per training fold and validates on the full held-out fold.

After the search:
  * untuned (default config from the comparison) and tuned configs are refitted
    on ALL training rows and scored on the student-test trips,
  * 5-fold out-of-fold (OOF) probabilities of the tuned config are saved for
    decoder tuning (so the HMM is tuned on honest, unseen-trip probabilities),
  * the tuned model is exported to artifacts/model.joblib.
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.metrics import candidate_metrics, load_split  # noqa: E402

SEED = 42
N_TRIALS = int(os.environ.get("N_TRIALS", 30))
TUNE_POINTS = 40_000
N_JOBS = 12
UNTUNED = dict(n_estimators=300, learning_rate=0.05, num_leaves=31)


def make(params):
    return LGBMClassifier(random_state=SEED, n_jobs=N_JOBS, verbose=-1, **params)


def sample_points(df, n, seed):
    keys = df[["trip", "t"]].drop_duplicates()
    if len(keys) > n:
        keys = keys.sample(n, random_state=seed)
    return df.merge(keys, on=["trip", "t"], how="inner")


def cv_score(params, train, folds):
    res = []
    for k, (tr_i, va_i) in enumerate(folds):
        tr = sample_points(train.iloc[tr_i], TUNE_POINTS, SEED + k)
        va = train.iloc[va_i]
        m = make(params).fit(tr[FEATURES].values, tr["y"].values)
        res.append(candidate_metrics(va, m.predict_proba(va[FEATURES].values)[:, 1]))
    return pd.DataFrame(res)


def main():
    out = os.path.join(ROOT, "results", "bayes_opt")
    os.makedirs(out, exist_ok=True)
    train, _ = load_split(ROOT, "train")
    test, _ = load_split(ROOT, "student_test")
    train = train.reset_index(drop=True)
    folds = list(GroupKFold(n_splits=5).split(train, groups=train["trip"].values))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        t0 = time.time()
        r = cv_score(params, train, folds)
        trial.set_user_attr("cv_macro_f1", float(r.macro_f1.mean()))
        trial.set_user_attr("cv_log_loss", float(r.log_loss.mean()))
        trial.set_user_attr("cv_top1_std", float(r.top1.std()))
        trial.set_user_attr("seconds", time.time() - t0)
        print(f"trial {trial.number}: top1={r.top1.mean():.4f} ({time.time()-t0:.0f}s) {params}", flush=True)
        return float(r.top1.mean())

    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="pathsnap_lgbm")
    t_search = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    t_search = time.time() - t_search
    study.trials_dataframe().to_csv(os.path.join(out, "optuna_trials.csv"), index=False)
    best = dict(study.best_params, subsample_freq=1)
    print("BEST", study.best_value, best, flush=True)

    # untuned vs tuned, identical protocol
    rows = []
    final_models = {}
    for tag, params in (("untuned", UNTUNED), ("tuned", best)):
        cv = cv_score(params, train, folds)
        t0 = time.time()
        m = make(params).fit(train[FEATURES].values, train["y"].values)
        fit_s = time.time() - t0
        t0 = time.perf_counter()
        p = m.predict_proba(test[FEATURES].values)[:, 1]
        inf = time.perf_counter() - t0
        r = candidate_metrics(test, p)
        rows.append({"model": f"LightGBM ({tag})", "cv_top1": cv.top1.mean(), "cv_top1_std": cv.top1.std(),
                     "cv_macro_f1": cv.macro_f1.mean(), "cv_log_loss": cv.log_loss.mean(),
                     "test_top1": r["top1"], "test_macro_f1": r["macro_f1"], "test_log_loss": r["log_loss"],
                     "test_precision": r["precision"], "test_recall": r["recall"],
                     "fit_full_s": fit_s, "infer_ms_per_1k_obs": 1e6 * inf / r["n_obs"]})
        final_models[tag] = m
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "tuned_vs_untuned.csv"), index=False)

    # OOF probabilities of the tuned config (full fold data) for decoder tuning
    oof = np.zeros(len(train))
    for k, (tr_i, va_i) in enumerate(folds):
        m = make(best).fit(train.iloc[tr_i][FEATURES].values, train.iloc[tr_i]["y"].values)
        oof[va_i] = m.predict_proba(train.iloc[va_i][FEATURES].values)[:, 1]
        joblib.dump(m, os.path.join(ROOT, "artifacts", f"oof_fold{k}.joblib"))
    fold_of_trip = {t: k for k, (_, va_i) in enumerate(folds) for t in train.iloc[va_i]["trip"].unique()}
    with open(os.path.join(ROOT, "artifacts", "oof_fold_of_trip.json"), "w") as f:
        json.dump(fold_of_trip, f, indent=2)

    final = final_models["tuned"]
    joblib.dump(final, os.path.join(ROOT, "artifacts", "model.joblib"))
    joblib.dump(final_models["untuned"], os.path.join(ROOT, "artifacts", "model_untuned.joblib"))
    imp = pd.DataFrame({"feature": FEATURES, "gain": final.booster_.feature_importance("gain")})
    imp.sort_values("gain", ascending=False).to_csv(os.path.join(out, "feature_importance.csv"), index=False)
    with open(os.path.join(out, "best_params.json"), "w") as f:
        json.dump({"best_cv_top1": study.best_value, "params": best, "n_trials": N_TRIALS,
                   "search_seconds": t_search, "tune_points_per_fold": TUNE_POINTS,
                   "sampler": "TPE(seed=42, n_startup_trials=8)"}, f, indent=2)


if __name__ == "__main__":
    main()
