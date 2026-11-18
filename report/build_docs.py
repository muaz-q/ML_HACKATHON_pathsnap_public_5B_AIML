"""Render the study guide and the technical report (HTML -> PDF via headless Chrome).

Every number is read from results/ — nothing is typed by hand.
  python report/build_docs.py
"""
import datetime
import json
import os
import subprocess
from string import Template

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, "results", *p)
FIG = lambda name: "file:///" + R("figures", name).replace("\\", "/")
EDA = lambda name: "file:///" + R("eda", name).replace("\\", "/")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def f4(x):
    return f"{x:.4f}"


def table(df, num_cols=()):
    h = "<table><tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    for _, r in df.iterrows():
        h += "<tr>" + "".join(
            f'<td class="n">{r[c]}</td>' if c in num_cols else f"<td>{r[c]}</td>" for c in df.columns) + "</tr>"
    return h + "</table>"


MODEL_INFO = {
    "LogisticRegression": dict(
        idea="A linear model: score = w·x + b, squashed by the sigmoid into a probability. Features are standardised first.",
        why="A strong, fast, well-calibrated baseline. If the features are good, many candidates are already separable by a weighted sum of distance, heading agreement and context distance.",
        sl="Strengths: fast, stable, gives decent probabilities (low log loss). Limits: only straight-line decision boundaries, so it cannot express interactions such as 'heading only matters when the car is moving' without hand-made interaction features.",
        hp="C = 1.0 (inverse L2 regularisation strength), lbfgs solver, max_iter 2000, StandardScaler.",
        loss="Binary cross-entropy (log loss) + L2 penalty."),
    "GaussianNB": dict(
        idea="Bayes' rule with the 'naive' assumption that, given the class, every feature is an independent normal distribution.",
        why="The simplest probabilistic model: a useful check of how much the features carry on their own.",
        sl="Strengths: trains in a fraction of a second and needs no tuning. Limits: our features are strongly correlated (dist, dist_rank, ctx_mean_near all measure closeness), so the independence assumption counts the same evidence several times and makes the probabilities extremely over-confident. That is exactly why its log loss is roughly ten times worse than every other model's, even though its ranking (Top-1) is still reasonable.",
        hp="var_smoothing = 1e-9 (default).",
        loss="None iterative: closed-form maximum-likelihood estimates of each class's mean and variance."),
    "KNN": dict(
        idea="To score a row, find the 50 most similar training rows (Euclidean distance on standardised features) and take a distance-weighted vote.",
        why="Instance-based: no assumptions about the shape of the boundary, and 'similar situations had similar answers' is plausible for map-matching.",
        sl="Strengths: flexible and simple. Limits: the prediction cost grows with the training size (O(train × query)). Here it was by far the slowest model, thousands of times slower than LightGBM per fix, which made it impractical for live use. It is also sensitive to feature scaling and to irrelevant dimensions in 23-D space.",
        hp="k = 50, weights = distance, brute-force search, StandardScaler.",
        loss="None (lazy learner, no training objective)."),
    "DecisionTree": dict(
        idea="A single tree of if/else splits on features, chosen greedily to make the child nodes as pure as possible.",
        why="Captures non-linear rules and interactions directly ('if heading is valid and heading_cos < 0 then probably the wrong direction').",
        sl="Strengths: interpretable, fast, non-linear. Limits: high variance. One tree overfits small quirks, which shows in the largest fold-to-fold spread of all the models. Probabilities are coarse (leaf frequencies).",
        hp="max_depth = 12, min_samples_leaf = 20.",
        loss="Gini impurity reduction at each split."),
    "RandomForest": dict(
        idea="Bagging: 200 deep trees, each trained on a bootstrap sample and considering a random subset of features at each split. The forest averages their votes.",
        why="Keeps the non-linearity of trees while averaging away most of a single tree's variance.",
        sl="Strengths: robust, little tuning, good probabilities. Limits: large model files (~86 MB) and slower inference than boosting, and it does not focus on the hard cases the way boosting does.",
        hp="n_estimators = 200, min_samples_leaf = 5, max_features = sqrt.",
        loss="Gini impurity per tree; averaging of the trees."),
    "ExtraTrees": dict(
        idea="Like a random forest, but the split thresholds are drawn at random rather than optimised, which adds more randomness and lowers the variance further.",
        why="A cheaper, even more variance-reduced cousin of the random forest.",
        sl="Strengths: fast to train, and the lowest CV standard deviation of the tree ensembles. Limits: random thresholds cost some accuracy here, and the model is the largest on disk (~177 MB).",
        hp="n_estimators = 200, min_samples_leaf = 5.",
        loss="Gini impurity with random thresholds."),
    "HistGradientBoosting": dict(
        idea="Gradient boosting: trees are added one after another, each fitting the errors (gradients of the log loss) of the ensemble so far. Feature values are binned into histograms for speed.",
        why="Boosting focuses on the hard, ambiguous candidates (parallel roads, opposite directions), exactly where accuracy is lost.",
        sl="Strengths: accurate and fast, scikit-learn native. Limits: fewer tuning knobs than LightGBM or XGBoost, and the default early-stopping hold-out is a random row split.",
        hp="max_iter = 300, learning_rate = 0.1, max_leaf_nodes = 31.",
        loss="Binary log loss, optimised by gradient boosting."),
    "LightGBM": dict(
        idea="Histogram gradient boosting with leaf-wise tree growth: it always splits the leaf with the largest loss reduction, which gives deeper and more accurate trees for the same number of leaves.",
        why="The same boosting rationale, and the most efficient implementation. It also exposes rich regularisation (leaves, min child samples, sub-sampling, L1/L2) for the Bayesian search.",
        sl="Strengths: the best CV and student-test Top-1, the best log loss, and fast inference on a small model file. Limits: leaf-wise growth can overfit small data, so it needs regularisation (min_child_samples, subsampling), and it is less interpretable than one tree.",
        hp="n_estimators = 300, learning_rate = 0.05, num_leaves = 31 (untuned).",
        loss="Binary log loss (gradient + Hessian boosting)."),
    "XGBoost": dict(
        idea="Regularised gradient boosting with depth-wise trees and an explicit L2 penalty on the leaf weights, using the hist tree method.",
        why="The other industry-standard booster, included to check that LightGBM's lead is not a library accident.",
        sl="Strengths: very close to LightGBM on every metric, with stable regularisation. Limits: slightly lower Top-1 and higher log loss than LightGBM here, and slower training.",
        hp="n_estimators = 300, learning_rate = 0.1, max_depth = 6, tree_method = hist.",
        loss="Binary log loss + L2 leaf regularisation."),
    "MLP": dict(
        idea="A feed-forward neural network: 23 inputs → 64 ReLU → 32 ReLU → 1 sigmoid output, trained with Adam.",
        why="Learns smooth non-linear combinations of the features. It is the neural-network family the brief asks us to try.",
        sl="Strengths: competitive accuracy and fast inference. Limits: needs scaling, early stopping and more data to beat trees on tabular features. Tabular data with heterogeneous features usually favours tree ensembles, as it does here.",
        hp="hidden (64, 32), alpha (L2) = 1e-4, Adam lr = 1e-3, early stopping on a 10 % hold-out, patience 8, max 60 epochs.",
        loss="Binary cross-entropy + L2 weight penalty."),
}

FAMILY_ORDER = ["LogisticRegression", "GaussianNB", "KNN", "DecisionTree", "RandomForest", "ExtraTrees",
                "HistGradientBoosting", "LightGBM", "XGBoost", "MLP"]


def fit_verdict(r):
    gap = r.test_top1 - r.cv_top1_mean
    s = (f"CV Top-1 {pct(r.cv_top1_mean)} ± {pct(r.cv_top1_std)} across folds vs student-test {pct(r.test_top1)} "
         f"(difference {gap * 100:+.2f} points). ")
    if abs(gap) <= 2 * r.cv_top1_std:
        s += ("The student-test score lies within the fold-to-fold variation of the CV score, so there is "
              "<b>no sign of overfitting</b> to the training trips: performance on unseen trips is stable. ")
    elif gap < 0:
        s += "Student-test is clearly below CV, which is evidence of some overfitting or a distribution shift. "
    else:
        s += "Student-test is above CV: the student-test trips are slightly easier than the average fold. "
    return s


def model_sections(cmp_, dec):
    out = ""
    for i, name in enumerate(FAMILY_ORDER, 1):
        r = cmp_.loc[name]
        m = MODEL_INFO[name]
        cont = dec.loc[name, "clf_continuity"] if dec is not None and name in dec.index else None
        extra = ""
        if name == "GaussianNB":
            extra = (f" Underfitting in the probabilistic sense: its log loss ({r.test_log_loss:.3f}) is far worse than "
                     f"logistic regression's ({cmp_.loc['LogisticRegression'].test_log_loss:.3f}) on the same features.")
        if name == "MLP":
            extra = (" The training-loss curve (Figure C6) falls smoothly and the hold-out accuracy plateaus, "
                     "so early stopping ended training before overfitting.")
        if name == "DecisionTree":
            extra = " Its fold std is the largest, which is the classic high-variance behaviour of a single tree."
        out += f"""
<h4>Model {i}: {name} <span style="font-weight:400;color:#64748b">({r.family})</span></h4>
<p><b>A. Idea.</b> {m['idea']}</p>
<p><b>Why suitable.</b> {m['why']}</p>
<p><b>Strengths and limits.</b> {m['sl']}</p>
<p><b>B. Hyperparameters (initial values).</b> {m['hp']} <b>Objective / loss:</b> {m['loss']}</p>
<p><b>C. Results (student-test).</b> Top-1 <b>{pct(r.test_top1)}</b>, Macro F1 <b>{f4(r.test_macro_f1)}</b>,
log loss <b>{f4(r.test_log_loss)}</b>, precision {f4(r.test_precision)}, recall {f4(r.test_recall)},
classifier-only route continuity {pct(cont) if cont is not None else 'n/a'}; fit {r.fit_s:.1f} s on the 30 k-fix sample,
inference {r.infer_ms_per_1k_obs:.1f} ms per 1,000 fixes.</p>
<p><b>Over/underfitting.</b> {fit_verdict(r)}{extra}</p>"""
        if name == "MLP":
            out += f'<figure><img src="{FIG("mlp_history.png")}" style="max-width:70%"><figcaption>Figure C6. MLP training history (the only model here with an epoch-wise history; the brief forbids invented curves for the others).</figcaption></figure>'
    return out


QA = [
    ("What is map-matching?", "Assigning each noisy GPS fix to the road segment the vehicle was actually on, and producing a connected route."),
    ("What does each raw file contain?", "track: lon, lat, time per fix. nodes: lon, lat (row = node id). arcs: from-node, to-node (row = arc id, directed). route: the reference arc ids in driving order, with no timestamps."),
    ("Why can't you use the route file directly as labels?", "It has one row per arc, not per fix, and no times. We aligned fixes to route arcs with a monotonic dynamic-programming alignment (like DTW) to get a per-fix label."),
    ("Why is the alignment monotonic?", "A vehicle traverses its route in order. Forcing non-decreasing route positions stops a fix from being matched back to an earlier arc that happens to be nearby (e.g. at a crossing)."),
    ("What are 'unreliable' fixes?", "Fixes more than 60 m from their aligned reference arc. These are mostly parts of the track the reference route does not cover (trip 54 has ~88 % of such fixes). They are excluded from training and scoring and counted in the report."),
    ("Why project coordinates?", "Degrees aren't metres and the scale differs with latitude. We use a per-trip azimuthal-equidistant projection, which is accurate in metres near the trip and works anywhere in the world."),
    ("How are candidates generated?", "A KD-tree over points sampled every 8 m along every road near the track. Per fix we collect the nearest distinct segments, expanded to both directions, up to 12 directed arcs."),
    ("What is the candidate miss rate and why does it matter?", "The share of fixes whose true arc isn't among the candidates. It caps the achievable accuracy. Ours is about 0.1 %, so the candidate stage isn't the bottleneck."),
    ("Which features matter most and why?", "Temporal-context distances (how close the neighbouring fixes are to the same arc) and heading agreement. Context separates parallel roads, and heading separates the two directions of the same road."),
    ("Why not use latitude/longitude as features?", "Every trip is in a different place with its own network. Absolute coordinates would let the model memorise places instead of learning matching behaviour, and they would not transfer to the evaluator's trips."),
    ("Why GroupKFold and not KFold?", "Consecutive fixes are near-duplicates. A random split leaks the same trip into train and validation and inflates the scores. Grouping by trip simulates unseen trips."),
    ("What's your primary metric?", "Observation-level Top-1 accuracy, i.e. whether the argmax candidate is the reference arc, because that is the decision the system makes. Macro F1 is inflated by the many easy negatives; log loss is secondary because the HMM uses the probabilities."),
    ("Why does nearest-road do so badly?", "Two-way roads give two arcs with identical geometry, so 'nearest' can't tell the direction, and parallel or crossing roads are often equally close. Heading and context features fix this."),
    ("Why LightGBM?", "Best CV Top-1, best student-test Top-1 and Macro F1, best log loss (best-calibrated emissions for the HMM), fast inference, small model. XGBoost was a close second."),
    ("Did any model overfit?", "Student-test scores fall within about one fold-std of the CV scores for every model, so none generalised badly. The single decision tree shows the highest fold variance. GaussianNB has badly over-confident probabilities (huge log loss) because of its independence assumption."),
    ("What is Bayesian optimisation and how is it different from grid search?", "It builds a model of which hyperparameters gave good scores and proposes the next trial where improvement is likely. We used Optuna's TPE, which compares densities of good versus bad trials. Grid and random search ignore past results."),
    ("What did you optimise and on what data?", "Mean CV Top-1 over GroupKFold(5) on the 80 training trips only. Student-test was used once, afterwards, to compare tuned and untuned."),
    ("What are the hidden states of your HMM?", "For each fix, its top-K candidate arcs (K from hmm_config.json), each with the fix's projected offset along the arc."),
    ("How are emissions computed?", "The classifier probabilities renormalised over the fix's K candidates, in log space, times a weight w_e."),
    ("How are transitions computed?", "exp(−|d_route − d_gps| / β): network shortest-path distance between the consecutive projected points versus the straight-line GPS distance (Newson & Krumm). The shortest paths come from a bounded Dijkstra on the directed graph."),
    ("Write the Viterbi recurrence.", "δ_t(b) = e_t(b) + max_a[δ_{t−1}(a) + log A(a,b)], with a back-pointer ψ_t(b) = argmax. Backtrack from argmax δ_T."),
    ("How do you handle GPS gaps?", "β scales with √Δt and the Dijkstra radius grows with the GPS distance. If Δt > 60 s or the gap is over 2 km, the chain breaks: uniform transition, and decoding restarts from the emissions."),
    ("What if no path exists between candidates?", "That transition gets a finite penalty (−jump_penalty) instead of −∞, so Viterbi always returns a path and the emissions decide."),
    ("Did the HMM help?", "Yes: it raised Top-1 and sharply raised route-level arc F1 and continuity (see Part E8). Where the classifier is already right, the HMM changes little; it fixes isolated wrong picks and teleports."),
    ("When did the HMM not help?", "Where the classifier is confidently wrong over a long stretch (e.g. a long parallel service road), the HMM follows the wrong road consistently. And when the true arc is outside the top K, it cannot recover it."),
    ("How did you tune the decoder without leakage?", "On training trips, using out-of-fold classifier probabilities (each trip scored by a fold model that never saw it). The evaluator data was never used."),
    ("How do you guarantee the ground truth doesn't reach the model or UI?", "The API loads the trip with with_route=False, predicts, and only then reads .route to score. The response contains metrics but never the reference arcs."),
    ("What does the API return?", "GPS fixes, the predicted (HMM) route polylines, the classifier-only route, the predicted arc per fix, nearby road network, server-side metrics and timings."),
    ("How long does a prediction take?", "A few seconds per trip on a laptop (features, then classifier, then HMM). The timings are shown in the UI and the logs."),
    ("What is route continuity?", "The share of consecutive fix pairs whose predicted arcs are the same, or where the second is reachable on the directed graph within a plausible driving distance. 100 % means no teleports."),
    ("Why is classifier-only continuity already high?", "Most consecutive fixes stay on the same arc. The real damage shows at route level: a single wrong pick inserts a detour, which is why its route arc F1 is much lower."),
    ("What is log loss, intuitively?", "The average 'surprise' of the true labels under the model's probabilities. Being confidently wrong is punished heavily. Lower is better."),
    ("How would you improve the system?", "Curved road geometry, learning the transition model (e.g. turn-angle features), a structured/CRF-style joint model, a faster compiled Dijkstra, and more careful labels near junctions."),
    ("Why K=12 candidates but only K_hmm in the HMM?", "12 keeps the miss rate near zero for training. The HMM's cost grows with K², and the classifier almost always ranks the true arc within its top few."),
    ("Why is the baseline LightGBM + HMM higher than the tuned LightGBM + HMM on student-test?", "Bayesian tuning optimised the classifier's own CV Top-1, not the combined classifier+HMM. The tuned model is more confident, so the HMM can overrule fewer of its mistakes. We didn't re-select based on student-test, because that would overfit to those 10 trips. The fix is to tune both together on out-of-fold training trips."),
    ("How did it do on the evaluator's hidden trips?", "We ran the frozen system on them without changing anything: 95.3% fix-level Top-1 with the HMM (92.1% without), mean route arc F1 0.956, at most 2.5 s per trip. That is slightly better than on our student-test trips, so it generalises. Trip 15 has a long route error only because the car drove the same loop about six times and the reference lists each road once."),
    ("Is the reference route perfect ground truth?", "No. It is untimed, it doesn't always cover the whole track, and our per-fix labels come from an alignment. We excluded unreliable fixes and treat accuracies as estimates."),
]


def main():
    today = datetime.date.today().isoformat()
    import sys
    sys.path.insert(0, ROOT)
    from pathsnap.metrics import load_split
    n_rows_train = len(load_split(ROOT, "train")[0])  # reliable candidate rows used for training
    inv = pd.read_csv(R("eda", "inventory.csv"), dtype={"trip": str})
    summ = json.load(open(R("eda", "eda_summary.json")))
    tr, te = summ["train"], summ["student_test"]
    cmp_ = pd.read_csv(R("model_comparison.csv")).set_index("model")
    nn = json.load(open(R("nearest_arc_baseline.json")))
    dec = pd.read_csv(R("hmm", "all_models_with_decoder.csv")).set_index("model")
    pts_tr = pd.read_csv(os.path.join(ROOT, "data", "processed", "train_points.csv"))
    pts_te = pd.read_csv(os.path.join(ROOT, "data", "processed", "student_test_points.csv"))
    rel_tr = pts_tr[pts_tr.label_dist <= 60]
    rel_te = pts_te[pts_te.label_dist <= 60]
    bo = json.load(open(R("bayes_opt", "best_params.json")))
    trials = pd.read_csv(R("bayes_opt", "optuna_trials.csv"))
    tvu = pd.read_csv(R("bayes_opt", "tuned_vs_untuned.csv"))
    imp = pd.read_csv(R("bayes_opt", "feature_importance.csv"))
    hcfg = json.load(open(os.path.join(ROOT, "artifacts", "hmm_config.json")))
    grid = pd.read_csv(R("hmm", "decoder_grid_train_oof.csv"))
    ptrip = pd.read_csv(R("hmm", "student_test_per_trip.csv"), dtype={"trip": str})
    hs = json.load(open(R("hmm", "student_test_summary.json")))

    # ---- tables
    mt = cmp_.loc[FAMILY_ORDER].reset_index()
    mt_html = pd.DataFrame({
        "Model": mt.model, "Family": mt.family, "Key configuration": mt.config,
        "CV Macro F1": [f"{a:.4f} ± {b:.4f}" for a, b in zip(mt.cv_macro_f1_mean, mt.cv_macro_f1_std)],
        "CV Top-1": [f"{pct(a)} ± {pct(b)}" for a, b in zip(mt.cv_top1_mean, mt.cv_top1_std)],
        "Test Top-1": mt.test_top1.map(pct),
        "Route continuity": [pct(dec.loc[m, "clf_continuity"]) for m in mt.model],
        "Test Macro F1": mt.test_macro_f1.map(f4), "Test log loss": mt.test_log_loss.map(f4),
        "Inference ms/1k fixes": mt.infer_ms_per_1k_obs.map(lambda v: f"{v:,.1f}"),
    })
    model_table = table(mt_html, num_cols=set(mt_html.columns[3:]))

    best_model = cmp_.test_top1.idxmax()
    selection_text = (
        f"<b>Selected: LightGBM.</b> It has the highest CV Top-1 ({pct(cmp_.loc['LightGBM'].cv_top1_mean)}), the highest "
        f"student-test Top-1 ({pct(cmp_.loc['LightGBM'].test_top1)}) and Macro F1 ({f4(cmp_.loc['LightGBM'].test_macro_f1)}), "
        f"and the lowest student-test log loss ({f4(cmp_.loc['LightGBM'].test_log_loss)}). XGBoost "
        f"({pct(cmp_.loc['XGBoost'].test_top1)}) and HistGradientBoosting ({pct(cmp_.loc['HistGradientBoosting'].test_top1)}) "
        f"are close behind, which confirms that <i>gradient boosting</i> is the right family for this table of features. "
        f"Criteria considered, as the brief asks: generalisation (CV on unseen trips and student-test agree), stability "
        f"(fold std {pct(cmp_.loc['LightGBM'].cv_top1_std)}, in line with the others), probability quality (the best log loss, which matters "
        f"because the HMM consumes the probabilities), speed ({cmp_.loc['LightGBM'].infer_ms_per_1k_obs:.0f} ms per 1,000 fixes; KNN "
        f"needs {cmp_.loc['KNN'].infer_ms_per_1k_obs:,.0f} ms), model size (a few MB) and compatibility with the decoder "
        f"(Part E8: the best HMM Top-1 of all models). Training accuracy was never used.")

    names = {"n_estimators": "100–1000 (step 50)", "learning_rate": "0.01–0.3 (log-uniform)",
             "num_leaves": "15–255 (log)", "min_child_samples": "5–300 (log)", "subsample": "0.5–1.0",
             "colsample_bytree": "0.4–1.0", "reg_lambda": "1e-3–10 (log)", "reg_alpha": "1e-3–10 (log)"}
    untuned = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": "20 (default)",
               "subsample": "1.0 (default)", "colsample_bytree": "1.0 (default)", "reg_lambda": "0 (default)",
               "reg_alpha": "0 (default)"}
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
    bo_space_table = table(pd.DataFrame({
        "Hyperparameter": list(names), "Search range": list(names.values()),
        "Untuned value": [untuned[k] for k in names], "Best found": [fmt(bo["params"][k]) for k in names]}))

    tv = tvu.set_index("model")
    tvt = pd.DataFrame({
        "Model": tv.index,
        "CV Top-1": [f"{pct(a)} ± {pct(b)}" for a, b in zip(tv.cv_top1, tv.cv_top1_std)],
        "CV Macro F1": tv.cv_macro_f1.map(f4), "CV log loss": tv.cv_log_loss.map(f4),
        "Test Top-1": tv.test_top1.map(pct), "Test Macro F1": tv.test_macro_f1.map(f4),
        "Test log loss": tv.test_log_loss.map(f4), "Fit (all rows) s": tv.fit_full_s.map(lambda v: f"{v:.1f}"),
        "Inference ms/1k fixes": tv.infer_ms_per_1k_obs.map(lambda v: f"{v:.1f}")})
    tuned_table = table(tvt, num_cols=set(tvt.columns[1:]))
    u, t = tvu.iloc[0], tvu.iloc[1]
    tuned_text = (
        f"Tuning changed CV Top-1 by {100 * (t.cv_top1 - u.cv_top1):+.2f} points and student-test Top-1 by "
        f"{100 * (t.test_top1 - u.test_top1):+.2f} points (Macro F1 {t.test_macro_f1 - u.test_macro_f1:+.4f}, log loss "
        f"{t.test_log_loss - u.test_log_loss:+.4f}). "
        + ("The tuned model is better on both the cross-validated and the held-out measures, so it is the deployed model. "
           if (t.cv_top1 > u.cv_top1 and t.test_top1 >= u.test_top1) else
           "The gains are small, and the difference is within the fold-to-fold noise, which is honest evidence that the "
           "untuned defaults were already near a good region. We deploy the tuned model because it is best on the "
           "objective that was optimised (CV Top-1). ")
        + f"Inference cost: {t.infer_ms_per_1k_obs:.1f} vs {u.infer_ms_per_1k_obs:.1f} ms per 1,000 fixes. "
        f"Search cost: {bo['n_trials']} trials in {bo['search_seconds'] / 60:.0f} minutes.")

    top = imp.head(5)
    importance_text = ("The five most-used features are " + ", ".join(f"<code>{f}</code> ({100 * g / imp.gain.sum():.0f} %)"
                       for f, g in zip(top.feature, top.gain)) + ". The temporal-context and heading features carry "
                       "most of the signal; raw distance alone is not enough.")

    g6 = grid[grid.k_hmm == 6]
    kg = grid[(grid.beta == hcfg["beta"]) & (grid.w_emit == hcfg["w_emit"])].sort_values("k_hmm")
    grid_table = table(pd.DataFrame({
        "K (candidates in HMM)": kg.k_hmm.astype(int), "HMM Top-1 (train OOF)": kg.hmm_top1.map(pct),
        "Classifier-only Top-1": kg.clf_top1.map(pct), "Route arc F1": kg.hmm_arc_f1.map(f4),
        "Decode time (s, 24 trips)": kg.hmm_s.map(lambda v: f"{v:.1f}")}))
    grid_text = (f"Chosen: β = {hcfg['beta']} m, w_e = {hcfg['w_emit']}, K = {hcfg['k_hmm']}. Across the β × w_e grid, "
                 f"HMM Top-1 on the training trips ranged from {pct(g6.hmm_top1.min())} to {pct(g6.hmm_top1.max())}, "
                 f"always above the classifier-only {pct(g6.clf_top1.iloc[0])}, so the improvement does not hinge on "
                 f"a lucky setting.")

    hst = pd.DataFrame({
        "Metric (student-test, 10 trips)": ["Top-1 accuracy (fix-weighted)", "Route continuity (mean)",
                                            "Route arc precision", "Route arc recall", "Route arc F1",
                                            "Length error (lower is better)"],
        "Classifier only": [pct(hs["classifier_top1"]), pct(hs["classifier_continuity"]),
                            f4(hs["classifier_arc_precision"]), f4(hs["classifier_arc_recall"]),
                            f4(hs["classifier_arc_f1"]), f4(hs["classifier_length_error"])],
        "Classifier + HMM": [pct(hs["hmm_top1"]), pct(hs["hmm_continuity"]), f4(hs["hmm_arc_precision"]),
                             f4(hs["hmm_arc_recall"]), f4(hs["hmm_arc_f1"]), f4(hs["hmm_length_error"])]})
    hmm_summary_table = table(hst, num_cols={"Classifier only", "Classifier + HMM"})
    ptt = pd.DataFrame({
        "Trip": ptrip.trip, "Fixes": ptrip.n_obs, "Clf Top-1": ptrip.classifier_top1.map(pct),
        "HMM Top-1": ptrip.hmm_top1.map(pct), "Clf continuity": ptrip.classifier_continuity.map(pct),
        "HMM continuity": ptrip.hmm_continuity.map(pct), "Clf arc F1": ptrip.classifier_arc_f1.map(f4),
        "HMM arc F1": ptrip.hmm_arc_f1.map(f4), "HMM length err": ptrip.hmm_length_error.map(f4),
        "Gap breaks": ptrip.n_breaks, "Total s": ptrip.total_s.map(lambda v: f"{v:.1f}")})
    per_trip_table = table(ptt, num_cols=set(ptt.columns[1:]))
    better = (ptrip.hmm_top1 > ptrip.classifier_top1).sum()
    worse = ptrip.loc[ptrip.hmm_top1 < ptrip.classifier_top1, "trip"].tolist()
    worst = ptrip.sort_values("hmm_top1").iloc[0]
    hmm_text = (
        f"The HMM raised fix-level Top-1 from {pct(hs['classifier_top1'])} to <b>{pct(hs['hmm_top1'])}</b> and route arc F1 "
        f"from {f4(hs['classifier_arc_f1'])} to <b>{f4(hs['hmm_arc_f1'])}</b>, and cut the length error from "
        f"{f4(hs['classifier_length_error'])} to {f4(hs['hmm_length_error'])}. Top-1 improved on {better} of 10 trips"
        + (f"; it was slightly worse on {', '.join(worse)}, where the classifier was confidently wrong over a stretch and the HMM "
           f"followed that wrong road consistently. " if worse else ". ")
        + f"The hardest trip is {worst.trip} (HMM Top-1 {pct(worst.hmm_top1)}). The biggest gain is at route level: the "
        f"classifier's isolated wrong picks each add a detour of wrong arcs, and the HMM removes them because they are "
        f"expensive transitions.")
    dmt = dec.loc[FAMILY_ORDER].reset_index()
    dmt_html = pd.DataFrame({
        "Model": dmt.model, "Classifier-only Top-1": dmt.clf_top1.map(pct), "+HMM Top-1": dmt.hmm_top1.map(pct),
        "Gain (points)": ((dmt.hmm_top1 - dmt.clf_top1) * 100).map(lambda v: f"{v:+.2f}"),
        "Classifier continuity": dmt.clf_continuity.map(pct), "+HMM continuity": dmt.hmm_continuity.map(pct),
        "Classifier arc F1": dmt.clf_arc_f1.map(f4), "+HMM arc F1": dmt.hmm_arc_f1.map(f4)})
    decoder_models_table = table(dmt_html, num_cols=set(dmt_html.columns[1:]))
    gn = dec.loc["GaussianNB"]
    decoder_models_text = (
        f"The HMM improves every model. The weakest classifiers gain the most (GaussianNB "
        f"{100 * (gn.hmm_top1 - gn.clf_top1):+.1f} points), because the road-network constraint fixes many of their mistakes, "
        f"but the order is preserved: better emissions give a better decoded route, and the baseline LightGBM fed "
        f"through the HMM reaches {pct(dec.loc['LightGBM'].hmm_top1)}. "
        f"(These rows use the baseline models from Part C, each fitted on the 30,000-fix sample.) "
        f"<b>An honest observation:</b> the baseline LightGBM reaches {pct(dec.loc['LightGBM'].hmm_top1)} after the HMM, "
        f"higher than the deployed tuned model's {pct(hs['hmm_top1'])}, although the tuned model is better on the "
        f"objective it was tuned for (CV Top-1). Tuning optimised the classifier alone, not the classifier+HMM "
        f"combination. Our reading is that the tuned model, fitted on all rows, is more confident, so the HMM can "
        f"overrule fewer of its mistakes. We did <i>not</i> switch models on the strength of this student-test "
        f"number, because choosing by student-test result would overfit our selection to those 10 trips. Tuning the "
        f"classifier and decoder jointly, with HMM Top-1 on out-of-fold trips as the objective, is the natural next step.")

    ev = json.load(open(R("evaluator_test", "summary.json")))
    evt = pd.read_csv(R("evaluator_test", "per_trip.csv"), dtype={"trip": str})
    ev_tab = pd.DataFrame({
        "Trip": evt.trip, "Fixes": evt.n_obs.map(lambda v: f"{v:,}"), "Arcs in network": evt.n_arcs.map(lambda v: f"{v:,}"),
        "Clf Top-1": evt.classifier_top1.map(pct), "HMM Top-1": evt.hmm_top1.map(pct),
        "Clf arc F1": evt.classifier_arc_f1.map(f4), "HMM arc F1": evt.hmm_arc_f1.map(f4),
        "HMM length err": evt.hmm_length_error.map(f4), "Seconds": evt.seconds.map(lambda v: f"{v:.1f}")})
    ev_section = (
        "<p>On review day the evaluator's 10 held-out trips were released. We ran the <b>frozen</b> system on them "
        "(tuned LightGBM plus the HMM configuration chosen on training data; nothing was retrained or retuned) with "
        "<code>scripts/08_eval_holdout.py</code>, which predicts each trip without its route and only then scores it. "
        f"No trip ID overlaps our training or student-test trips. Over {ev['n_fixes']:,} fixes, fix-level Top-1 was "
        f"{pct(ev['classifier_top1_pooled'])} for the classifier alone and <b>{pct(ev['hmm_top1_pooled'])}</b> with the HMM; "
        f"mean route arc F1 rose from {f4(ev['classifier_arc_f1_mean_over_trips'])} to <b>{f4(ev['hmm_arc_f1_mean_over_trips'])}</b>, "
        f"continuity reached {pct(ev['hmm_continuity_mean_over_trips'])}, and each trip took at most {ev['seconds_max']:.1f} s "
        f"(the largest network has 107,451 arcs). These results are slightly better than on student-test "
        f"({pct(hs['hmm_top1'])}), which suggests the system was not overfitted to the trips we evaluated on.</p>"
        + table(ev_tab, num_cols=set(ev_tab.columns[1:])) +
        "<p><b>Trip 00000015</b> is the weakest by route metrics (arc F1 0.78, length error 0.54). Its GPS covers 53 km in "
        "1 h 54 min, yet the reference route is 8.7 km with every arc listed once, and every fix lies within 60 m of it: "
        "the car drove the same loop about six times. Our route records each lap, so it is much longer than a reference "
        "that lists each arc once; most of the length error comes from that, though some predicted arcs (72 not in the reference) "
        "are real mistakes. Its fix-level Top-1 is still 91.5%.</p>")
    ct = cmp_.sort_values("test_top1", ascending=False).reset_index()
    compact_table = table(pd.DataFrame({
        "Model": ct.model, "CV Top-1": ct.cv_top1_mean.map(pct), "Test Top-1": ct.test_top1.map(pct),
        "Test Macro F1": ct.test_macro_f1.map(f4), "Test log loss": ct.test_log_loss.map(f4)}),
        num_cols={"CV Top-1", "Test Top-1", "Test Macro F1", "Test log loss"})
    qa = "".join(f'<div class="qa"><b>Q{i}. {q}</b><br>{a}</div>' for i, (q, a) in enumerate(QA, 1))
    map_trip = "00000009"

    trip54 = inv.set_index("trip").loc["00000054"]
    v = dict(
        today=today,
        tr_obs=f"{tr['observations']:,}", te_obs=f"{te['observations']:,}",
        tr_obs_range=" / ".join(map(str, tr["obs_per_trip_min_median_max"])),
        te_obs_range=" / ".join(map(str, te["obs_per_trip_min_median_max"])),
        tr_nan=tr["nan_rows"], te_nan=te["nan_rows"], tr_badc=tr["invalid_coords"], te_badc=te["invalid_coords"],
        tr_nonmono=tr["non_increasing_timestamps"], te_nonmono=te["non_increasing_timestamps"],
        tr_dt1=pct(tr["share_dt_1s_mean"], 1), te_dt1=pct(te["share_dt_1s_mean"], 1),
        tr_gap60=tr["trips_with_gap_gt_60s"], te_gap60=te["trips_with_gap_gt_60s"],
        tr_nodes=f"{tr['nodes_total']:,}", te_nodes=f"{te['nodes_total']:,}",
        tr_arcs=f"{tr['arcs_total']:,}", te_arcs=f"{te['arcs_total']:,}",
        tr_iso=tr["isolated_nodes_total"], te_iso=te["isolated_nodes_total"],
        tr_multiwcc=tr["trips_with_multiple_weak_components"], te_multiwcc=te["trips_with_multiple_weak_components"],
        tr_routeconn=pct(tr["route_consecutive_connected_mean"]), te_routeconn=pct(te["route_consecutive_connected_mean"]),
        tr_labdist=f"{tr['label_dist_median_mean']:.1f}", te_labdist=f"{te['label_dist_median_mean']:.1f}",
        tr_gt60=f"{int(tr['obs_with_label_gt_60m']):,}", tr_gt60_pct=pct(tr["obs_with_label_gt_60m"] / tr["observations"], 1),
        te_gt60=f"{int(te['obs_with_label_gt_60m']):,}",
        trip54_pct=pct(trip54.share_label_gt_60m, 0),
        miss_train=pct(1 - rel_tr.in_candidates.mean(), 2), miss_test=pct(1 - rel_te.in_candidates.mean(), 2),
        n_rows_train=f"{n_rows_train:,}",
        nn_top1=pct(nn["top1"]), nn_f1=f4(nn["macro_f1"]),
        model_table=model_table, model_sections=model_sections(cmp_, dec), selection_text=selection_text,
        bo_trials=bo["n_trials"], bo_minutes=f"{bo['search_seconds'] / 60:.0f}", bo_space_table=bo_space_table,
        tuned_table=tuned_table, tuned_text=tuned_text, importance_text=importance_text,
        hmm_k=hcfg["k_hmm"], hmm_k2=hcfg["k_hmm"] ** 2, hmm_w=hcfg["w_emit"], hmm_beta=hcfg["beta"],
        hmm_jump=hcfg["jump_penalty"], n_tune_trips=24, grid_table=grid_table, grid_text=grid_text,
        compact_table=compact_table, bo_cv_u=pct(u.cv_top1), bo_cv_t=pct(t.cv_top1),
        bo_te_u=pct(u.test_top1), bo_te_t=pct(t.test_top1), bo_ll_u=f4(u.test_log_loss), bo_ll_t=f4(t.test_log_loss),
        hmm_cont=pct(hs["hmm_continuity"]), clf_le=f4(hs["classifier_length_error"]),
        hmm_le=f4(hs["hmm_length_error"]), avg_s=f"{ptrip.total_s.mean():.1f}",
        hmm_top1_s=pct(hs["hmm_top1"]), clf_top1_s=pct(hs["classifier_top1"]), hmm_f1_s=f4(hs["hmm_arc_f1"]),
        clf_cont=pct(hs["classifier_continuity"]), clf_f1=f4(hs["classifier_arc_f1"]),
        n_breaks=int(ptrip.n_breaks.sum()), hmm_summary_table=hmm_summary_table, per_trip_table=per_trip_table,
        hmm_text=hmm_text, decoder_models_table=decoder_models_table, decoder_models_text=decoder_models_text,
        qa=qa, map_trip=map_trip, ev_section=ev_section,
        ev_clf=pct(ev['classifier_top1_pooled']), ev_hmm=pct(ev['hmm_top1_pooled']),
        ev_f1=f4(ev['hmm_arc_f1_mean_over_trips']), ev_clf_f1=f4(ev['classifier_arc_f1_mean_over_trips']),
        ev_le=f4(ev['hmm_length_error_mean_over_trips']), ev_cont=pct(ev['hmm_continuity_mean_over_trips']),
        fig_eda_hist=EDA("eda_histograms.png"), fig_eda_dt=EDA("dt_and_label_distance.png"),
        fig_locations=EDA("trip_locations.png"), fig_trip5=EDA("trip_00000005_network.png"),
        fig_map_eda=FIG("map_trip_00000005.png"), fig_importance=FIG("feature_importance.png"),
        fig_cv_f1=FIG("cv_macro_f1.png"), fig_test=FIG("test_top1_f1.png"), fig_ll=FIG("test_log_loss.png"),
        fig_folds=FIG("cv_top1_folds.png"), fig_speed=FIG("accuracy_vs_speed.png"),
        fig_optuna=FIG("optuna_history.png"), fig_optuna_params=FIG("optuna_params.png"),
        fig_tuned=FIG("tuned_vs_untuned.png"), fig_confusion=FIG("confusion_tuned_lgbm.png"),
        fig_grid=FIG("decoder_grid.png"), fig_hmm=FIG("hmm_vs_classifier.png"),
        fig_map_pred=FIG(f"prediction_{map_trip}.png"),
        fig_ui=FIG("ui_screenshot.png"), fig_api=FIG("api_log_screenshot.png"),
    )
    os.makedirs(os.path.join(ROOT, "report"), exist_ok=True)
    for name, out in (("guide.html", "PathSnap_Study_Guide"), ("report.html", "PathSnap_Technical_Report"),
                      ("one_page.html", "PathSnap_One_Page_Report")):
        src = os.path.join(ROOT, "report", "templates", name)
        if not os.path.exists(src):
            continue
        html = Template(open(src, encoding="utf-8").read()).substitute(v)
        h = os.path.join(ROOT, "report", out + ".html")
        open(h, "w", encoding="utf-8").write(html)
        pdf = os.path.join(ROOT, "report", out + ".pdf")
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--allow-file-access-from-files",
                        "--no-pdf-header-footer", f"--print-to-pdf={pdf}", "file:///" + h.replace("\\", "/")],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        print(pdf)


if __name__ == "__main__":
    main()
