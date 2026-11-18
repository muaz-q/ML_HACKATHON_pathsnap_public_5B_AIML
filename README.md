# PathSnap: Map-Matching & Route Reconstruction

UE24CS352A Machine Learning Hackathon, team 5B AIML.

PathSnap takes the noisy GPS fixes of a trip, plus that trip's road network, and recovers the road arc the vehicle was on at each fix. It then joins those arcs into a continuous route that could actually be driven.

## Approach

1. **Candidates.** For every GPS fix, take the 12 nearest directed road arcs.
2. **Features.** Compute 23 location-independent features for each (fix, candidate) pair: distance, heading agreement, temporal context and topology.
3. **Classifier.** Score each candidate with P(candidate is the reference arc). At least 5 models are compared with trip-grouped cross-validation.
4. **Tuning.** Bayesian hyperparameter optimisation (Optuna TPE) of the selected model.
5. **HMM + Viterbi.** Decode the most likely connected route. Emissions come from the classifier; transitions come from shortest-path distances on the road network.
6. **Deployment.** FastAPI backend on Laptop 1, browser UI on Laptop 2.

## Status

| Phase | Contents | Status |
|---|---|---|
| 1 | Data loading and cleaning, projection, route→label alignment, EDA, candidates + features (prepared dataset) | in progress |
| 2 | Metrics, model comparison, Bayesian optimisation, HMM/Viterbi, FastAPI + UI, report | in progress |

Each team member has a branch and contributes to every phase. See [`docs/WORK_SPLIT.md`](docs/WORK_SPLIT.md).

## Layout

```
pathsnap/    library code (data, labels, features, metrics, hmm, pipeline)
scripts/     numbered experiment scripts 01..07
api/         FastAPI backend
ui/          evaluator UI
artifacts/   trained models and configs
results/     logs, CSVs, figures, maps
report/      technical report
data/        raw trips (not committed): data/train/<id>/, data/student_test/<id>/
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
