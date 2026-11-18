# PathSnap: Map-Matching & Route Reconstruction

PathSnap takes a trip's noisy GPS fixes and its road network, and returns the road arc the vehicle was on at every fix plus a continuous, drivable route.

The pipeline has five stages:

1. **Candidates.** For every GPS fix, take the 12 nearest directed road arcs.
2. **Features.** Compute 23 features for each (fix, candidate) pair: distance, heading agreement, temporal context, topology.
3. **Classifier.** Score each candidate with P(candidate is the reference arc). Ten models were compared; tuned LightGBM won.
4. **HMM + Viterbi.** Combine the classifier probabilities (emissions) with road-network shortest-path distances (transitions) to decode the most likely route.
5. **Deployment.** The FastAPI backend runs on Laptop 1 and the browser UI on Laptop 2.

## Repository layout

| Folder | Contents |
|---|---|
| `pathsnap/` | Library: `data.py` (loading, projection), `labels.py` (route → per-fix labels), `features.py` (candidates + features), `hmm.py` (HMM/Viterbi), `metrics.py`, `pipeline.py` (end-to-end inference) |
| `scripts/01_eda.py` | Dataset exploration: inventory, data-quality checks, plots, Folium map |
| `scripts/02_build_dataset.py` | Builds the candidate tables for train / student_test |
| `scripts/03_train_models.py` | Trains and compares 10 models with trip-grouped 5-fold CV |
| `scripts/04_bayes_opt.py` | Optuna TPE Bayesian optimisation of LightGBM; exports `artifacts/model.joblib` |
| `scripts/05_hmm_eval.py` | Decoder tuning; classifier-only vs classifier+HMM evaluation |
| `scripts/06_plots_and_maps.py` | Every report figure and the Folium prediction maps |
| `api/main.py` | FastAPI backend (Laptop 1) |
| `ui/index.html` | Evaluator UI (Laptop 2) |
| `artifacts/` | Final model, untuned model, HMM config, baseline models |
| `results/` | CSV logs, Optuna trials, figures, maps |
| `report/` | Technical report and study guide |

## Install

Requires Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Put the raw data in place. It is not committed because it is too large.

```
data/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
data/student_test/<trip_id>/...
```

## Reproduce the experiments

```bash
python scripts/01_eda.py
python scripts/02_build_dataset.py train,student_test
python scripts/03_train_models.py
python scripts/04_bayes_opt.py          # N_TRIALS env var, default 30
python scripts/05_hmm_eval.py
python scripts/06_plots_and_maps.py
```

All seeds are fixed at 42.

## Deployment (two laptops)

**Laptop 1: API.** Put the 10 evaluator trips in a folder, then:

```bash
set PATHSNAP_EVAL_DIR=C:\path\to\evaluator_trips      # Linux/macOS: export PATHSNAP_EVAL_DIR=...
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

`PATHSNAP_EVAL_DIR` defaults to `data/student_test`. Find Laptop 1's IP with `ipconfig`, and allow port 8000 through the firewall. Each request is logged in this terminal: trip, classifier Top-1, HMM Top-1, continuity, route F1 and timing.

**Laptop 2: UI.**

```bash
cd ui
python -m http.server 5500
```

Open `http://localhost:5500`, enter `http://<laptop-1-ip>:8000`, click **Connect**, then pick a trip.

### Endpoints

- `GET /health`
- `GET /trips` lists the trip ids in `PATHSNAP_EVAL_DIR`.
- `POST /predict {"trip_id": "..."}` runs the full pipeline. It returns the GPS fixes, the predicted route, the classifier-only route, the predicted arc per fix, the nearby road network, server-side metrics and timings.

## Leakage safeguards

- The API loads the trip with `with_route=False` and runs prediction first. The `.route` file is read only afterwards, to compute metrics, and the reference geometry is never sent to the UI.
- No feature uses absolute coordinates, node/arc ids or the reference route.
- Cross-validation is grouped by trip. Decoder parameters were tuned on out-of-fold training probabilities. Evaluator trips were never used.
