# Work split

Each stage has an owner and a branch. The owner commits their stage on that branch and opens a pull request into `main`. Merge in order (1 → 2 → 3 → 4), because later stages import earlier code.

| Stage | Branch | Owner | Files |
|---|---|---|---|
| 1. Data & EDA | `stage1-data-eda` | Member 1 | `pathsnap/__init__.py`, `pathsnap/data.py`, `pathsnap/labels.py`, `scripts/01_eda.py`, `scripts/02_build_dataset.py`, `results/eda/*` |
| 2. Features & models | `stage2-features-models` | Member 2 | `pathsnap/features.py`, `pathsnap/metrics.py`, `scripts/03_train_models.py`, `results/model_*.csv`, `results/mlp_loss_curve.csv`, `results/nearest_arc_baseline.json`, `results/train_models.log`, `results/figures/` (model plots), `artifacts/baseline_models/*` |
| 3. Tuning & HMM | `stage3-tuning-hmm` | Member 3 | `scripts/04_bayes_opt.py`, `pathsnap/hmm.py`, `pathsnap/pipeline.py`, `scripts/05_hmm_eval.py`, `results/bayes_opt/*`, `results/hmm/*`, `artifacts/model.joblib`, `artifacts/model_untuned.joblib`, `artifacts/hmm_config.json` |
| 4. API, UI & visualisation | `stage4-api-ui-viz` | Member 4 | `api/main.py`, `ui/index.html`, `scripts/06_plots_and_maps.py`, `scripts/07_screenshot_maps.py`, `results/maps/*`, `results/figures/*`, `report/*`, final README |

## How to commit your stage

```bash
git clone https://github.com/muaz-q/ML_HACKATHON_pathsnap_public_5B_AIML.git
cd ML_HACKATHON_pathsnap_public_5B_AIML
git checkout <your-branch>
git pull origin main                 # pick up merged earlier stages
# copy your stage's files into the repo (same folder paths)
git add -A
git commit -m "<stage>: <what you added>"
git push origin <your-branch>
```

Then open a pull request from your branch into `main` on GitHub.

Before committing, run your stage's scripts and read every file you commit. You must be able to explain them in the viva.
