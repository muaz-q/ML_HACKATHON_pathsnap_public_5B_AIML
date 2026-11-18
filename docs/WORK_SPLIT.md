# Work split

Everyone works on every phase. Each person has their own branch and keeps it for the whole hackathon. Each person's files are separate, so pull requests can be merged in any order.

| Person | Branch |
|---|---|
| Prakhruti | `prakhruti` |
| Meha | `meha` |
| Shranaya | `shranaya` |
| Muaz | `muaz` |

## Phase 1 (Review 1): data preparation

| Person | Part | Files |
|---|---|---|
| Prakhruti | Loading, cleaning, projection to metres | `pathsnap/__init__.py`, `pathsnap/data.py` |
| Meha | Route → per-fix labels; exploratory analysis | `pathsnap/labels.py`, `scripts/01_eda.py`, `results/eda/*` |
| Shranaya | Candidate generation + 23 features | `pathsnap/features.py` |
| Muaz | Prepared-dataset builder + summary | `scripts/02_build_dataset.py`, `scripts/02b_data_prep_summary.py`, `results/data_prep/*` |

## Phase 2 (final): models, tuning, HMM, system

| Person | Part | Files |
|---|---|---|
| Prakhruti | Metrics + 10-model comparison | `pathsnap/metrics.py`, `scripts/03_train_models.py`, model results |
| Meha | Model plots + Bayesian optimisation | `scripts/03b_plot_model_comparison.py`, `scripts/04_bayes_opt.py`, tuning results, tuned model |
| Shranaya | HMM/Viterbi + decoder evaluation | `pathsnap/hmm.py`, `pathsnap/pipeline.py`, `scripts/05_hmm_eval.py`, HMM results |
| Muaz | API, UI, maps, report | `api/`, `ui/`, `scripts/06_*.py`, `scripts/07_*.py`, maps, figures, report, README |

## How to commit your part

First time only:

```bash
git clone https://github.com/muaz-q/ML_HACKATHON_pathsnap_public_5B_AIML.git
cd ML_HACKATHON_pathsnap_public_5B_AIML
git checkout <your-branch>
```

For each phase:

```bash
git checkout <your-branch>
git pull origin main
# copy your files in (same folder paths), then commit as STEPS.txt says
git push origin <your-branch>
```

Then open a pull request from your branch into `main` on GitHub.

Before you commit, read every file in your part. Each person explains their own part in the viva.
