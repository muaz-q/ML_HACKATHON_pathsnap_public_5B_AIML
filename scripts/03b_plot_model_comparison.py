"""Model-comparison figures (stage 2) from results/model_comparison.csv,
results/model_cv_folds.csv and results/mlp_loss_curve.csv written by 03_train_models.py.
Outputs: results/figures/{cv_macro_f1,test_top1_f1,test_log_loss,cv_top1_folds,accuracy_vs_speed,mlp_history}.png
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
C1, C2, C3 = "#1565c0", "#d32f2f", "#2e7d32"


def bar(ax, names, vals, err=None, color=C1, fmt="{:.3f}"):
    y = np.arange(len(names))
    ax.barh(y, vals, xerr=err, color=color, alpha=.85, capsize=3)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8)


def model_plots():
    cmp_ = pd.read_csv(os.path.join(ROOT, "results", "model_comparison.csv"))
    cmp_ = cmp_.sort_values("cv_top1_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    bar(ax, cmp_.model, cmp_.cv_macro_f1_mean, cmp_.cv_macro_f1_std)
    ax.set_xlim(0.85, 1.0); ax.set_xlabel("Macro F1 (mean ± std over 5 trip-grouped folds)")
    ax.set_title("Cross-validation Macro F1 by model (train, GroupKFold=5)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "cv_macro_f1.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    bar(ax[0], cmp_.model, cmp_.test_top1, color=C1); ax[0].set_xlim(.75, 1); ax[0].set_title("Student-test Top-1 accuracy")
    bar(ax[1], cmp_.model, cmp_.test_macro_f1, color=C3); ax[1].set_xlim(.85, 1); ax[1].set_title("Student-test Macro F1")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "test_top1_f1.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    bar(ax, cmp_.model, cmp_.test_log_loss, color=C2)
    ax.set_xscale("log"); ax.set_xlabel("Log loss (log scale, lower is better)")
    ax.set_title("Student-test log loss by model")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "test_log_loss.png"), dpi=140); plt.close(fig)

    folds = pd.read_csv(os.path.join(ROOT, "results", "model_cv_folds.csv"))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    order = cmp_.model.tolist()
    ax.boxplot([folds[folds.model == m].top1 for m in order], tick_labels=order)
    ax.set_ylabel("Top-1 accuracy per fold"); ax.set_title("Fold-to-fold stability of CV Top-1 accuracy")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "cv_top1_folds.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(cmp_.infer_ms_per_1k_obs, cmp_.test_top1, c=C1)
    for _, r in cmp_.iterrows():
        ax.annotate(r.model, (r.infer_ms_per_1k_obs, r.test_top1), fontsize=8, xytext=(4, 2), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("Inference time, ms per 1,000 GPS observations (log)")
    ax.set_ylabel("Student-test Top-1"); ax.set_title("Accuracy vs inference cost"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "accuracy_vs_speed.png"), dpi=140); plt.close(fig)

    mlp = pd.read_csv(os.path.join(ROOT, "results", "mlp_loss_curve.csv"))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mlp.epoch, mlp.train_loss, "o-", c=C1, label="training loss (log loss)")
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
    ax2 = ax.twinx(); ax2.plot(mlp.epoch, mlp.val_accuracy, "s-", c=C2, label="validation accuracy (10% hold-out)")
    ax2.set_ylabel("validation accuracy")
    ax.set_title("MLP training history (the only model with epoch-wise history)")
    fig.legend(loc="center right", bbox_to_anchor=(.85, .5)); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "mlp_history.png"), dpi=140); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    model_plots()
