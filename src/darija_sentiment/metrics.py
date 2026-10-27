"""Metrics, bootstrap confidence intervals, confusion matrices and result I/O."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .config import FIGURES_DIR, LABELS, METRICS_DIR, PREDICTIONS_DIR, SEED


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", labels=range(len(LABELS)))


def evaluate(y_true, y_pred) -> dict:
    per_class = f1_score(y_true, y_pred, average=None, labels=range(len(LABELS)))
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        **{f"f1_{name}": score for name, score in zip(LABELS, per_class)},
        "n": len(y_true),
    }


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, alpha: float = 0.05) -> tuple:
    """Percentile bootstrap interval for macro-F1 (resampling test examples)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(SEED)
    scores = [
        macro_f1(y_true[idx], y_pred[idx])
        for idx in rng.integers(0, len(y_true), size=(n_boot, len(y_true)))
    ]
    return tuple(np.quantile(scores, [alpha / 2, 1 - alpha / 2]))


def paired_bootstrap(y_true, pred_a, pred_b, n_boot: int = 1000) -> dict:
    """How often system A beats system B on the same resampled test sets."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(SEED)
    diffs = np.array([
        macro_f1(y_true[idx], pred_a[idx]) - macro_f1(y_true[idx], pred_b[idx])
        for idx in rng.integers(0, len(y_true), size=(n_boot, len(y_true)))
    ])
    return {
        "mean_diff": diffs.mean(),
        "ci_low": np.quantile(diffs, 0.025),
        "ci_high": np.quantile(diffs, 0.975),
        "p_a_not_better": (diffs <= 0).mean(),
    }


def evaluate_by_script(df: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    """Overall metrics plus one row per writing script."""
    rows = [{"subset": "all", **evaluate(df["label"], df[pred_col])}]
    for script in ("arabic", "latin", "mixed"):
        part = df[df["script"] == script]
        if len(part):
            rows.append({"subset": script, **evaluate(part["label"], part[pred_col])})
    return pd.DataFrame(rows).set_index("subset")


def plot_confusion(y_true, y_pred, title: str, ax=None, normalize: bool = True):
    """Row-normalised confusion matrix: each row shows where one true class goes."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(LABELS)))
    shown = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    ax = ax or plt.subplots(figsize=(4.2, 3.8))[1]
    ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            dark = shown[i, j] > (0.5 if normalize else shown.max() / 2)
            ax.text(
                j, i, f"{shown[i, j]:.2f}\n({cm[i, j]})" if normalize else cm[i, j],
                ha="center", va="center", fontsize=9, color="white" if dark else "black",
            )
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    return ax


def save_predictions(df: pd.DataFrame, system: str, split: str = "test") -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREDICTIONS_DIR / f"{system}_{split}.csv", index=False)


def load_predictions(system: str, split: str = "test") -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_DIR / f"{system}_{split}.csv", keep_default_na=False)


def save_metrics(metrics: dict, system: str) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / f"{system}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)


def save_figure(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
