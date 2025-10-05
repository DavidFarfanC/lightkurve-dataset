"""Generate canonical evaluation figures (metrics tables, confusion matrices, reliability curves)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from sklearn.metrics import confusion_matrix, f1_score, log_loss

EPS = 1e-12


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if not np.any(mask):
            continue
        acc = (predictions[mask] == labels[mask]).mean()
        conf = confidences[mask].mean()
        ece += abs(acc - conf) * mask.mean()
    return float(ece)


def compute_brier(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    one_hot = np.eye(n_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def reliability_curve(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    accuracy = np.empty(bins)
    confidence = np.empty(bins)
    counts = np.zeros(bins, dtype=int)
    for i in range(bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if np.any(mask):
            accuracy[i] = (predictions[mask] == labels[mask]).mean()
            confidence[i] = confidences[mask].mean()
            counts[i] = int(mask.sum())
        else:
            accuracy[i] = np.nan
            confidence[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1])
    return accuracy, confidence, counts


def plot_metrics_table(ax, headers: Iterable[str], rows: Iterable[Iterable[str]], title: str) -> None:
    ax.axis("off")
    table = ax.table(cellText=list(rows), colLabels=list(headers), cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.3)
    ax.set_title(title, fontsize=12, pad=10)


def plot_confusion(ax, matrix: np.ndarray, classes: Iterable[str], title: str) -> None:
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(classes)), labels=list(classes))
    ax.set_yticks(range(len(classes)), labels=list(classes))
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, int(value), ha="center", va="center", color="black", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_reliability(ax, curve_1: Tuple[np.ndarray, np.ndarray, np.ndarray], curve_2: Tuple[np.ndarray, np.ndarray, np.ndarray], label_1: str, label_2: str) -> None:
    accuracy1, confidence1, _ = curve_1
    accuracy2, confidence2, _ = curve_2
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.plot(confidence1, accuracy1, marker="o", label=label_1)
    ax.plot(confidence2, accuracy2, marker="s", label=label_2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend(loc="lower right")


def reorder_probs(probs: np.ndarray, classes: Iterable[str], label_map: Dict[str, int] | None) -> np.ndarray:
    if label_map is None:
        return probs
    indices = [label_map[label] for label in classes]
    return probs[:, indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True, help="OOF probabilities NPZ")
    parser.add_argument("--holdout", type=Path, required=True, help="Hold-out metrics JSON")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory to store PNG files")
    parser.add_argument("--label-map", type=Path, help="JSON label→index mapping", required=False)
    parser.add_argument("--holdout-probs", type=Path, help="Optional hold-out probabilities NPZ", required=False)
    args = parser.parse_args()

    label_map = None
    if args.label_map and args.label_map.exists():
        label_map = json.loads(args.label_map.read_text())

    holdout_data = json.loads(args.holdout.read_text())
    classes = holdout_data["classes"]

    with np.load(args.oof, allow_pickle=True) as data:
        probs_oof = data["probs"].astype(float)
        labels_oof = data["labels"].astype(int)
    probs_oof = reorder_probs(probs_oof, classes, label_map)

    probs_holdout = None
    labels_holdout = None
    if args.holdout_probs and args.holdout_probs.exists():
        with np.load(args.holdout_probs, allow_pickle=True) as data:
            probs_holdout = reorder_probs(data["probs"].astype(float), classes, label_map)
            labels_holdout = data["labels"].astype(int)

    preds_oof = probs_oof.argmax(axis=1)
    cm_oof = confusion_matrix(labels_oof, preds_oof, labels=list(range(len(classes))))
    ece_oof = expected_calibration_error(probs_oof, labels_oof)
    brier_oof = compute_brier(probs_oof, labels_oof, len(classes))
    macro_f1_oof = f1_score(labels_oof, preds_oof, average="macro")
    log_loss_oof = log_loss(labels_oof, probs_oof, labels=list(range(len(classes))))

    metrics_holdout = holdout_data["metrics_multiclass"]
    macro_f1_holdout = metrics_holdout.get("macro_f1", float("nan"))
    log_loss_holdout = metrics_holdout.get("log_loss", float("nan"))
    ece_holdout = metrics_holdout.get("ece", float("nan"))
    brier_holdout = metrics_holdout.get("brier", float("nan"))

    binary_metrics = holdout_data.get("metrics_binary", {})
    cm_holdout = np.array(holdout_data.get("confusion_matrix", []), dtype=float)

    if probs_holdout is not None and labels_holdout is not None:
        curve_holdout = reliability_curve(probs_holdout, labels_holdout)
        curve_holdout = (
            curve_holdout[0],
            curve_holdout[1],
            curve_holdout[2],
        )
    else:
        rel = holdout_data.get("reliability", {})
        accuracy = np.array(rel.get("accuracy", []), dtype=float)
        confidence = np.array(rel.get("confidence", []), dtype=float)
        counts = np.array(rel.get("counts", []), dtype=int)
        curve_holdout = (accuracy, confidence, counts)

    curve_oof = reliability_curve(probs_oof, labels_oof)

    args.outdir.mkdir(parents=True, exist_ok=True)

    headers = ["Split", "Macro-F1", "LogLoss", "ECE", "Brier"]
    rows = [
        [
            "OOF (mean)",
            f"{macro_f1_oof:.3f}",
            f"{log_loss_oof:.3f}",
            f"{ece_oof:.3f}",
            f"{brier_oof:.3f}",
        ],
        [
            "Hold-out",
            f"{macro_f1_holdout:.3f}",
            f"{log_loss_holdout:.3f}",
            f"{ece_holdout:.3f}",
            f"{brier_holdout:.3f}",
        ],
    ]
    fig, ax = plt.subplots(figsize=(6, 2))
    plot_metrics_table(ax, headers, rows, "Multiclass Metrics")
    fig.tight_layout()
    fig.savefig(args.outdir / "metrics_multiclass.png", dpi=200)
    plt.close(fig)

    if binary_metrics:
        missions = sorted(binary_metrics.keys())
        headers_bin = ["Mission", "ROC-AUC", "PR-AUC", "P@R≥0.96", "R@P≥0.9"]
        rows_bin = []
        for mission in missions:
            vals = binary_metrics[mission]
            rows_bin.append(
                [
                    mission,
                    f"{vals.get('roc_auc', float('nan')):.3f}",
                    f"{vals.get('pr_auc', float('nan')):.3f}",
                    f"{vals.get('precision_at_recall_0.96', float('nan')):.3f}",
                    f"{vals.get('recall_at_precision_0.90', float('nan')):.3f}",
                ]
            )
        fig, ax = plt.subplots(figsize=(7, 1.8 + 0.3 * len(missions)))
        plot_metrics_table(ax, headers_bin, rows_bin, "Binary Metrics (CP vs no-CP)")
        fig.tight_layout()
        fig.savefig(args.outdir / "metrics_binary.png", dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    plot_confusion(axes[0], cm_oof, classes, "Confusion (OOF)")
    if cm_holdout.size:
        plot_confusion(axes[1], cm_holdout, classes, "Confusion (Hold-out)")
    else:
        axes[1].axis("off")
        axes[1].set_title("No hold-out confusion matrix")
    fig.tight_layout()
    fig.savefig(args.outdir / "confusion_matrices.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(1, 1)
    ax_rel = fig.add_subplot(gs[0])
    plot_reliability(ax_rel, curve_oof, curve_holdout, "OOF", "Hold-out")
    fig.tight_layout()
    fig.savefig(args.outdir / "reliability.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
