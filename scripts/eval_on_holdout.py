"""Evaluate LightGBM model on hold-out split and export metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

try:  # optional dependency for vetting config
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

EPS = 1e-12


def temperature_scale(probs: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    temperature = max(temperature, 1e-3)
    logp = np.log(np.clip(probs, EPS, 1.0)) / temperature
    logp -= logp.max(axis=1, keepdims=True)
    exp = np.exp(logp)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: NDArray[np.float64], targets: NDArray[np.int_], bins: int = 10) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if not np.any(mask):
            continue
        acc = (predictions[mask] == targets[mask]).mean()
        conf = confidences[mask].mean()
        ece += np.abs(acc - conf) * mask.mean()
    return float(ece)


def compute_reliability_data(probs: NDArray[np.float64], targets: NDArray[np.int_], bins: int = 10) -> Dict[str, list[float]]:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    accs = []
    confs = []
    counts = []
    for i in range(bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if np.any(mask):
            accs.append(float((predictions[mask] == targets[mask]).mean()))
            confs.append(float(confidences[mask].mean()))
            counts.append(int(mask.sum()))
        else:
            accs.append(float("nan"))
            confs.append(float(bin_edges[i : i + 2].mean()))
            counts.append(0)
    return {"bin_edges": bin_edges.tolist(), "accuracy": accs, "confidence": confs, "counts": counts}


def precision_at_recall_threshold(y_true: NDArray[np.int_], scores: NDArray[np.float64], recall_thresh: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    mask = recall >= recall_thresh
    if not np.any(mask):
        return 0.0
    return float(precision[mask].max())


def recall_at_precision_threshold(y_true: NDArray[np.int_], scores: NDArray[np.float64], precision_thresh: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    mask = precision >= precision_thresh
    if not np.any(mask):
        return 0.0
    return float(recall[mask].max())


def physical_vetting_probs(
    probs: NDArray[np.float64],
    features: pd.DataFrame,
    weights: Dict[str, float],
) -> NDArray[np.float64]:
    logits = np.log(np.clip(probs, EPS, 1.0))

    vshape = features.get("vshape_ratio", pd.Series(np.zeros(len(features)))).to_numpy()
    odd_even = features.get("odd_even_diff_ppm", pd.Series(np.zeros(len(features)))).abs().to_numpy()
    secondary = features.get("secondary_depth_ppm", pd.Series(np.zeros(len(features)))).abs().to_numpy()
    duration = features.get("duration_days", pd.Series(np.zeros(len(features)))).to_numpy()
    period = features.get("period_days", pd.Series(np.ones(len(features)))).to_numpy()
    rho = features.get("rho_star_cgs", pd.Series(np.ones(len(features)))).to_numpy()

    flag_v = np.nan_to_num(vshape) > weights.get("v_threshold", 0.6)
    flag_oe = np.nan_to_num(odd_even) > weights.get("oe_threshold", 300.0)
    flag_sec = np.nan_to_num(secondary) > weights.get("sec_threshold", 200.0)
    flag_dur = np.nan_to_num(duration) > np.nan_to_num(period) / weights.get("dur_ratio", 4.0)
    flag_rho = (~np.isfinite(rho)) | (rho < weights.get("rho_min", 0.1)) | (rho > weights.get("rho_max", 5.0))

    penalties = (
        weights.get("v", 0.3) * flag_v
        + weights.get("oe", 0.2) * flag_oe
        + weights.get("sec", 0.2) * flag_sec
        + weights.get("dur", 0.15) * flag_dur
        + weights.get("rho", 0.15) * flag_rho
    )

    adjusted = logits.copy()
    adjusted[:, 0] -= penalties  # reduce CP logit
    adjusted[:, 1] += 0.5 * penalties  # nudge towards FP when flags fire
    return temperature_scale(np.exp(adjusted - adjusted.max(axis=1, keepdims=True)), 1.0)


def load_calibration(path: Optional[Path]) -> Optional[float]:
    if not path:
        return None
    data = json.loads(path.read_text())
    if data.get("method") == "temperature":
        return float(data.get("value", 1.0))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cal", type=Path, help="Calibration parameters JSON", required=False)
    parser.add_argument("--label-map", type=Path, help="JSON mapping label→index", required=False)
    parser.add_argument("--vetting-config", type=Path, help="YAML con pesos/umbrales de vetting", required=False)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probs-out", type=Path, help="Ruta opcional para guardar probabilidades (NPZ)")
    args = parser.parse_args()

    split = json.loads(args.split_json.read_text())
    holdout_ids = {str(sid) for sid in split.get("holdout_ids", [])}
    if not holdout_ids:
        raise ValueError("Split JSON does not contain hold-out IDs")

    df = pd.read_parquet(args.features)
    df["target_id"] = df["target_id"].astype(str)
    df_holdout = df[df["target_id"].isin(holdout_ids)].reset_index(drop=True)
    if df_holdout.empty:
        raise ValueError("Hold-out dataframe is empty")

    feature_cols = [col for col in df.columns if col not in {"target_id", "mission", "label"}]

    if args.label_map and args.label_map.exists():
        class_to_idx = json.loads(args.label_map.read_text())
        classes = sorted(class_to_idx, key=class_to_idx.get)
        idx_map = {int(v): k for k, v in class_to_idx.items()}
    else:
        classes = sorted(df["label"].astype(str).unique())
        idx_map = {i: label for i, label in enumerate(classes)}

    y_true_labels = df_holdout["label"].astype(str).to_numpy()
    class_indices = {label: idx for idx, label in idx_map.items()}
    y_true = np.array([class_indices[label] for label in y_true_labels])

    booster = lgb.Booster(model_file=str(args.model))
    probs = booster.predict(df_holdout[feature_cols].to_numpy())
    if probs.ndim == 1:
        probs = np.vstack([1 - probs, probs]).T

    temperature = load_calibration(args.cal)
    if temperature is not None:
        probs = temperature_scale(probs, temperature)

    vetting_weights = None
    if args.vetting_config and args.vetting_config.exists():
        if yaml is None:
            raise ImportError("PyYAML is required to parse the vetting config")
        vetting_weights = yaml.safe_load(args.vetting_config.read_text())
        probs_vetted = physical_vetting_probs(probs, df_holdout, vetting_weights)
    else:
        probs_vetted = probs

    preds = probs_vetted.argmax(axis=1)
    one_hot = np.eye(len(classes), dtype=float)[y_true]
    metrics = {
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "log_loss": float(log_loss(y_true, probs_vetted, labels=list(range(len(classes))))),
        "ece": float(expected_calibration_error(probs_vetted, y_true)),
        "report": classification_report(y_true, preds, target_names=classes, zero_division=0),
        "brier": float(np.mean(np.sum((probs_vetted - one_hot) ** 2, axis=1))),
    }

    missions = df_holdout["mission"].astype(str).unique()
    binary_metrics = {}
    for mission in missions:
        mask = df_holdout["mission"].astype(str) == mission
        y_bin = (df_holdout.loc[mask, "label"] == "CP").astype(int).to_numpy()
        scores = probs_vetted[mask, class_indices["CP"]]
        if len(np.unique(y_bin)) < 2:
            continue
        binary_metrics[mission] = {
            "roc_auc": float(roc_auc_score(y_bin, scores)),
            "pr_auc": float(average_precision_score(y_bin, scores)),
            "precision_at_recall_0.96": precision_at_recall_threshold(y_bin, scores, 0.96),
            "recall_at_precision_0.90": recall_at_precision_threshold(y_bin, scores, 0.90),
        }

    cm = confusion_matrix(y_true, preds, labels=list(range(len(classes))))
    reliability = compute_reliability_data(probs_vetted, y_true)

    summary = {
        "classes": classes,
        "metrics_multiclass": metrics,
        "metrics_binary": binary_metrics,
        "confusion_matrix": cm.tolist(),
        "reliability": reliability,
        "temperature": temperature,
        "n_samples": int(len(df_holdout)),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"Saved hold-out metrics to {args.out}")

    if args.probs_out:
        args.probs_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.probs_out,
            probs=probs_vetted.astype(np.float32),
            labels=y_true,
            target_id=df_holdout["target_id"].to_numpy(),
            classes=np.array(classes),
        )
        print(f"Saved hold-out probabilities to {args.probs_out}")


if __name__ == "__main__":
    main()
