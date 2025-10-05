"""Blend LightGBM and CNN probabilities with do-no-harm guarantees and physical vetting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from sklearn.metrics import classification_report, f1_score, log_loss

EPS = 1e-12


def softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def temperature_scale(probs: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    temperature = max(temperature, 1e-3)
    logp = np.log(np.clip(probs, EPS, 1.0)) / temperature
    return softmax(logp)


def optimise_temperature(probs: NDArray[np.float64], y_true: NDArray[np.int_]) -> float:
    def objective(t: float) -> float:
        scaled = temperature_scale(probs, t)
        return log_loss(y_true, scaled, labels=list(range(probs.shape[1])))

    res = minimize_scalar(objective, bounds=(0.5, 5.0), method="bounded")
    return float(res.x if res.success else 1.0)


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


def load_oof(path: Path, fallback_ids: NDArray[np.str_] | None = None) -> tuple[NDArray[np.float64], NDArray[np.int_], NDArray[np.str_]]:
    data = np.load(path, allow_pickle=True)
    probs = data["probs"].astype(float)
    labels = data["labels"].astype(int)
    if "target_id" in data:
        target_ids = data["target_id"].astype(str)
    elif fallback_ids is not None and len(fallback_ids) == len(labels):
        target_ids = fallback_ids
    else:
        raise ValueError(f"OOF file {path} missing 'target_id' and no fallback provided")
    return probs, labels, target_ids


def compute_alpha(probs_lgbm: NDArray[np.float64], probs_cnn: NDArray[np.float64], y_true: NDArray[np.int_], alpha_max: float) -> float:
    grid = np.linspace(0.0, alpha_max, int(alpha_max * 100) + 1)
    best_alpha = 0.0
    best_loss = np.inf
    for alpha in grid:
        blend = (1 - alpha) * probs_lgbm + alpha * probs_cnn
        loss = log_loss(y_true, blend, labels=list(range(blend.shape[1])))
        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha
    return best_alpha


def apply_gating(alpha_base: float, features: pd.DataFrame) -> NDArray[np.float64]:
    period = features["period_days"].to_numpy()
    duration = features["duration_days"].to_numpy()
    snr = features["snr_in_transit"].to_numpy()
    vshape = features["vshape_ratio"].to_numpy()

    snr_thr = np.nanpercentile(snr, 30)
    dur_q90 = np.nanpercentile(duration, 90)

    cond_low_snr = np.nan_to_num(snr) < snr_thr
    cond_good_shape = np.nan_to_num(vshape) < 0.5
    cond_short_period = np.nan_to_num(period) < 5.0
    cond_short_duration = np.nan_to_num(duration) < dur_q90

    alpha_case = np.full_like(snr, alpha_base, dtype=float)
    ramp_mask = (cond_low_snr & cond_good_shape) | (cond_short_period & cond_short_duration)
    alpha_case[ramp_mask] = np.minimum(alpha_base + 0.15, 0.4)
    return alpha_case


def physical_vetting_probs(probs: NDArray[np.float64], features: pd.DataFrame, weights: dict[str, float]) -> NDArray[np.float64]:
    logits = np.log(np.clip(probs, EPS, 1.0))

    vshape = features.get("vshape_ratio", pd.Series(np.zeros(len(features)))).to_numpy()
    odd_even = features.get("odd_even_diff_ppm", pd.Series(np.zeros(len(features)))).abs().to_numpy()
    secondary = features.get("secondary_depth_ppm", pd.Series(np.zeros(len(features)))).abs().to_numpy()
    duration = features.get("duration_days", pd.Series(np.zeros(len(features)))).to_numpy()
    period = features.get("period_days", pd.Series(np.ones(len(features)))).to_numpy()
    rho = features.get("rho_star_cgs", pd.Series(np.ones(len(features)))).to_numpy()

    flag_v = np.nan_to_num(vshape) > 0.6
    flag_oe = np.nan_to_num(odd_even) > 300.0
    flag_sec = np.nan_to_num(secondary) > 200.0
    flag_dur = np.nan_to_num(duration) > np.nan_to_num(period) / 4.0
    flag_rho = (~np.isfinite(rho)) | (rho < 0.1) | (rho > 5.0)

    penalties = (
        weights["v"] * flag_v
        + weights["oe"] * flag_oe
        + weights["sec"] * flag_sec
        + weights["dur"] * flag_dur
        + weights["rho"] * flag_rho
    )

    adjusted = logits.copy()
    adjusted[:, 0] -= penalties  # reduce CP logit
    adjusted[:, 1] += 0.5 * penalties  # nudge towards FP when flags fire
    return softmax(adjusted)


def compute_metrics(probs: NDArray[np.float64], y_true: NDArray[np.int_], classes: list[str]) -> dict[str, float]:
    preds = probs.argmax(axis=1)
    metrics = {
        "macro_f1": f1_score(y_true, preds, average="macro"),
        "log_loss": log_loss(y_true, probs, labels=list(range(len(classes)))),
        "ece": expected_calibration_error(probs, y_true),
        "report": classification_report(y_true, preds, target_names=classes, zero_division=0),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/processed/lightgbm/features.parquet"))
    parser.add_argument("--lightgbm-oof", type=Path, default=Path("data/processed/lightgbm/lightgbm_oof.npz"))
    parser.add_argument("--cnn-oof", type=Path, default=Path("data/processed/cnn/cnn_oof.npz"))
    parser.add_argument("--alpha-max", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("data/processed/ensemble/blend_probs.npz"))
    args = parser.parse_args()

    features = pd.read_parquet(args.features).reset_index(drop=True)
    fallback_ids = features["target_id"].astype(str).to_numpy()
    probs_lgbm, y_true_lgbm, ids_lgbm = load_oof(args.lightgbm_oof, fallback_ids)
    probs_cnn, y_true_cnn, ids_cnn = load_oof(args.cnn_oof)

    features = features.assign(target_id=features["target_id"].astype(str)).reset_index(drop=True)
    df_lgbm = pd.DataFrame({
        "target_id": ids_lgbm.astype(str),
        "label_lgbm": y_true_lgbm,
        "idx_lgbm": np.arange(len(ids_lgbm)),
    })
    df_cnn = pd.DataFrame({
        "target_id": ids_cnn.astype(str),
        "label_cnn": y_true_cnn,
        "idx_cnn": np.arange(len(ids_cnn)),
    })

    merged = features[["target_id", "label"]].merge(df_lgbm, on="target_id", how="inner")
    merged = merged.merge(df_cnn, on="target_id", how="inner")

    if merged.empty:
        raise ValueError("No overlapping targets between LightGBM and CNN OOF files")

    if len(merged) != len(features):
        print(
            f"Warning: {len(features) - len(merged)} targets present in features but missing in one of the OOF files; they will be ignored."
        )

    # Map string labels to integer indices following feature ordering
    classes = sorted(features["label"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    features["label_idx"] = features["label"].map(label_to_idx)
    merged = merged.merge(features[["target_id", "label_idx"]], on="target_id", how="left", suffixes=("", "_feat"))

    if not np.array_equal(merged["label_idx"].to_numpy(), merged["label_lgbm"].to_numpy()):
        raise ValueError("Label mismatch between features and LightGBM OOF")
    if not np.array_equal(merged["label_idx"].to_numpy(), merged["label_cnn"].to_numpy()):
        raise ValueError("Label mismatch between features and CNN OOF")

    y_true = merged["label_idx"].to_numpy()
    probs_lgbm = probs_lgbm[merged["idx_lgbm"].to_numpy()]
    probs_cnn = probs_cnn[merged["idx_cnn"].to_numpy()]
    features = features.loc[merged.index].reset_index(drop=True)

    if len(classes) != probs_lgbm.shape[1]:
        raise ValueError("Class count mismatch between features and OOF predictions")

    # Temperature scaling per model
    temp_lgbm = optimise_temperature(probs_lgbm, y_true)
    temp_cnn = optimise_temperature(probs_cnn, y_true)
    probs_lgbm_cal = temperature_scale(probs_lgbm, temp_lgbm)
    probs_cnn_cal = temperature_scale(probs_cnn, temp_cnn)

    # Do-no-harm blend
    alpha_base = compute_alpha(probs_lgbm_cal, probs_cnn_cal, y_true, args.alpha_max)
    print(f"Best alpha (capped at {args.alpha_max}): {alpha_base:.3f}")

    alpha_case = apply_gating(alpha_base, features)
    probs_blend = (1 - alpha_case[:, None]) * probs_lgbm_cal + alpha_case[:, None] * probs_cnn_cal

    # Physical vetting
    weights = {"v": 0.3, "oe": 0.2, "sec": 0.2, "dur": 0.15, "rho": 0.15}
    probs_phys = physical_vetting_probs(probs_blend, features, weights)

    # Metrics
    metrics_lgbm = compute_metrics(probs_lgbm_cal, y_true, classes)
    metrics_cnn = compute_metrics(probs_cnn_cal, y_true, classes)
    metrics_blend = compute_metrics(probs_blend, y_true, classes)
    metrics_phys = compute_metrics(probs_phys, y_true, classes)

    print("\n=== Metrics ===")
    print(f"LightGBM (calibrated) Macro-F1: {metrics_lgbm['macro_f1']:.3f} | LogLoss: {metrics_lgbm['log_loss']:.3f} | ECE: {metrics_lgbm['ece']:.3f}")
    print(f"CNN (calibrated)      Macro-F1: {metrics_cnn['macro_f1']:.3f} | LogLoss: {metrics_cnn['log_loss']:.3f} | ECE: {metrics_cnn['ece']:.3f}")
    print(f"Blend (alpha={alpha_base:.3f}) Macro-F1: {metrics_blend['macro_f1']:.3f} | LogLoss: {metrics_blend['log_loss']:.3f} | ECE: {metrics_blend['ece']:.3f}")
    print(f"Blend + vetting        Macro-F1: {metrics_phys['macro_f1']:.3f} | LogLoss: {metrics_phys['log_loss']:.3f} | ECE: {metrics_phys['ece']:.3f}")

    summary = {
        "classes": classes,
        "alpha_base": alpha_base,
        "temperature_lgbm": temp_lgbm,
        "temperature_cnn": temp_cnn,
        "metrics_lgbm": metrics_lgbm,
        "metrics_cnn": metrics_cnn,
        "metrics_blend": metrics_blend,
        "metrics_phys": metrics_phys,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, probs_phys=probs_phys.astype(np.float32), probs_blend=probs_blend.astype(np.float32), labels=y_true)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved blended probabilities to {args.output}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
