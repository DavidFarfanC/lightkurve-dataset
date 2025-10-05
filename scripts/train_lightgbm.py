"""LightGBM training toolkit with cross-validation, optional hold-out, and calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

EPS = 1e-12


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


def compute_class_weights(labels: NDArray[np.int_], n_classes: int) -> Dict[int, float]:
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    weights = counts.sum() / (counts + EPS)
    weights /= weights.mean()
    return {cls: weight for cls, weight in enumerate(weights)}


def mean_std(values: Iterable[float]) -> str:
    arr = np.array(list(values))
    return f"{arr.mean():.3f} ± {arr.std(ddof=1):.3f}"


def temperature_scale(probs: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    temperature = max(temperature, 1e-3)
    logp = np.log(np.clip(probs, EPS, 1.0)) / temperature
    logp -= logp.max(axis=1, keepdims=True)
    exp = np.exp(logp)
    return exp / exp.sum(axis=1, keepdims=True)


def optimise_temperature(probs: NDArray[np.float64], labels: NDArray[np.int_]) -> float:
    grid = np.linspace(0.5, 3.0, 51)
    best_t = 1.0
    best_loss = float("inf")
    for t in grid:
        scaled = temperature_scale(probs, t)
        loss = log_loss(labels, scaled, labels=list(range(probs.shape[1])))
        if loss < best_loss:
            best_loss = loss
            best_t = t
    return float(best_t)


def load_split(path: Path) -> Tuple[set[str], set[str]]:
    data = json.loads(path.read_text())
    train_ids = {str(sid) for sid in data.get("train_ids", [])}
    holdout_ids = {str(sid) for sid in data.get("holdout_ids", [])}
    if not train_ids:
        raise ValueError("Split JSON must contain 'train_ids'")
    return train_ids, holdout_ids


def train_lightgbm(
    df: pd.DataFrame,
    feature_cols: List[str],
    classes: List[str],
    class_to_idx: Dict[str, int],
    folds: int,
    repeats: int,
    learning_rate: float,
    num_leaves: int,
    estimators: int,
    rng_seed: int = 42,
) -> tuple[lgb.LGBMClassifier, Dict[str, float], NDArray[np.float64], NDArray[np.int_], List[float], List[float], List[float]]:
    X = df[feature_cols].to_numpy()
    y = df["label"].map(class_to_idx).to_numpy()
    groups = df["target_id"].astype(str).to_numpy()

    n_samples = len(df)
    n_classes = len(classes)
    prob_sum = np.zeros((n_samples, n_classes), dtype=float)
    prob_counts = np.zeros(n_samples, dtype=float)

    macro_f1_scores: List[float] = []
    log_losses: List[float] = []
    eces: List[float] = []

    for repeat, seed in enumerate(range(repeats), start=1):
        sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups), start=1):
            print(f"Repeat {repeat}/{repeats} - Fold {fold}/{folds}")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            class_weights = compute_class_weights(y_train, n_classes)
            clf = lgb.LGBMClassifier(
                boosting_type="gbdt",
                objective="multiclass",
                num_class=n_classes,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                n_estimators=estimators,
                feature_fraction=0.8,
                subsample=0.9,
                subsample_freq=1,
                reg_alpha=0.1,
                reg_lambda=0.1,
                class_weight=class_weights,
                random_state=seed,
            )
            clf.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="multi_logloss",
                callbacks=[lgb.early_stopping(40, verbose=False)],
            )

            probs = clf.predict_proba(X_val)
            preds = probs.argmax(axis=1)
            macro_f1 = f1_score(y_val, preds, average="macro")
            macro_f1_scores.append(macro_f1)
            log_losses.append(log_loss(y_val, probs, labels=list(range(n_classes))))
            eces.append(expected_calibration_error(probs, y_val))

            prob_sum[val_idx] += probs
            prob_counts[val_idx] += 1

            print(
                f"  Macro-F1: {macro_f1:.3f} | Log-loss: {log_losses[-1]:.3f} | ECE: {eces[-1]:.3f}"
            )

    if np.any(prob_counts == 0):
        raise RuntimeError("Some samples never received OOF predictions; check cross-validation setup.")

    oof_probs = prob_sum / prob_counts[:, None]

    class_weights = compute_class_weights(y, n_classes)
    final_clf = lgb.LGBMClassifier(
        boosting_type="gbdt",
        objective="multiclass",
        num_class=n_classes,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        n_estimators=estimators,
        feature_fraction=0.8,
        subsample=0.9,
        subsample_freq=1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight=class_weights,
        random_state=rng_seed,
    )
    final_clf.fit(X, y)

    return (
        final_clf,
        class_weights,
        oof_probs,
        y,
        macro_f1_scores,
        log_losses,
        eces,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("data/processed/lightgbm/features.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/lightgbm/lightgbm_mock_cv.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--estimators", type=int, default=400)
    parser.add_argument("--split-json", type=Path, help="Optional train/holdout split JSON")
    parser.add_argument(
        "--calibrate",
        choices=["none", "oof_temperature"],
        default="none",
        help="Calibration method applied to OOF probabilities",
    )
    parser.add_argument("--save-model", type=Path, help="Where to save the final booster")
    parser.add_argument("--save-cal", type=Path, help="Where to save calibration parameters (JSON)")
    parser.add_argument("--oof-output", type=Path, default=Path("data/processed/lightgbm/lightgbm_oof.npz"))
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    df["target_id"] = df["target_id"].astype(str)

    holdout_ids: set[str] = set()
    train_ids: Optional[set[str]] = None
    if args.split_json:
        train_ids, holdout_ids = load_split(args.split_json)
        df = df[df["target_id"].isin(train_ids)].reset_index(drop=True)
        print(f"Using split: {len(train_ids)} train stars / {len(holdout_ids)} hold-out stars")

    feature_cols = [col for col in df.columns if col not in {"target_id", "mission", "label"}]
    y_labels = df["label"].astype(str)

    classes = sorted(y_labels.unique())
    class_to_idx = {label: idx for idx, label in enumerate(classes)}

    (
        final_clf,
        class_weights,
        oof_probs,
        y_indices,
        macro_scores,
        losses,
        eces,
    ) = train_lightgbm(
        df,
        feature_cols,
        classes,
        class_to_idx,
        args.folds,
        args.repeats,
        args.learning_rate,
        args.num_leaves,
        args.estimators,
    )

    print("\n=== Cross-validated LightGBM metrics ===")
    print(f"Macro-F1: {mean_std(macro_scores)}")
    print(f"Log-loss: {mean_std(losses)}")
    print(f"ECE: {mean_std(eces)}")

    summary = {
        "classes": classes,
        "macro_f1_scores": macro_scores,
        "log_losses": losses,
        "eces": eces,
        "mean_macro_f1": float(np.mean(macro_scores)),
        "std_macro_f1": float(np.std(macro_scores, ddof=1)),
        "mean_log_loss": float(np.mean(losses)),
        "std_log_loss": float(np.std(losses, ddof=1)),
        "mean_ece": float(np.mean(eces)),
        "std_ece": float(np.std(eces, ddof=1)),
        "split_json": str(args.split_json) if args.split_json else None,
        "train_stars": int(len(train_ids)) if train_ids else int(df["target_id"].nunique()),
        "holdout_stars": int(len(holdout_ids)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"Saved metrics to {args.output}")

    oof_path = args.oof_output
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        oof_path,
        probs=oof_probs.astype(np.float32),
        labels=y_indices,
        target_id=df["target_id"].astype(str).to_numpy(),
        class_to_idx=np.array([class_to_idx[c] for c in classes]),
    )
    print(f"Saved OOF probabilities to {oof_path}")

    temperature: Optional[float] = None
    if args.calibrate == "oof_temperature":
        temperature = optimise_temperature(oof_probs, y_indices)
        print(f"Temperature scaling (OOF) → T={temperature:.3f}")

    if args.save_model:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        final_clf.booster_.save_model(str(args.save_model))  # type: ignore[union-attr]
        label_map_path = args.save_model.with_suffix(".labels.json")
        label_map_path.write_text(json.dumps(class_to_idx, indent=2))
        feature_path = args.save_model.with_suffix(".features.json")
        feature_path.write_text(json.dumps(feature_cols, indent=2))
        print(f"Saved final LightGBM booster to {args.save_model}")
        print(f"Saved label mapping to {label_map_path}")
        print(f"Saved feature list to {feature_path}")

    if temperature is not None and args.save_cal:
        args.save_cal.parent.mkdir(parents=True, exist_ok=True)
        args.save_cal.write_text(json.dumps({"method": "temperature", "value": temperature}, indent=2))
        print(f"Saved calibration parameters to {args.save_cal}")


if __name__ == "__main__":
    main()
