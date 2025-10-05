"""Cross-validated training of the dual-branch CNN on phase-folded light curves.

Key features implemented per hackathon recommendations:
- StratifiedKFold validation (proxy for StratifiedGroupKFold when groups are unique).
- Focal loss with class weights to emphasise minority classes.
- Lightweight architecture (two Conv1d blocks) with moderate dropout.
- On-the-fly augmentations (gaussian noise + random masking) for the training fold.
- Early stopping governed by macro-F1 instead of accuracy.
- Aggregated metrics (mean ± std) for macro-F1, log-loss, and ECE across folds/repeats.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

torch.set_num_threads(1)
from sklearn.metrics import classification_report, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def compute_class_weights(labels: Iterable[int], n_classes: int) -> torch.Tensor:
    counts = np.bincount(list(labels), minlength=n_classes).astype(float)
    weights = counts.sum() / (counts + 1e-6)
    weights /= weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


def expected_calibration_error(probs: NDArray[np.float64], targets: NDArray[np.int_], bins: int = 10) -> float:
    """Compute ECE using equally spaced probability bins."""

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


# ---------------------------------------------------------------------------
# Dataset with augmentations
# ---------------------------------------------------------------------------


class PhaseCurveDataset(Dataset):
    def __init__(
        self,
        global_inputs: torch.Tensor,
        local_inputs: torch.Tensor,
        labels: torch.Tensor,
        indices: np.ndarray,
        augment: bool = False,
    ) -> None:
        self.global_data = global_inputs[indices]
        self.local_data = local_inputs[indices]
        self.labels = labels[indices]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = self.global_data[idx]
        l = self.local_data[idx]
        y = self.labels[idx]

        if self.augment:
            g = self._augment(g)
            l = self._augment(l)

        return g, l, y

    @staticmethod
    def _augment(signal: torch.Tensor) -> torch.Tensor:
        # Gaussian noise
        noise = torch.randn_like(signal) * 0.0005
        augmented = signal + noise
        # Random channel masking (drop 5% of points)
        mask = torch.rand(signal.shape[-1], device=signal.device) > 0.05
        augmented = augmented * mask
        return augmented


# ---------------------------------------------------------------------------
# Model + Focal loss
# ---------------------------------------------------------------------------


@dataclass
class CNNConfig:
    global_channels: int
    local_channels: int
    global_length: int
    local_length: int
    n_classes: int
    dropout: float = 0.4


class DualBranchCNN(nn.Module):
    def __init__(self, config: CNNConfig) -> None:
        super().__init__()

        self.global_branch = nn.Sequential(
            nn.Conv1d(config.global_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )

        self.local_branch = nn.Sequential(
            nn.Conv1d(config.local_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )

        with torch.no_grad():
            g_feat = self.global_branch(torch.zeros(1, config.global_channels, config.global_length))
            l_feat = self.local_branch(torch.zeros(1, config.local_channels, config.local_length))
            concat_dim = g_feat.shape[1] + l_feat.shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.n_classes),
        )

    def forward(self, global_input: torch.Tensor, local_input: torch.Tensor) -> torch.Tensor:
        g = self.global_branch(global_input)
        l = self.local_branch(local_input)
        concat = torch.cat([g, l], dim=1)
        return self.classifier(concat)


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=logits.shape[1]).float()
        alpha = self.alpha.to(logits.device)
        weight = (1 - probs) ** self.gamma
        focal = -alpha * weight * log_probs
        loss = (targets_one_hot * focal).sum(dim=1)
        return loss.mean()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for global_x, local_x, y in loader:
        global_x, local_x, y = global_x.to(device), local_x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(global_x, local_x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * y.size(0)
    return running_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, NDArray[np.int_], NDArray[np.float64]]:
    model.eval()
    loss_total = 0.0
    logits_list: List[NDArray[np.float64]] = []
    labels_list: List[NDArray[np.int_]] = []
    with torch.no_grad():
        for global_x, local_x, y in loader:
            global_x, local_x, y = global_x.to(device), local_x.to(device), y.to(device)
            logits = model(global_x, local_x)
            loss = criterion(logits, y)
            loss_total += loss.item() * y.size(0)
            logits_list.append(logits.softmax(dim=1).cpu().numpy())
            labels_list.append(y.cpu().numpy())
    probs = np.concatenate(logits_list)
    labels = np.concatenate(labels_list)
    return loss_total / len(loader.dataset), labels, probs


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/cnn/cnn_dataset.npz"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/cnn/cnn_mock.pt"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=2.0)
    args = parser.parse_args()

    pack = np.load(args.dataset)
    global_inputs = torch.from_numpy(pack["global_inputs"]).float()
    local_inputs = torch.from_numpy(pack["local_inputs"]).float()
    labels_raw = pack["labels"]

    metadata_path = args.dataset.with_name("cnn_metadata.csv")
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        target_ids = metadata.get("target_id", metadata.index).astype(str).to_numpy()
    else:
        target_ids = np.arange(len(labels_raw)).astype(str)

    if len(target_ids) != len(labels_raw):
        raise ValueError("Metadata target_id length does not match dataset labels")

    classes = sorted(set(labels_raw.tolist()))
    class_to_idx = {c: idx for idx, c in enumerate(classes)}
    y_indices = np.array([class_to_idx[c] for c in labels_raw], dtype=np.int64)
    y_tensor = torch.from_numpy(y_indices)

    config = CNNConfig(
        global_channels=global_inputs.shape[1],
        local_channels=local_inputs.shape[1],
        global_length=global_inputs.shape[2],
        local_length=local_inputs.shape[2],
        n_classes=len(classes),
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    n_samples = len(y_indices)
    n_classes = len(classes)
    prob_sum = np.zeros((n_samples, n_classes), dtype=float)
    prob_counts = np.zeros(n_samples, dtype=float)

    macro_f1_scores: List[float] = []
    log_losses: List[float] = []
    eces: List[float] = []
    reports: List[Dict[str, Any]] = []

    skf_seeds = range(args.repeats)
    for repeat, seed in enumerate(skf_seeds, start=1):
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros_like(y_indices), y_indices), start=1):
            print(f"Repeat {repeat}/{args.repeats} - Fold {fold}/{args.folds}")

            train_dataset = PhaseCurveDataset(global_inputs, local_inputs, y_tensor, train_idx, augment=True)
            val_dataset = PhaseCurveDataset(global_inputs, local_inputs, y_tensor, val_idx, augment=False)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

            class_weights = compute_class_weights(y_indices[train_idx], config.n_classes)
            criterion = FocalLoss(alpha=class_weights, gamma=args.gamma)
            model = DualBranchCNN(config).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

            best_state = None
            best_f1 = -np.inf
            epochs_without_improvement = 0

            for epoch in range(1, args.epochs + 1):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, y_true, probs = evaluate(model, val_loader, criterion, device)
                preds = probs.argmax(axis=1)
                macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)

                print(
                    f"  Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | macro_f1={macro_f1:.3f}"
                )

                if macro_f1 > best_f1 + 1e-4:
                    best_f1 = macro_f1
                    best_state = model.state_dict()
                    best_metrics = (val_loss, y_true, probs)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= args.patience:
                        print("  Early stopping triggered")
                        break

            assert best_state is not None
            model.load_state_dict(best_state)
            val_loss, y_true, probs = best_metrics
            preds = probs.argmax(axis=1)

            macro_f1_scores.append(best_f1)
            log_losses.append(log_loss(y_true, probs, labels=list(range(config.n_classes))))
            eces.append(expected_calibration_error(probs, y_true))
            report = classification_report(
                y_true,
                preds,
                target_names=classes,
                zero_division=0,
                output_dict=True,
            )
            reports.append(report)

            prob_sum[val_idx] += probs
            prob_counts[val_idx] += 1

    def mean_std(values: List[float]) -> str:
        arr = np.array(values)
        return f"{arr.mean():.3f} ± {arr.std(ddof=1):.3f}"

    print("\n=== Cross-validated metrics ===")
    print(f"Macro-F1: {mean_std(macro_f1_scores)}")
    print(f"Log-loss: {mean_std(log_losses)}")
    print(f"ECE: {mean_std(eces)}")

    summary = {
        "classes": classes,
        "macro_f1_scores": macro_f1_scores,
        "log_losses": log_losses,
        "eces": eces,
        "mean_macro_f1": np.mean(macro_f1_scores).item(),
        "std_macro_f1": np.std(macro_f1_scores, ddof=1).item(),
        "mean_log_loss": np.mean(log_losses).item(),
        "std_log_loss": np.std(log_losses, ddof=1).item(),
        "mean_ece": np.mean(eces).item(),
        "std_ece": np.std(eces, ddof=1).item(),
        "per_fold_reports": reports,
    }

    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved metrics to {metrics_path}")

    if np.any(prob_counts == 0):
        raise RuntimeError("Some samples never received OOF predictions; check K-fold configuration.")

    oof_probs = prob_sum / prob_counts[:, None]
    oof_path = Path("data/processed/cnn/cnn_oof.npz")
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        oof_path,
        probs=oof_probs.astype(np.float32),
        labels=y_indices,
        target_id=target_ids,
        class_to_idx=np.array([class_to_idx[c] for c in classes]),
    )
    print(f"Saved OOF probabilities to {oof_path}")

    # Train final model on full dataset (no augmentation) for deployment
    final_dataset = PhaseCurveDataset(
        global_inputs,
        local_inputs,
        y_tensor,
        np.arange(len(y_tensor)),
        augment=True,
    )
    final_loader = DataLoader(final_dataset, batch_size=args.batch_size, shuffle=True)
    final_class_weights = compute_class_weights(y_indices, config.n_classes)
    final_criterion = FocalLoss(alpha=final_class_weights, gamma=args.gamma)
    final_model = DualBranchCNN(config).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(final_model, final_loader, final_criterion, final_optimizer, device)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Final model epoch {epoch:02d} | train_loss={train_loss:.4f}")

    torch.save(
        {
            "state_dict": final_model.state_dict(),
            "config": config.__dict__,
            "classes": classes,
        },
        args.output,
    )
    label_map_path = args.output.with_suffix(".labels.json")
    label_map_path.write_text(json.dumps(class_to_idx, indent=2))
    print(f"Saved final CNN weights to {args.output}")
    print(f"Saved label mapping to {label_map_path}")


if __name__ == "__main__":
    main()
