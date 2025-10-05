"""Dataset builder for the CNN (raw curve) branch."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CNNConfig, PhaseFoldConfig, QualityMaskConfig
from ..preprocessing.pipeline import preprocess_lightcurve

logger = logging.getLogger(__name__)


def _build_channel_stack(values: list[np.ndarray]) -> np.ndarray:
    return np.stack(values, axis=0).astype(np.float32)


def build_cnn_dataset(
    metadata: pd.DataFrame,
    quality_cfg: QualityMaskConfig,
    phase_cfg: PhaseFoldConfig,
    cnn_cfg: CNNConfig,
    output_dir: Path,
) -> dict[str, np.ndarray]:
    """Generate global/local view tensors suitable for 1-D CNN training."""

    global_tensors: list[np.ndarray] = []
    global_phase_list: list[np.ndarray] = []
    local_tensors: list[np.ndarray] = []
    local_phase_list: list[np.ndarray] = []
    labels: list[str] = []
    meta_records: list[dict[str, str]] = []

    for row in metadata.to_dict(orient="records"):
        path = Path(row["lc_path"])
        if not path.exists():
            logger.warning("Skipping missing light curve file: %s", path)
            continue

        prepared = preprocess_lightcurve(
            path=path,
            quality_cfg=quality_cfg,
            phase_cfg=phase_cfg,
            metadata=row,
        )

        global_channels = [prepared["global_flux"]]
        if cnn_cfg.include_error_channel:
            global_channels.append(prepared["global_err"])
        if cnn_cfg.include_mask_channel:
            global_channels.append(np.ones_like(prepared["global_flux"], dtype=float))

        local_channels = [prepared["local_flux"]]
        if cnn_cfg.include_error_channel:
            local_channels.append(prepared["local_err"])
        if cnn_cfg.include_mask_channel:
            local_channels.append(np.ones_like(prepared["local_flux"], dtype=float))

        global_tensors.append(_build_channel_stack(global_channels))
        local_tensors.append(_build_channel_stack(local_channels))
        global_phase_list.append(prepared["global_phase"].astype(np.float32))
        local_phase_list.append(prepared["local_phase"].astype(np.float32))

        labels.append(row["label"])
        meta_records.append(
            {
                "target_id": row["target_id"],
                "mission": row.get("mission", "unknown"),
                "label": row["label"],
            }
        )

    if not global_tensors:
        raise RuntimeError("No light curves processed for CNN dataset")

    global_array = np.stack(global_tensors, axis=0)
    local_array = np.stack(local_tensors, axis=0)
    labels_array = np.array(labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "cnn_dataset.npz",
        global_inputs=global_array,
        local_inputs=local_array,
        global_phase=np.stack(global_phase_list, axis=0),
        local_phase=np.stack(local_phase_list, axis=0),
        labels=labels_array,
    )

    pd.DataFrame(meta_records).to_csv(output_dir / "cnn_metadata.csv", index=False)
    (output_dir / "cnn_config.json").write_text(json.dumps(asdict(cnn_cfg), indent=2))

    logger.info(
        "Saved CNN tensors: %s global, %s local to %s",
        global_array.shape,
        local_array.shape,
        output_dir / "cnn_dataset.npz",
    )

    return {
        "global_inputs": global_array,
        "local_inputs": local_array,
        "labels": labels_array,
    }


__all__ = ["build_cnn_dataset"]
