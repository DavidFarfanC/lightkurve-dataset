"""Dataset builder for the LightGBM (feature-based) branch."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import FeatureConfig, PhaseFoldConfig, QualityMaskConfig
from ..features.physical import extract_physical_features
from ..preprocessing.pipeline import preprocess_lightcurve

logger = logging.getLogger(__name__)


def build_feature_dataset(
    metadata: pd.DataFrame,
    quality_cfg: QualityMaskConfig,
    phase_cfg: PhaseFoldConfig,
    feature_cfg: FeatureConfig,
    output_path: Path,
) -> pd.DataFrame:
    """Generate a feature table ready for tree-based classification."""

    records: list[dict[str, float | str]] = []

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

        try:
            features = extract_physical_features(
                time=prepared["time"],
                flux=prepared["flux"],
                flux_err=prepared["flux_err"],
                phase=prepared["phase"],
                period=float(row["period"]),
                duration=float(row.get("duration", 0.1)),
                epoch=float(row["epoch"]),
                config=feature_cfg,
                metadata=row,
            )
        except Exception as exc:  # pragma: no cover - edge cases logged for debugging
            logger.warning("Skipping %s due to feature extraction error: %s", row["target_id"], exc)
            continue

        records.append(
            {
                "target_id": row["target_id"],
                "mission": row.get("mission", "unknown"),
                "label": row["label"],
                **features.features,
            }
        )

    feature_table = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_parquet(output_path, index=False)
    logger.info("Saved %s feature rows to %s", len(feature_table), output_path)
    return feature_table


__all__ = ["build_feature_dataset"]
