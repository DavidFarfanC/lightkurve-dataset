"""Command-line entry point to prepare datasets for both hybrid branches."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from dataclasses import replace

from lightkurve_dataset.config import PipelineConfig
from lightkurve_dataset.data.catalog import load_metadata_catalog
from lightkurve_dataset.data.fetch import download_lightcurves
from lightkurve_dataset.datasets.cnn import build_cnn_dataset
from lightkurve_dataset.datasets.lightgbm import build_feature_dataset
from lightkurve_dataset.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def _attach_lightcurve_paths(metadata: pd.DataFrame, download_dir: Path) -> pd.DataFrame:
    files = list(download_dir.rglob("*.fits"))
    if not files:
        logger.warning("No FITS files found in %s", download_dir)
        metadata["lc_path"] = None
        return metadata

    def _find_path(row: pd.Series) -> str | None:
        target = str(row["target_id"])
        mission = str(row.get("mission", "")).lower()
        for path in files:
            name = path.name.lower()
            if target.lower() in name and (not mission or mission in name):
                return str(path)
        return None

    metadata = metadata.copy()
    metadata["lc_path"] = metadata.apply(_find_path, axis=1)
    missing = metadata["lc_path"].isna().sum()
    if missing:
        logger.warning("Could not find light curves for %s targets", missing)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download light curves before preprocessing",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        help="Limit the number of targets processed (applied per mission for quick experiments)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where processed datasets will be written",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Optional override for the metadata catalog path",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Continue even if some targets do not have a matching light-curve file",
    )
    args = parser.parse_args()

    config = PipelineConfig()
    configure_logging(config.logs_dir)

    metadata_path = args.catalog or config.metadata_path
    metadata = load_metadata_catalog(metadata_path)

    if args.max_targets is not None and args.max_targets > 0:
        metadata = (
            metadata.groupby("mission", group_keys=False)
            .head(args.max_targets)
            .reset_index(drop=True)
        )
        config.fetch.max_targets = args.max_targets

    if args.download:
        for mission, group in metadata.groupby("mission"):
            fetch_cfg = replace(config.fetch, mission=str(mission))
            targets = group["target_id"].astype(str).tolist()
            logger.info("Downloading %s targets for mission %s", len(targets), mission)
            download_lightcurves(targets, fetch_cfg)

    metadata = _attach_lightcurve_paths(metadata, config.fetch.download_dir)
    if metadata["lc_path"].isna().any():
        missing_ids = metadata.loc[metadata["lc_path"].isna(), "target_id"].tolist()
        if args.allow_missing:
            logger.warning("Dropping %s targets without light curves", len(missing_ids))
            metadata = metadata.dropna(subset=["lc_path"]).reset_index(drop=True)
        else:
            logger.error("Missing light-curve files for targets: %s", missing_ids[:10])
            raise SystemExit(1)

    feature_output = args.output_dir / "lightgbm" / "features.parquet"
    cnn_output_dir = args.output_dir / "cnn"

    build_feature_dataset(
        metadata=metadata,
        quality_cfg=config.quality,
        phase_cfg=config.phase,
        feature_cfg=config.features,
        output_path=feature_output,
    )

    build_cnn_dataset(
        metadata=metadata,
        quality_cfg=config.quality,
        phase_cfg=config.phase,
        cnn_cfg=config.cnn,
        output_dir=cnn_output_dir,
    )

    logger.info("Pipeline finished successfully")


if __name__ == "__main__":
    main()
