"""Helpers to retrieve light curves and catalogs using Lightkurve and NASA archives."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from .catalog import load_metadata_catalog
from ..config import DataFetchConfig

try:  # Optional dependency to keep import-time errors friendly
    import lightkurve as lk  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - import guard
    lk = None  # type: ignore

logger = logging.getLogger(__name__)


def ensure_lightkurve() -> None:
    """Raise a user-friendly error if Lightkurve is unavailable."""

    if lk is None:
        raise ModuleNotFoundError(
            "lightkurve package is required for data download. Install via 'pip install lightkurve'."
        )


def download_lightcurves(
    targets: Sequence[str],
    config: DataFetchConfig,
) -> list[Path]:
    """Download PDCSAP/SAP light curves for the provided targets.

    Parameters
    ----------
    targets:
        Iterable of TIC IDs (for TESS) or Kepler IDs.
    config:
        Download configuration.

    Returns
    -------
    list[Path]
        Paths to the downloaded FITS files.
    """

    ensure_lightkurve()

    download_dir = config.download_dir.expanduser()
    download_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files: list[Path] = []
    mission = config.mission

    for idx, target in enumerate(targets):
        if config.max_targets is not None and idx >= config.max_targets:
            logger.info("Reached max_targets=%s, stopping download", config.max_targets)
            break

        logger.info("Searching %s light curves for target %s", mission, target)
        try:
            search = lk.search_lightcurve(
                target,
                mission=mission,
                author=config.author,
                exptime=None,
            )
        except Exception as exc:  # pragma: no cover - upstream network errors
            logger.warning("Search failed for %s (%s): %s", target, mission, exc)
            continue

        if search is None or len(search) == 0:
            logger.info("No light curves found for target %s", target)
            continue

        try:
            lc_collection = search.download_all(  # type: ignore[union-attr]
                download_dir=str(download_dir),
                flux_column=f"{config.product}_FLUX",
                quality_bitmask=config.product,
                prefer_short_cadence=False,
                fetch=None if config.force_download else "default",
            )
        except Exception as exc:  # pragma: no cover - upstream network errors
            logger.warning("Download failed for %s (%s): %s", target, mission, exc)
            continue

        if lc_collection is None:
            logger.info("No files downloaded for target %s", target)
            continue

        for lc in lc_collection:
            path = Path(lc.meta.get("FILENAME", lc.meta.get("TARGETID", "")))
            final_path = download_dir / path.name if not path.is_absolute() else path
            if final_path.exists():
                downloaded_files.append(final_path)
            else:
                # Lightkurve returns LightCurve objects even when download=False.
                # We persist them here for reproducibility.
                filename = download_dir / f"{target}_{lc.meta.get('mission', mission)}.fits"
                lc.to_fits(path=str(filename), overwrite=True)
                downloaded_files.append(filename)

    logger.info("Downloaded %s files to %s", len(downloaded_files), download_dir)
    return downloaded_files


def load_targets_from_catalog(catalog_path: Path | None = None) -> list[str]:
    """Return the list of targets to download from the curated metadata catalog."""

    catalog = load_metadata_catalog(catalog_path)
    id_column = "tic_id" if "tic_id" in catalog.columns else "kepid"
    targets = catalog[id_column].astype(str).tolist()
    logger.info("Loaded %s targets from catalog", len(targets))
    return targets


__all__ = ["download_lightcurves", "load_targets_from_catalog"]
