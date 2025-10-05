"""End-to-end preprocessing helpers for a single light curve."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import PhaseFoldConfig, QualityMaskConfig
from .cleaning import clean_lightcurve
from .phase import create_global_local_views, fold_phase

try:
    import lightkurve as lk  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - import guard
    lk = None  # type: ignore

logger = logging.getLogger(__name__)


class PreprocessingError(RuntimeError):
    """Raised when a light curve cannot be preprocessed."""



def load_lightcurve(path: Path) -> "lk.LightCurve":
    if lk is None:
        raise ModuleNotFoundError(
            "lightkurve must be installed to load FITS light curve files."
        )
    try:
        return lk.read(path)
    except Exception as exc:
        try:
            from astropy.io import fits  # type: ignore
        except ModuleNotFoundError as astropy_exc:  # pragma: no cover - optional dependency missing
            raise exc  # re-raise original error if astropy isn't available

        with fits.open(path) as hdul:  # type: ignore[attr-defined]
            data = hdul[1].data
            if data is None:
                raise exc
            names = list(getattr(data, "names", []) or [])
            time = data["TIME"].astype(float) if "TIME" in names else data.field(0).astype(float)
            flux = data["FLUX"].astype(float) if "FLUX" in names else data.field(1).astype(float)
            flux_err = (
                data["FLUX_ERR"].astype(float)
                if "FLUX_ERR" in names
                else None
            )

        lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)  # type: ignore[attr-defined]
        lc.meta["FILENAME"] = str(path)
        return lc



def preprocess_lightcurve(
    path: Path,
    quality_cfg: QualityMaskConfig,
    phase_cfg: PhaseFoldConfig,
    metadata: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Run the cleaning + phase folding steps for a single FITS file."""

    try:
        lc = load_lightcurve(path)
    except Exception as exc:  # pragma: no cover - I/O and parsing edge cases
        raise PreprocessingError(f"Failed to load light curve {path}") from exc

    time, flux, flux_err = clean_lightcurve(lc, quality_cfg)

    period = float(metadata["period"])
    epoch = float(metadata["epoch"])
    duration = float(metadata.get("duration", 0.1))

    phase = fold_phase(time, period, epoch)
    views = create_global_local_views(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=period,
        epoch=epoch,
        duration=duration,
        config=phase_cfg,
    )

    return {
        **views,
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "phase": phase,
    }


__all__ = ["preprocess_lightcurve", "PreprocessingError"]
