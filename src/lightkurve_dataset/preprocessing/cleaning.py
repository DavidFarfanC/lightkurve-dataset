"""Light-curve cleaning utilities shared by both modeling branches."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from ..config import QualityMaskConfig

try:  # Optional import guard for runtime availability
    import lightkurve as lk  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - import guard
    lk = None  # type: ignore

logger = logging.getLogger(__name__)


class LightCurveCleaningError(RuntimeError):
    """Raised when the light-curve cleaning pipeline fails irrecoverably."""



def ensure_lightkurve() -> None:
    if lk is None:
        raise ModuleNotFoundError(
            "lightkurve package is required for light curve preprocessing. Install via 'pip install lightkurve'."
        )



def standardize_lightcurve(lc: "lk.LightCurve") -> "lk.LightCurve":
    if not isinstance(lc, lk.LightCurve):
        raise TypeError(f"Expected a LightCurve, received {type(lc)!r}")
    return lc.remove_nans()



def clean_lightcurve(
    lc: "lk.LightCurve",
    config: QualityMaskConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply quality mask, detrending, and normalization to a LightCurve.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Clean time, flux, and flux error arrays ready for phase folding.
    """

    ensure_lightkurve()

    lc = standardize_lightcurve(lc)

    if config.quality_bitmask is not None and hasattr(lc, "remove_quality_flags"):
        try:
            lc = lc.remove_quality_flags(quality_bitmask=config.quality_bitmask)
        except Exception as exc:  # pragma: no cover - upstream API variability
            logger.warning("Quality masking failed, continuing without mask: %s", exc)

    cleaned = lc

    if config.normalize:
        cleaned = cleaned.normalize(unit="ppm")

    try:
        cleaned = cleaned.remove_outliers(sigma=config.sigma_clip)
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning("Sigma clipping failed (sigma=%s): %s", config.sigma_clip, exc)

    if config.window_length > 0:
        try:
            flattened = cleaned.flatten(
                window_length=config.window_length,
                polyorder=config.polyorder,
                return_trend=False,
            )
            cleaned = cleaned.copy()
            cleaned.flux = flattened.flux
            cleaned.flux_err = flattened.flux_err
        except Exception as exc:  # pragma: no cover - fallback path
            logger.warning("Flattening failed (window=%s): %s", config.window_length, exc)

    cleaned = cleaned.remove_nans()
    return cleaned.time.value, cleaned.flux.value, cleaned.flux_err.value


__all__ = ["clean_lightcurve", "LightCurveCleaningError"]
