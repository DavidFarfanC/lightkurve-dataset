"""Extract physically motivated features from cleaned light curves."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..config import FeatureConfig

try:  # Optional dependency - BoxLeastSquares resides in astropy
    from astropy.timeseries import BoxLeastSquares  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional import guard
    try:
        from astropy.stats import BoxLeastSquares  # type: ignore
    except ModuleNotFoundError:
        BoxLeastSquares = None  # type: ignore

logger = logging.getLogger(__name__)

G_CONST = 6.67430e-11  # m^3 kg^-1 s^-2
SECONDS_PER_DAY = 86400.0
RADIUS_RATIO_EPS = 1e-6


@dataclass(slots=True)
class PhysicalFeatureResult:
    features: dict[str, float]
    diagnostics: dict[str, Any]



def _compute_depth_ppm(flux_in: np.ndarray, flux_out: np.ndarray) -> float:
    baseline = np.nanmedian(flux_out)
    depth = baseline - np.nanmedian(flux_in)
    return float(depth * 1e6)



def _compute_snr(depth_ppm: float, flux_out: np.ndarray, n_in: int) -> float:
    if n_in <= 0:
        return 0.0
    sigma = np.nanstd(flux_out)
    if sigma <= 0:
        return 0.0
    return float(depth_ppm / sigma * np.sqrt(n_in))



def _compute_vshape_ratio(duration_full: float, duration_flat: float) -> float:
    if duration_full <= 0:
        return 0.0
    duration_flat = max(duration_flat, 1e-6)
    return float(1.0 - duration_flat / duration_full)



def _transit_masks(phase: np.ndarray, duration: float, period: float) -> tuple[np.ndarray, np.ndarray]:
    half_width = duration / period / 2
    in_mask = np.abs(phase) <= half_width
    out_mask = np.abs(phase) >= min(0.5 * half_width + 0.01, 0.25)
    return in_mask, out_mask



def _odd_even_masks(phase: np.ndarray, period: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    half_width = duration / period / 2
    odd_mask = (phase >= -half_width) & (phase <= half_width)
    even_phase = (phase + 0.5) % 1 - 0.5
    even_mask = (even_phase >= -half_width) & (even_phase <= half_width)
    return odd_mask, even_mask



def _secondary_mask(phase: np.ndarray, period: float, duration: float, tolerance: float) -> np.ndarray:
    center = 0.5
    half_width = duration / period / 2
    return np.abs(((phase - center + 0.5) % 1) - 0.5) <= (half_width + tolerance)



def _compute_bls_metrics(time: np.ndarray, flux: np.ndarray, period: float, duration: float) -> dict[str, float]:
    if BoxLeastSquares is None:
        return {"bls_sde": np.nan, "bls_power": np.nan}

    try:
        bls = BoxLeastSquares(time, flux)
        durations = np.array([max(duration * 0.5, 0.01 * period), duration, min(duration * 1.5, period / 2)])
        periods = np.linspace(0.5 * period, 1.5 * period, 50)
        power = bls.power(periods, durations)
    except Exception as exc:  # pragma: no cover - astropy edge cases
        logger.warning("BLS computation failed: %s", exc)
        return {"bls_sde": np.nan, "bls_power": np.nan}

    max_power = np.nanmax(power.power)
    sde = (max_power - np.nanmean(power.power)) / np.nanstd(power.power)
    return {"bls_sde": float(sde), "bls_power": float(max_power)}



def _density_from_transit(period: float, duration: float, radius_ratio: float | None, impact_param: float | None) -> float:
    try:
        period_seconds = period * SECONDS_PER_DAY
        duration_seconds = duration * SECONDS_PER_DAY
        k = radius_ratio or 0.1
        b = impact_param or 0.0
        trig = np.sin(np.pi * duration_seconds / period_seconds)
        trig = np.clip(trig, RADIUS_RATIO_EPS, 1 - RADIUS_RATIO_EPS)
        term = np.sqrt(max((1 + k) ** 2 - b**2, RADIUS_RATIO_EPS))
        a_over_r = term / trig
        rho = 3 * np.pi / (G_CONST * period_seconds**2) * a_over_r**3
        return float(rho)
    except Exception:  # pragma: no cover - guard numerical instabilities
        return float("nan")



def extract_physical_features(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    phase: np.ndarray,
    period: float,
    duration: float,
    epoch: float,
    config: FeatureConfig,
    metadata: Mapping[str, Any] | None = None,
) -> PhysicalFeatureResult:
    """Compute a dictionary of physically informed summary statistics."""

    in_mask, out_mask = _transit_masks(phase, duration, period)
    if in_mask.sum() < 5:
        raise ValueError("Transit window insufficient for feature extraction")

    flux_in = flux[in_mask]
    flux_out = flux[out_mask]

    depth_ppm = _compute_depth_ppm(flux_in, flux_out)
    snr = _compute_snr(depth_ppm, flux_out, in_mask.sum())

    odd_mask, even_mask = _odd_even_masks(phase, period, duration)
    odd_depth = _compute_depth_ppm(flux[odd_mask], flux_out) if odd_mask.sum() > 0 else np.nan
    even_depth = _compute_depth_ppm(flux[even_mask], flux_out) if even_mask.sum() > 0 else np.nan
    odd_even_diff = odd_depth - even_depth if np.isfinite(odd_depth) and np.isfinite(even_depth) else np.nan

    secondary_mask = _secondary_mask(phase, period, duration, config.secondary_phase_tolerance)
    secondary_depth = _compute_depth_ppm(flux[secondary_mask], flux_out) if secondary_mask.sum() > 0 else np.nan

    duration_full = duration
    duration_flat = duration * 0.7  # placeholder until ingress/egress modeling is applied
    vshape = _compute_vshape_ratio(duration_full, duration_flat)

    if config.compute_bls:
        bls_metrics = _compute_bls_metrics(time, flux, period, duration)
    else:
        bls_metrics = {"bls_sde": np.nan, "bls_power": np.nan}

    rho_star = _density_from_transit(period, duration, metadata.get("rp_rs") if metadata else None, metadata.get("impact") if metadata else None)

    features = {
        "depth_ppm": depth_ppm,
        "snr_in_transit": snr,
        "odd_depth_ppm": float(odd_depth) if np.isfinite(odd_depth) else np.nan,
        "even_depth_ppm": float(even_depth) if np.isfinite(even_depth) else np.nan,
        "odd_even_diff_ppm": float(odd_even_diff) if np.isfinite(odd_even_diff) else np.nan,
        "secondary_depth_ppm": float(secondary_depth) if np.isfinite(secondary_depth) else np.nan,
        "vshape_ratio": vshape,
        "duration_days": duration,
        "period_days": period,
        "epoch_bjd": epoch,
        "rho_star_cgs": rho_star * 1e-3 if np.isfinite(rho_star) else np.nan,
        **bls_metrics,
    }

    diagnostics = {
        "in_samples": int(in_mask.sum()),
        "out_samples": int(out_mask.sum()),
    }

    return PhysicalFeatureResult(features=features, diagnostics=diagnostics)


__all__ = ["extract_physical_features", "PhysicalFeatureResult"]
