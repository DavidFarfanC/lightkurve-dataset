"""Phase folding and binning utilities."""

from __future__ import annotations

import numpy as np

from ..config import PhaseFoldConfig


def fold_phase(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    """Return phase values in the range [-0.5, 0.5)."""

    phase = ((time - epoch + 0.5 * period) % period) / period - 0.5
    return phase


def bin_curve(phase: np.ndarray, values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Bin values onto an evenly spaced phase grid."""

    order = np.argsort(phase)
    sorted_phase = phase[order]
    sorted_values = values[order]

    grid = np.linspace(-0.5, 0.5, bins)
    binned = np.interp(grid, sorted_phase, sorted_values)
    return grid, binned


def create_global_local_views(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    epoch: float,
    duration: float,
    config: PhaseFoldConfig,
) -> dict[str, np.ndarray]:
    """Produce phase-folded global and local views of the transit."""

    phase = fold_phase(time, period, epoch)
    global_phase, global_flux = bin_curve(phase, flux, config.global_bins)
    _, global_err = bin_curve(phase, flux_err, config.global_bins)

    local_half_window = config.local_window_multiplier * duration / period / 2
    mask = np.abs(phase) <= local_half_window
    if mask.sum() < 10:
        mask = np.argsort(np.abs(phase))[: config.local_bins]

    local_phase_raw = phase[mask]
    local_flux_raw = flux[mask]
    local_err_raw = flux_err[mask]

    local_phase, local_flux = bin_curve(local_phase_raw, local_flux_raw, config.local_bins)
    _, local_err = bin_curve(local_phase_raw, local_err_raw, config.local_bins)

    return {
        "global_phase": global_phase,
        "global_flux": global_flux,
        "global_err": global_err,
        "local_phase": local_phase,
        "local_flux": local_flux,
        "local_err": local_err,
    }


__all__ = ["fold_phase", "create_global_local_views"]
