"""Generate synthetic light-curve FITS files for quick experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightkurve import LightCurve

RNG = np.random.default_rng(42)


def simulate_flux(time: np.ndarray, period: float, epoch: float, duration: float, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Create a simple transit-like signal with Gaussian noise."""

    phase = ((time - epoch + 0.5 * period) % period) / period - 0.5
    half_width = duration / period / 2

    base_depth = {
        "CP": 0.006,
        "PC": 0.003,
        "FP": 0.0,
    }.get(label, 0.002)

    depth = base_depth * RNG.uniform(0.6, 1.4)
    noise_level = 3.5e-4 if label == "CP" else 5e-4

    flux = np.ones_like(time)
    in_transit = np.abs(phase) <= half_width

    if label != "FP":
        shape = np.cos(phase[in_transit] * np.pi / half_width / 2) ** 2
        flux[in_transit] -= depth * shape
    else:
        flux += RNG.normal(0.0, depth or 0.001, size=len(flux)) * 0.2

    flux += RNG.normal(0.0, noise_level, size=len(flux))
    flux_err = np.full_like(flux, noise_level * 1.2)
    return flux, flux_err


def build_lightcurve(row: pd.Series, cadence: float, cycles: int) -> LightCurve:
    period = float(row["period"])
    epoch = float(row["epoch"])
    duration = max(float(row["duration"]), 0.05)

    span = cycles * period
    n_points = int(np.ceil(span / cadence))
    start = epoch - span / 2
    time = np.linspace(start, start + span, n_points)

    flux, flux_err = simulate_flux(time, period, epoch, duration, row["label"])
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    lc.meta.update(
        {
            "MISSION": row.get("mission", "Synthetic"),
            "TARGETID": row["target_id"],
            "PERIOD": period,
            "EPOCH": epoch,
            "DURATION": duration,
        }
    )
    return lc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/metadata/catalog.csv"))
    parser.add_argument("--per-label", type=int, default=50, help="Number of targets per label to simulate")
    parser.add_argument("--output", type=Path, default=Path("data/raw/mock"))
    parser.add_argument("--cycles", type=int, default=8, help="Number of orbital cycles to simulate")
    parser.add_argument("--cadence", type=float, default=0.020833, help="Cadence in days (default 30 min)")
    args = parser.parse_args()

    catalog = pd.read_csv(args.catalog)
    subset = catalog.groupby("label", group_keys=False).head(args.per_label).reset_index(drop=True)

    args.output.mkdir(parents=True, exist_ok=True)

    for row in subset.itertuples(index=False):
        mission = getattr(row, "mission", "Synthetic") or "Synthetic"
        mission_dir = args.output / mission.lower()
        mission_dir.mkdir(parents=True, exist_ok=True)

        lc = build_lightcurve(pd.Series(row._asdict()), args.cadence, args.cycles)
        filename = mission_dir / f"{row.target_id}_{mission.lower()}_mock.fits"
        lc.to_fits(path=filename, overwrite=True)

    print(f"Generated {len(subset)} synthetic light curves in {args.output}")


if __name__ == "__main__":
    main()
