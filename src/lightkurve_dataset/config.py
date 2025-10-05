"""Configuration objects for the hybrid exoplanet classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class DataFetchConfig:
    """Parameters that control how light curves and metadata are downloaded."""

    mission: Literal["Kepler", "K2", "TESS"] = "TESS"
    product: Literal["PDCSAP", "SAP"] = "PDCSAP"
    max_targets: int | None = None
    download_dir: Path = Path("data/raw")
    force_download: bool = False
    author: str | None = None  # e.g., "SPOC" for TESS PDCSAP


@dataclass(slots=True)
class QualityMaskConfig:
    """Masking and clipping configuration for raw light curves."""

    quality_bitmask: str | None = "default"  # See lightkurve.KeplerQualityFlags
    sigma_clip: float = 5.0
    window_length: int = 101  # Savitzky-Golay filter window
    polyorder: int = 3
    normalize: bool = True


@dataclass(slots=True)
class PhaseFoldConfig:
    """Options for phase folding and binning the time series."""

    global_bins: int = 2001
    local_bins: int = 201
    local_window_multiplier: float = 3.0  # multiples of transit duration
    oversample_factor: int = 5  # for interpolation prior to binning


@dataclass(slots=True)
class FeatureConfig:
    """Feature extraction parameters for the gradient boosting branch."""

    detrend_method: Literal["savgol", "spline"] = "savgol"
    odd_even_threshold: float = 3.0
    secondary_phase_tolerance: float = 0.05
    snr_window: int = 11
    compute_bls: bool = True


@dataclass(slots=True)
class CNNConfig:
    """Parameters governing the dataset tensors saved for the CNN branch."""

    include_error_channel: bool = True
    include_mask_channel: bool = True
    augment_phase_jitter: float = 0.02  # expressed as fraction of transit duration
    augment_mask_fraction: float = 0.05
    augment_red_noise_std: float = 300e-6


@dataclass(slots=True)
class PipelineConfig:
    """Top-level configuration for the end-to-end preprocessing pipeline."""

    fetch: DataFetchConfig = field(default_factory=DataFetchConfig)
    quality: QualityMaskConfig = field(default_factory=QualityMaskConfig)
    phase: PhaseFoldConfig = field(default_factory=PhaseFoldConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    metadata_path: Path = Path("data/metadata/catalog.csv")
    processed_dir: Path = Path("data/processed")
    logs_dir: Path = Path("logs")


__all__ = [
    "DataFetchConfig",
    "QualityMaskConfig",
    "PhaseFoldConfig",
    "FeatureConfig",
    "CNNConfig",
    "PipelineConfig",
]
