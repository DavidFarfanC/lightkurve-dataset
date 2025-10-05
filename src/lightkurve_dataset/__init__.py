"""Core package for the exoplanet hybrid classification pipeline."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lightkurve-dataset")
except PackageNotFoundError:  # pragma: no cover - metadata only available when installed
    __version__ = "0.0.0"

__all__ = ["__version__"]
