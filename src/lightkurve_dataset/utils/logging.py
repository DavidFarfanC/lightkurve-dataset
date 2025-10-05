"""Utility helpers for consistent logging across the project."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_dir: Path) -> None:
    """Set up a basic logging configuration that writes to file and stdout."""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


__all__ = ["configure_logging"]
