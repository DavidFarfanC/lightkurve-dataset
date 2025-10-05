"""Utilities for working with mission catalogs and metadata tables."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import pandas as pd


class CatalogColumns(TypedDict, total=False):
    target_id: str
    label: str
    mission: str
    period: str
    epoch: str
    duration: str


DEFAULT_COLUMN_MAP: CatalogColumns = {
    "target_id": "tic_id",
    "label": "disposition",
    "mission": "mission",
    "period": "orbital_period",
    "epoch": "transit_epoch",
    "duration": "transit_duration",
}


def load_metadata_catalog(path: Path | None = None, column_map: CatalogColumns | None = None) -> pd.DataFrame:
    """Load a catalog CSV and normalize its schema.

    The function expects, at minimum, columns identifying the target, label, and
    basic transit parameters (period, epoch, duration). Additional columns are
    carried through unchanged.
    """

    if path is None:
        path = Path("data/metadata/catalog.csv")

    if not path.exists():
        raise FileNotFoundError(
            f"Catalog file '{path}' not found. Provide it via PipelineConfig.metadata_path."
        )

    column_map = column_map or DEFAULT_COLUMN_MAP
    df = pd.read_csv(path)

    # Normalize column names to snake_case for convenience.
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    required_pairs = [
        (column_map["target_id"], "target_id"),
        (column_map["label"], "label"),
        (column_map["period"], "period"),
        (column_map["epoch"], "epoch"),
    ]
    missing: list[str] = []
    for original, canonical in required_pairs:
        if original in df.columns or canonical in df.columns:
            continue
        missing.append(original)
    if missing:
        raise ValueError(f"Catalog is missing required columns: {missing}")

    rename_map = {value: key for key, value in column_map.items() if value in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    return df


__all__ = ["load_metadata_catalog", "DEFAULT_COLUMN_MAP"]
