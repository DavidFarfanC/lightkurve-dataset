"""Create a hold-out split by star ID for final evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np


def load_star_ids(
    parquet_path: Path | None,
    star_column: str,
    metadata_csv: Path | None,
) -> List[str]:
    ids: List[str] | None = None

    if parquet_path and parquet_path.exists():
        try:
            import pandas as pd  # type: ignore

            df = pd.read_parquet(parquet_path)
            if star_column not in df.columns:
                raise KeyError
            ids = df[star_column].astype(str).unique().tolist()
        except Exception:
            ids = None

    if ids is None and metadata_csv and metadata_csv.exists():
        with metadata_csv.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            if star_column not in reader.fieldnames:
                raise KeyError(f"Column '{star_column}' not found in metadata CSV {metadata_csv}")
            seen = set()
            ids = []
            for row in reader:
                sid = row[star_column]
                if sid not in seen:
                    ids.append(sid)
                    seen.add(sid)

    if ids is None:
        raise ValueError("Could not load star IDs from parquet or metadata CSV")
    return ids


def make_holdout(
    star_ids: Iterable[str],
    test_size: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    star_ids = list(dict.fromkeys(star_ids))  # preserve order, remove duplicates
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if len(star_ids) < 2:
        raise ValueError("Need at least two unique star IDs to create a hold-out split")

    rng = np.random.default_rng(seed)
    n_holdout = max(1, int(round(len(star_ids) * test_size)))
    holdout_ids = set(rng.choice(star_ids, size=n_holdout, replace=False))
    train_ids = [sid for sid in star_ids if sid not in holdout_ids]
    return train_ids, list(holdout_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Parquet file with features", required=False)
    parser.add_argument("--metadata-csv", type=Path, help="CSV file with star metadata", required=False)
    parser.add_argument("--star-id", type=str, default="target_id", help="Column identifying the star/object")
    parser.add_argument("--test-size", type=float, default=0.12, help="Fraction of stars for hold-out")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()

    star_ids = load_star_ids(args.input, args.star_id, args.metadata_csv)
    train_ids, holdout_ids = make_holdout(star_ids, args.test_size, args.seed)

    split = {
        "source": str(args.input) if args.input else None,
        "metadata_csv": str(args.metadata_csv) if args.metadata_csv else None,
        "star_column": args.star_id,
        "seed": args.seed,
        "test_size": args.test_size,
        "train_ids": train_ids,
        "holdout_ids": holdout_ids,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(split, indent=2))
    print(f"Hold-out created with {len(holdout_ids)} stars (train={len(train_ids)})")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
