"""Construct a unified catalog (target_id, label, period, epoch, duration, mission)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/external")
OUTPUT_PATH = Path("data/metadata/catalog.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_kepler() -> pd.DataFrame:
    path = DATA_DIR / "koi_cumulative.csv"
    df = pd.read_csv(path, comment="#")
    cols = ["kepid", "koi_disposition", "koi_period", "koi_time0bk", "koi_duration"]
    df = df[cols].dropna()
    label_map = {
        "CONFIRMED": "CP",
        "CANDIDATE": "PC",
        "FALSE POSITIVE": "FP",
    }
    df = df[df["koi_disposition"].isin(label_map)]
    return pd.DataFrame(
        {
            "target_id": df["kepid"].astype(int).astype(str),
            "label": df["koi_disposition"].map(label_map),
            "period": df["koi_period"].astype(float),
            "epoch": df["koi_time0bk"].astype(float),
            "duration": (df["koi_duration"].astype(float) / 24.0),
            "mission": "Kepler",
        }
    )


def load_tess() -> pd.DataFrame:
    path = DATA_DIR / "toi.csv"
    df = pd.read_csv(path, comment="#")
    cols = ["tid", "tfopwg_disp", "pl_orbper", "pl_tranmid", "pl_trandurh"]
    df = df[cols].dropna()
    label_map = {
        "CP": "CP",
        "KP": "CP",
        "PC": "PC",
        "FP": "FP",
        "FA": "FP",
        "APC": "PC",
    }
    df = df[df["tfopwg_disp"].isin(label_map)]
    return pd.DataFrame(
        {
            "target_id": df["tid"].astype(int).astype(str),
            "label": df["tfopwg_disp"].map(label_map),
            "period": df["pl_orbper"].astype(float),
            "epoch": df["pl_tranmid"].astype(float),
            "duration": (df["pl_trandurh"].astype(float) / 24.0),
            "mission": "TESS",
        }
    )


def main() -> None:
    frames = [load_kepler(), load_tess()]
    catalog = pd.concat(frames, ignore_index=True)
    catalog = catalog.dropna()
    catalog = catalog.drop_duplicates(subset=["target_id", "mission"])
    catalog.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(catalog)} targets to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
