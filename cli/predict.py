"""CLI for generating calibrated predictions with physical vetting overlays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lightkurve_dataset.inference import InferenceEngine


def parse_series(value: object) -> np.ndarray | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, (list, tuple)):
                return np.asarray(parsed, dtype=float)
        except json.JSONDecodeError:
            pass
        separators = [",", " ", "|"]
        for sep in separators:
            parts = [p for p in text.replace("|", sep).split(sep) if p.strip()]
            if len(parts) >= 2:
                try:
                    return np.asarray([float(p) for p in parts], dtype=float)
                except ValueError:
                    continue
    return None


def save_phase_plot(phase: np.ndarray, flux: np.ndarray, path: Path, title: str) -> None:
    if len(phase) != len(flux) or len(phase) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 2.4))
    plt.plot(phase, flux, linewidth=1.0)
    plt.xlabel("Phase")
    plt.ylabel("Flux")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def load_curves(paths: List[Path]) -> Tuple[pd.DataFrame, List[Tuple[Path, int]]]:
    frames: List[pd.DataFrame] = []
    provenance: List[Tuple[Path, int]] = []
    for path in paths:
        df = pd.read_csv(path)
        df["__source_file__"] = str(path)
        df["__row_idx__"] = range(len(df))
        frames.append(df)
        provenance.extend([(path, i) for i in range(len(df))])
    if not frames:
        raise ValueError("No input curves loaded")
    merged = pd.concat(frames, ignore_index=True)
    return merged, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", nargs="+", help="CSV files or glob patterns with feature rows", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cal", type=Path, required=False)
    parser.add_argument("--vetting", type=Path, required=False)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--plots", type=Path, help="Optional directory for phase plots", required=False)
    args = parser.parse_args()

    paths: List[Path] = []
    for pattern in args.curves:
        expanded = sorted(Path().glob(pattern))
        if not expanded:
            raise FileNotFoundError(f"Pattern '{pattern}' did not match any files")
        paths.extend(expanded)

    df, provenance = load_curves(paths)
    engine = InferenceEngine(args.model, calibration_path=args.cal, vetting_config=args.vetting)
    predictions = engine.predict(df, include_shap=True)

    args.outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.plots or args.outdir
    plot_dir.mkdir(parents=True, exist_ok=True)

    for idx, pred in enumerate(predictions):
        row = df.iloc[idx]
        target_raw = str(pred.get("target_id") or row.get("target_id") or Path(provenance[idx][0]).stem)
        target = target_raw.replace("/", "_").replace("\\", "_")
        mission = pred.get("mission") or row.get("mission")

        phase_global = parse_series(row.get("phase_global"))
        flux_global = parse_series(row.get("flux_global"))
        phase_local = parse_series(row.get("phase_local"))
        flux_local = parse_series(row.get("flux_local"))

        plots = {"global": None, "local": None}
        if phase_global is not None and flux_global is not None:
            global_path = plot_dir / f"{target}_global.png"
            save_phase_plot(phase_global, flux_global, global_path, f"Global phase — {target}")
            plots["global"] = str(global_path)
        if phase_local is not None and flux_local is not None:
            local_path = plot_dir / f"{target}_local.png"
            save_phase_plot(phase_local, flux_local, local_path, f"Local phase — {target}")
            plots["local"] = str(local_path)

        output = {
            "target_id": target,
            "mission": None if mission is None else str(mission),
            "probabilities": pred["probabilities"],
            "flags": pred["flags"],
            "top_features": pred.get("top_features", []),
            "source": provenance[idx][0].as_posix(),
            "plots": plots,
        }
        out_path = args.outdir / f"pred_{target}.json"
        out_path.write_text(json.dumps(output, indent=2))
        print(f"Saved prediction to {out_path}")


if __name__ == "__main__":
    main()
