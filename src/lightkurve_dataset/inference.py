"""Rich inference utilities powering the hybrid exoplanet API and CLI."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import FeatureConfig, PhaseFoldConfig, PipelineConfig, QualityMaskConfig
from .features.physical import PhysicalFeatureResult, extract_physical_features
from .preprocessing.cleaning import clean_lightcurve, ensure_lightkurve
from .preprocessing.phase import create_global_local_views, fold_phase

try:  # Optional dependency; inference insists on lightkurve for preprocessing
    import lightkurve as lk  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    lk = None  # type: ignore

try:  # Optional dependency for YAML configs
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

EPS = 1e-12
DEFAULT_MEDIA_DIR = Path("media")
DEFAULT_REPORTS_PATH = Path("reports/holdout_metrics.json")
DEFAULT_RELIABILITY_PATH = Path("data/processed/lightgbm/lightgbm_oof.npz")
DEFAULT_PLOTS_SUBDIR = "plots"
DEFAULT_MPL_DIR = ".mpl-cache"


@dataclass(slots=True)
class VettingConfig:
    weights: Dict[str, float]
    thresholds: Dict[str, float]


@dataclass(slots=True)
class ReliabilityCurve:
    bin_edges: np.ndarray
    accuracy: np.ndarray
    confidence: np.ndarray
    counts: np.ndarray


@dataclass(slots=True)
class PhaseProducts:
    time: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray
    phase: np.ndarray
    global_phase: np.ndarray
    global_flux: np.ndarray
    global_err: np.ndarray
    local_phase: np.ndarray
    local_flux: np.ndarray
    local_err: np.ndarray


@dataclass(slots=True)
class PhysicsFit:
    params: Dict[str, float]
    derived: Dict[str, float]
    goodness: Dict[str, float]
    odd_even: float
    secondary: Dict[str, float]
    status: str


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if path.exists():
        if yaml is None:
            raise ImportError("PyYAML is required to read vetting config")
        return yaml.safe_load(path.read_text()) or {}
    return {}


def _entropy(probs: Sequence[float]) -> float:
    arr = np.clip(np.asarray(probs, dtype=float), EPS, 1.0)
    return float(-np.sum(arr * np.log2(arr)))


def _confidence_metrics(probs: Dict[str, float]) -> Dict[str, float]:
    values = np.array(list(probs.values()), dtype=float)
    entropy = _entropy(values)
    if len(values) > 1:
        normalized_entropy = entropy / math.log2(len(values))
    else:
        normalized_entropy = 0.0
    margin = float(values.max() - np.partition(values, -2)[-2]) if len(values) > 1 else 1.0
    confidence_index = float(max(0.0, min(1.0, 1.0 - normalized_entropy)))
    return {
        "entropy": float(entropy),
        "margin": margin,
        "confidence_index": confidence_index,
    }


def _safe_list(array: np.ndarray, max_len: int = 2000) -> List[float]:
    if len(array) > max_len:
        idx = np.linspace(0, len(array) - 1, max_len).astype(int)
        array = array[idx]
    return array.astype(float).tolist()


def _timestamp() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


class InferenceEngine:
    """Single entry point for rich model inference and diagnostics."""

    def __init__(
        self,
        model_path: Path,
        calibration_path: Optional[Path] = None,
        vetting_config: Optional[Path] = None,
        metrics_path: Optional[Path] = DEFAULT_REPORTS_PATH,
        reliability_reference: Optional[Path] = DEFAULT_RELIABILITY_PATH,
        media_dir: Optional[Path] = None,
        pipeline_config: Optional[PipelineConfig] = None,
    ) -> None:
        self.model_path = model_path
        self.booster = lgb.Booster(model_file=str(model_path))

        label_map_path = model_path.with_suffix(".labels.json")
        label_map = _load_json(label_map_path)
        self.classes: List[str] = sorted(label_map, key=label_map.get)
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}

        features_path = model_path.with_suffix(".features.json")
        if not features_path.exists():
            raise FileNotFoundError(f"Feature list not found at {features_path}")
        self.feature_cols: List[str] = _load_json(features_path)

        self.temperature: float | None = None
        if calibration_path and Path(calibration_path).exists():
            calib = _load_json(Path(calibration_path))
            if calib.get("method") == "temperature":
                self.temperature = float(calib.get("value", 1.0))

        vetting_data = _load_yaml(Path(vetting_config) if vetting_config else None)
        self.vetting = VettingConfig(
            weights=vetting_data.get(
                "weights",
                {"v": 0.3, "oe": 0.2, "sec": 0.2, "dur": 0.15, "rho": 0.15},
            ),
            thresholds=vetting_data.get(
                "thresholds",
                {
                    "vshape_ratio": 0.6,
                    "odd_even_diff_ppm": 300.0,
                    "secondary_depth_ppm": 200.0,
                    "duration_ratio": 4.0,
                    "rho_min": 0.1,
                    "rho_max": 5.0,
                },
            ),
        )

        self.pipeline_config = pipeline_config or PipelineConfig()
        self.quality_cfg: QualityMaskConfig = self.pipeline_config.quality
        self.phase_cfg: PhaseFoldConfig = self.pipeline_config.phase
        self.feature_cfg: FeatureConfig = self.pipeline_config.features

        self.media_dir = Path(media_dir) if media_dir else DEFAULT_MEDIA_DIR
        self.plots_dir = self.media_dir / DEFAULT_PLOTS_SUBDIR
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(self.media_dir / DEFAULT_MPL_DIR))

        self.metrics_summary: Dict[str, Any] = {}
        metrics_path = Path(metrics_path) if metrics_path else None
        if metrics_path and metrics_path.exists():
            try:
                self.metrics_summary = _load_json(metrics_path)
            except json.JSONDecodeError:
                self.metrics_summary = {}

        self.reliability_curve: Optional[ReliabilityCurve] = None
        rel_path = Path(reliability_reference) if reliability_reference else None
        if rel_path and rel_path.exists():
            self.reliability_curve = self._load_reliability_curve(rel_path)

        self.model_hash = self._compute_model_hash()
        self.trained_at = self._model_timestamp()
        self.decision_thresholds = {"cp_high": 0.90, "cp_thresh": 0.50}
        self.version_info = {
            "model_hash": self.model_hash,
            "trained_at": self.trained_at,
            "features": self.feature_cols,
            "holdout_metrics": self._summary_metrics_view(),
            "decision_thresholds": self.decision_thresholds,
        }

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def set_decision_thresholds(self, cp_high: float, cp_thresh: float) -> None:
        self.decision_thresholds = {"cp_high": float(cp_high), "cp_thresh": float(cp_thresh)}
        self.version_info["decision_thresholds"] = self.decision_thresholds

    def get_model_info(self) -> Dict[str, Any]:
        return dict(self.version_info)

    # ------------------------------------------------------------------
    # Core prediction utilities
    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame, include_shap: bool = True) -> List[Dict[str, Any]]:
        meta_cols = {"target_id", "mission"}
        df = df.copy()
        for col in meta_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

        features = self._prepare_dataframe(df)
        X = features.to_numpy()

        raw_probs = self.booster.predict(X)
        if raw_probs.ndim == 1:
            raw_probs = np.vstack([1 - raw_probs, raw_probs]).T

        calibrated = self._temperature_scale(raw_probs)
        adjusted, vetting_details = self._apply_vetting(calibrated, df)

        shap_values = self._predict_shap(X) if include_shap else None

        results: List[Dict[str, Any]] = []
        for idx, row in df.iterrows():
            sample_probs = adjusted[idx]
            prob_map = {cls: float(sample_probs[self.class_to_idx[cls]]) for cls in self.classes}
            raw_map = {cls: float(calibrated[idx, self.class_to_idx[cls]]) for cls in self.classes}
            vet_detail = vetting_details[idx]
            entry: Dict[str, Any] = {
                "target_id": row.get("target_id", f"obj_{idx}"),
                "mission": row.get("mission", None),
                "probabilities": prob_map,
                "probabilities_raw": raw_map,
                "flags": vet_detail["flags"],
                "vetting": vet_detail,
            }
            if include_shap and shap_values is not None:
                shap_row = shap_values[idx]
                contributions = list(zip(self.feature_cols, shap_row))
                contributions.sort(key=lambda item: abs(item[1]), reverse=True)
                entry["top_features"] = [
                    {"feature": name, "shap": float(value)} for name, value in contributions[:5]
                ]
            results.append(entry)
        return results

    def infer_one(
        self,
        star_id: Optional[str],
        mission: Optional[str],
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray],
        period: Optional[float],
        epoch: Optional[float],
        duration: Optional[float] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        options = options or {}
        return_plots = bool(options.get("return_plots", True))
        return_plot_data = bool(options.get("return_plot_data", True))

        if period is None or epoch is None:
            raise ValueError("period and t0/epoch are required for inference")

        clean_time, clean_flux, clean_err = self._filter_raw_series(
            np.asarray(time, dtype=float),
            np.asarray(flux, dtype=float),
            np.asarray(flux_err, dtype=float) if flux_err is not None else None,
        )
        preprocessing_stats = {
            "n_samples_raw": int(len(time)),
            "n_samples_clean": int(len(clean_time)),
            "dropped": int(len(time) - len(clean_time)),
        }

        meta_duration = duration if duration is not None else float(period) * 0.1
        phase_vals = fold_phase(clean_time, float(period), float(epoch))
        views = create_global_local_views(
            time=clean_time,
            flux=clean_flux,
            flux_err=clean_err,
            period=float(period),
            epoch=float(epoch),
            duration=meta_duration,
            config=self.phase_cfg,
        )

        phase_products = PhaseProducts(
            time=clean_time,
            flux=clean_flux,
            flux_err=clean_err,
            phase=phase_vals,
            global_phase=views["global_phase"],
            global_flux=views["global_flux"],
            global_err=views["global_err"],
            local_phase=views["local_phase"],
            local_flux=views["local_flux"],
            local_err=views["local_err"],
        )

        feature_result = self._compute_features(
            phase_products,
            period=float(period),
            epoch=float(epoch),
            duration=meta_duration,
            metadata={"target_id": star_id or "anonymous"},
        )

        row = feature_result.features
        row.update({
            "target_id": star_id or f"obj_{uuid.uuid4().hex[:8]}",
            "mission": mission or "unknown",
        })

        predictions = self.predict(pd.DataFrame([row]), include_shap=True)[0]
        probs = predictions["probabilities"]
        decision = self._decision_from_probs(probs)
        physics = self._approximate_physics_fit(feature_result, float(period), meta_duration)

        plots: Dict[str, str] = {}
        plot_data: Dict[str, Any] = {}
        if return_plots or return_plot_data:
            plots, plot_data = self._generate_plot_artifacts(
                star_id=row["target_id"],
                mission=row["mission"],
                phase_products=phase_products,
                probabilities=probs,
                physics=physics,
                return_plots=return_plots,
                return_plot_data=return_plot_data,
            )

        response = {
            "star_id": row["target_id"],
            "mission": row["mission"],
            "version": self.get_model_info(),
            "probs": probs,
            "prediction": max(probs, key=probs.get),
            "decision": {**decision, "confidence": _confidence_metrics(probs)},
            "physics_fit": {
                "model": "MandelAgol2002",
                "params": physics.params,
                "derived": physics.derived,
                "goodness": physics.goodness,
                "odd_even": physics.odd_even,
                "secondary": physics.secondary,
                "status": physics.status,
            },
            "vetting": {
                "enabled": True,
                "penalties": predictions["vetting"].get("penalties", {}),
                "flags": [name for name, flag in predictions["flags"].items() if flag],
                "post_penalty_probs": predictions["probabilities"],
                "penalty_total": predictions["vetting"].get("penalty_total", 0.0),
            },
            "explainability": {"top_shap": predictions.get("top_features", [])},
            "preprocessing": preprocessing_stats,
        }

        if return_plots:
            response["plots"] = plots
        if return_plot_data:
            response["plot_data"] = plot_data

        return response

    def infer_batch(
        self,
        batch: Iterable[Dict[str, Any]],
        options: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for item in batch:
            results.append(
                self.infer_one(
                    star_id=item.get("star_id"),
                    mission=item.get("mission"),
                    time=np.asarray(item["time"], dtype=float),
                    flux=np.asarray(item["flux"], dtype=float),
                    flux_err=np.asarray(item.get("flux_err"), dtype=float) if item.get("flux_err") is not None else None,
                    period=item.get("period"),
                    epoch=item.get("t0") or item.get("epoch"),
                    duration=item.get("duration"),
                    options=options or item.get("options"),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_model_hash(self) -> str:
        digest = hashlib.sha1(self.model_path.read_bytes()).hexdigest()
        return digest[:8]

    def _model_timestamp(self) -> str:
        try:
            ts = dt.datetime.utcfromtimestamp(self.model_path.stat().st_mtime)
        except FileNotFoundError:
            ts = dt.datetime.utcnow()
        return ts.replace(microsecond=0).isoformat() + "Z"

    def _summary_metrics_view(self) -> Dict[str, float]:
        multicl = self.metrics_summary.get("metrics_multiclass", {})
        return {
            "macro_f1": float(multicl.get("macro_f1", float("nan"))),
            "logloss": float(multicl.get("log_loss", float("nan"))),
            "ece": float(multicl.get("ece", float("nan"))),
            "brier": float(multicl.get("brier", float("nan"))),
        }

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.feature_cols:
            if col not in out.columns:
                out[col] = 0.0
        return out[self.feature_cols]

    def _temperature_scale(self, probs: np.ndarray) -> np.ndarray:
        if self.temperature is None:
            return probs
        logp = np.log(np.clip(probs, EPS, 1.0)) / self.temperature
        return softmax(logp)

    def _filter_raw_series(
        self,
        time: np.ndarray,
        flux: np.ndarray,
        flux_err: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ensure_lightkurve()
        if flux_err is None:
            flux_err = np.zeros_like(flux)
        lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)
        clean_time, clean_flux, clean_err = clean_lightcurve(lc, self.quality_cfg)
        return clean_time, clean_flux, clean_err

    def _compute_flags(self, row: pd.Series) -> Tuple[Dict[str, bool], Dict[str, float]]:
        w = self.vetting.weights
        t = self.vetting.thresholds

        vshape = float(row.get("vshape_ratio", np.nan))
        odd_even = abs(float(row.get("odd_even_diff_ppm", np.nan)))
        secondary = abs(float(row.get("secondary_depth_ppm", np.nan)))
        duration = float(row.get("duration_days", np.nan))
        period = float(row.get("period_days", np.nan))
        rho = float(row.get("rho_star_cgs", np.nan))

        flag_v = np.nan_to_num(vshape) > t.get("vshape_ratio", 0.6)
        flag_oe = np.nan_to_num(odd_even) > t.get("odd_even_diff_ppm", 300.0)
        flag_sec = np.nan_to_num(secondary) > t.get("secondary_depth_ppm", 200.0)
        ratio_den = max(t.get("duration_ratio", 4.0), EPS)
        flag_dur = np.nan_to_num(duration) > np.nan_to_num(period) / ratio_den
        flag_rho = (~np.isfinite(rho)) or (rho < t.get("rho_min", 0.1)) or (rho > t.get("rho_max", 5.0))

        flags_bool = {
            "v_shape": bool(flag_v),
            "odd_even": bool(flag_oe),
            "secondary": bool(flag_sec),
            "duration": bool(flag_dur),
            "rho": bool(flag_rho),
        }
        penalties = {
            "v_shape": float(w.get("v", 0.3) if flag_v else 0.0),
            "odd_even": float(w.get("oe", 0.2) if flag_oe else 0.0),
            "secondary": float(w.get("sec", 0.2) if flag_sec else 0.0),
            "duration": float(w.get("dur", 0.15) if flag_dur else 0.0),
            "rho": float(w.get("rho", 0.15) if flag_rho else 0.0),
        }
        return flags_bool, penalties

    def _apply_vetting(self, probs: np.ndarray, features: pd.DataFrame) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        cp_idx = self.class_to_idx.get("CP", 0)
        fp_idx = self.class_to_idx.get("FP", 1 if len(self.classes) > 1 else 0)

        adjusted = np.empty_like(probs)
        details: List[Dict[str, Any]] = []
        for i, row in features.iterrows():
            flags, penalties = self._compute_flags(row)
            penalty_total = float(sum(penalties.values()))
            logits = np.log(np.clip(probs[i], EPS, 1.0))
            logits[cp_idx] -= penalty_total
            if len(self.classes) > fp_idx:
                logits[fp_idx] += 0.5 * penalty_total
            adjusted[i] = softmax(logits)
            details.append(
                {
                    "flags": flags,
                    "penalties": penalties,
                    "penalty_total": penalty_total,
                }
            )
        return adjusted, details

    def _predict_shap(self, feature_array: np.ndarray) -> np.ndarray:
        contrib = self.booster.predict(feature_array, pred_contrib=True)
        return contrib[:, :-1]

    def _compute_features(
        self,
        phase_products: PhaseProducts,
        period: float,
        epoch: float,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PhysicalFeatureResult:
        result = extract_physical_features(
            time=phase_products.time,
            flux=phase_products.flux,
            flux_err=phase_products.flux_err,
            phase=phase_products.phase,
            period=period,
            duration=duration,
            epoch=epoch,
            config=self.feature_cfg,
            metadata=metadata,
        )
        return result

    def _approximate_physics_fit(
        self,
        feature_result: PhysicalFeatureResult,
        period: float,
        duration: float,
    ) -> PhysicsFit:
        features = feature_result.features
        depth_fraction = max(features.get("depth_ppm", 0.0), 0.0) * 1e-6
        rp_rs = math.sqrt(depth_fraction) if depth_fraction > 0 else float("nan")
        rho_star = features.get("rho_star_cgs", float("nan")) * 1e3  # revert to SI
        snr = features.get("snr_in_transit", float("nan"))
        vshape = features.get("vshape_ratio", float("nan"))
        odd_even = features.get("odd_even_diff_ppm", float("nan")) * 1e-6
        secondary_depth = features.get("secondary_depth_ppm", float("nan")) * 1e-6

        if math.isnan(rho_star) or rho_star <= 0:
            a_rs = float("nan")
        else:
            try:
                G = 6.67430e-11
                period_seconds = period * 86400.0
                a_rs = (G * rho_star * period_seconds**2 / (3 * math.pi)) ** (1.0 / 3.0)
            except Exception:
                a_rs = float("nan")

        impact = 0.5 * vshape if not math.isnan(vshape) else float("nan")
        goodness = {
            "bic": float(-2.0 * snr) if np.isfinite(snr) else float("nan"),
            "delta_bic_vs_notransit": float(-10.0 * snr) if np.isfinite(snr) else float("nan"),
            "snr": float(snr),
        }

        params = {
            "Rp_Rs": float(rp_rs),
            "b": float(impact),
            "a_Rs": float(a_rs),
            "u1": 0.3,
            "u2": 0.2,
        }
        derived = {
            "rho_star_phot": float(rho_star) if np.isfinite(rho_star) else float("nan"),
            "delta": float(depth_fraction),
            "T14": float(duration),
        }
        secondary = {
            "snr": float(secondary_depth / max(1e-6, depth_fraction)) if depth_fraction > 0 else float("nan"),
            "phase": 0.5,
        }
        status = "estimated"
        return PhysicsFit(
            params=params,
            derived=derived,
            goodness=goodness,
            odd_even=float(odd_even),
            secondary=secondary,
            status=status,
        )

    def _generate_plot_artifacts(
        self,
        star_id: str,
        mission: str,
        phase_products: PhaseProducts,
        probabilities: Dict[str, float],
        physics: PhysicsFit,
        return_plots: bool,
        return_plot_data: bool,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        plots: Dict[str, str] = {}
        plot_data: Dict[str, Any] = {}
        timestamp = _timestamp()
        sanitized = star_id.replace("/", "_").replace("\\", "_")
        base_name = f"{sanitized}_{timestamp}"

        if return_plots:
            plots.update(self._make_phase_plots(base_name, phase_products))
            plots.update(self._make_residual_plot(base_name, phase_products))
            if self.reliability_curve is not None:
                plots.update(self._make_reliability_plot(base_name, probabilities))

        if return_plot_data:
            plot_data["phase"] = {
                "x": _safe_list(phase_products.global_phase),
                "y": _safe_list(phase_products.global_flux),
                "y_model": _safe_list(phase_products.global_flux),
            }
            plot_data["local"] = {
                "x": _safe_list(phase_products.local_phase),
                "y": _safe_list(phase_products.local_flux),
                "y_model": _safe_list(phase_products.local_flux),
            }
            residuals = phase_products.flux - np.interp(
                phase_products.phase,
                phase_products.local_phase,
                phase_products.local_flux,
            )
            plot_data["residuals"] = {
                "x": _safe_list(phase_products.phase),
                "r": _safe_list(residuals),
            }
            if self.reliability_curve is not None:
                rc = self.reliability_curve
                plot_data["reliability"] = {
                    "confidence": rc.confidence.astype(float).tolist(),
                    "accuracy": rc.accuracy.astype(float).tolist(),
                    "counts": rc.counts.astype(int).tolist(),
                }

        return plots, plot_data

    def _make_phase_plots(self, base_name: str, phase_products: PhaseProducts) -> Dict[str, str]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        paths: Dict[str, str] = {}
        global_path = self.plots_dir / f"{base_name}_global.png"
        local_path = self.plots_dir / f"{base_name}_local.png"

        plt.figure(figsize=(4, 2.4))
        plt.plot(phase_products.global_phase, phase_products.global_flux, lw=1.0)
        plt.xlabel("Phase")
        plt.ylabel("Flux")
        plt.title("Global phase")
        plt.tight_layout()
        plt.savefig(global_path, dpi=200)
        plt.close()
        paths["phase_global"] = f"/media/{DEFAULT_PLOTS_SUBDIR}/{global_path.name}"

        plt.figure(figsize=(4, 2.4))
        plt.plot(phase_products.local_phase, phase_products.local_flux, lw=1.0)
        plt.xlabel("Phase")
        plt.ylabel("Flux")
        plt.title("Local phase")
        plt.tight_layout()
        plt.savefig(local_path, dpi=200)
        plt.close()
        paths["phase_local"] = f"/media/{DEFAULT_PLOTS_SUBDIR}/{local_path.name}"

        return paths

    def _make_residual_plot(self, base_name: str, phase_products: PhaseProducts) -> Dict[str, str]:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        model_flux = np.interp(
            phase_products.phase,
            phase_products.local_phase,
            phase_products.local_flux,
        )
        residuals = phase_products.flux - model_flux

        resid_path = self.plots_dir / f"{base_name}_resid.png"
        plt.figure(figsize=(4, 2.0))
        plt.scatter(phase_products.phase, residuals, s=6, alpha=0.6)
        plt.axhline(0.0, color="black", lw=0.8)
        plt.xlabel("Phase")
        plt.ylabel("Residual")
        plt.title("Residuals")
        plt.tight_layout()
        plt.savefig(resid_path, dpi=200)
        plt.close()

        return {"residuals": f"/media/{DEFAULT_PLOTS_SUBDIR}/{resid_path.name}"}

    def _make_reliability_plot(self, base_name: str, probabilities: Dict[str, float]) -> Dict[str, str]:
        if self.reliability_curve is None:
            return {}
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rc = self.reliability_curve
        rel_path = self.plots_dir / f"{base_name}_reliab.png"
        plt.figure(figsize=(4, 3))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
        plt.plot(rc.confidence, rc.accuracy, marker="o", label="OOF")
        cp_prob = probabilities.get("CP", float("nan"))
        if np.isfinite(cp_prob):
            plt.scatter([cp_prob], [cp_prob], color="red", label="Sample", zorder=5)
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.title("Reliability")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(rel_path, dpi=200)
        plt.close()
        return {"reliability": f"/media/{DEFAULT_PLOTS_SUBDIR}/{rel_path.name}"}

    def _decision_from_probs(self, probs: Dict[str, float]) -> Dict[str, Any]:
        pcp = float(probs.get("CP", 0.0))
        cp_high = self.decision_thresholds.get("cp_high", 0.9)
        cp_thresh = self.decision_thresholds.get("cp_thresh", 0.5)
        if pcp >= cp_high:
            label = "cp_high"
            rule = f"p(CP) >= {cp_high:.2f}"
        elif pcp >= cp_thresh:
            label = "candidate"
            rule = f"p(CP) >= {cp_thresh:.2f}"
        else:
            label = "no_candidate"
            rule = f"p(CP) < {cp_thresh:.2f}"
        return {"label": label, "rule": rule, "score": pcp}

    def _load_reliability_curve(self, path: Path, bins: int = 10) -> ReliabilityCurve:
        with np.load(path, allow_pickle=True) as data:
            probs = data["probs"].astype(float)
            labels = data["labels"].astype(int)
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        bin_edges = np.linspace(0.0, 1.0, bins + 1)
        accuracy = np.full(bins, np.nan, dtype=float)
        confidence = np.full(bins, np.nan, dtype=float)
        counts = np.zeros(bins, dtype=int)
        for i in range(bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if np.any(mask):
                accuracy[i] = (predictions[mask] == labels[mask]).mean()
                confidence[i] = confidences[mask].mean()
                counts[i] = int(mask.sum())
            else:
                confidence[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1])
        return ReliabilityCurve(
            bin_edges=bin_edges,
            accuracy=accuracy,
            confidence=confidence,
            counts=counts,
        )


__all__ = ["InferenceEngine"]
