"""FastAPI service exposing rich inference for the hybrid exoplanet classifier."""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lightkurve_dataset.inference import InferenceEngine  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_BATCH = 50
DEFAULT_MODEL_PATH = Path("models/lightgbm_final.txt")
DEFAULT_CAL_PATH = Path("models/lightgbm_cal.json")
DEFAULT_VET_PATH = Path("configs/vetting.yaml")


class PredictJSON(BaseModel):
    star_id: Optional[str] = Field(None, description="Identifier for the star/target")
    mission: Optional[str] = Field(None, description="Mission name (Kepler/K2/TESS)")
    time: List[float] = Field(..., description="Time values (days)")
    flux: List[float] = Field(..., description="Normalized flux values")
    flux_err: Optional[List[float]] = Field(None, description="Flux uncertainties")
    period: Optional[float] = Field(None, description="Orbital period in days")
    t0: Optional[float] = Field(None, description="Transit epoch (BJD)")
    duration: Optional[float] = Field(None, description="Transit duration in days")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional inference options")

    @field_validator("time", "flux", "flux_err", mode="before")
    @classmethod
    def _ensure_list(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, np.ndarray):
            return value.astype(float).tolist()
        return value


class BatchPredictRequest(BaseModel):
    items: List[PredictJSON]
    options: Optional[Dict[str, Any]] = None


class BatchPredictResponse(BaseModel):
    results: List[Dict[str, Any]]


def _error_response(code: str, message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


def _parse_curve_file(file_bytes: bytes, filename: str) -> Dict[str, np.ndarray]:
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(io.BytesIO(file_bytes))
        missing = {"time", "flux"} - set(df.columns.str.lower())
        if missing:
            raise ValueError(f"faltan columnas {', '.join(sorted(missing))}")
        # Normalize column names
        cols = {c.lower(): c for c in df.columns}
        time = df[cols["time"]].to_numpy(dtype=float)
        flux = df[cols["flux"]].to_numpy(dtype=float)
        flux_err = (
            df[cols.get("flux_err")].to_numpy(dtype=float) if "flux_err" in cols else None
        )
        return {"time": time, "flux": flux, "flux_err": flux_err}

    if suffix in {".fits", ".fit", ".fz"}:
        try:
            from lightkurve_dataset.preprocessing.pipeline import load_lightcurve  # lazy import
        except ModuleNotFoundError as exc:  # pragma: no cover - guard
            raise ValueError("se requiere lightkurve para leer archivos FITS") from exc

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)
        try:
            lc = load_lightcurve(tmp_path)
            return {
                "time": lc.time.value.astype(float),
                "flux": lc.flux.value.astype(float),
                "flux_err": lc.flux_err.value.astype(float) if lc.flux_err is not None else None,
            }
        finally:
            tmp_path.unlink(missing_ok=True)

    raise ValueError("formato de curva no soportado (usa .csv o .fits)")


def create_app(engine: InferenceEngine) -> FastAPI:
    app = FastAPI(title="Hybrid Exoplanet Classifier", version="1.0.0")
    app.mount("/media", StaticFiles(directory=str(engine.media_dir)), name="media")

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/models/info")
    def model_info() -> Dict[str, Any]:
        return engine.get_model_info()

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Any, exc: ValueError) -> JSONResponse:
        return _error_response("BAD_INPUT", str(exc))

    @app.post("/api/predict")
    async def predict_endpoint(
        curve: UploadFile = File(None),
        meta: str = Form(None),
        options: str = Form(None),
        body: PredictJSON = Body(None),
    ) -> JSONResponse:
        try:
            if curve is not None:
                file_bytes = await curve.read()
                if not file_bytes:
                    raise ValueError("archivo de curva vacío")
                curve_arrays = _parse_curve_file(file_bytes, curve.filename)
                meta_payload = json.loads(meta or "{}")
                opts_payload = json.loads(options or "{}")
            else:
                if body is None:
                    raise ValueError("se requiere payload JSON o archivo de curva")
                curve_arrays = {
                    "time": np.asarray(body.time, dtype=float),
                    "flux": np.asarray(body.flux, dtype=float),
                    "flux_err": np.asarray(body.flux_err, dtype=float)
                    if body.flux_err is not None
                    else None,
                }
                meta_payload = {
                    "star_id": body.star_id,
                    "mission": body.mission,
                    "period": body.period,
                    "t0": body.t0,
                    "duration": body.duration,
                }
                opts_payload = body.options or {}

            period = meta_payload.get("period")
            epoch = meta_payload.get("t0") or meta_payload.get("epoch")
            duration = meta_payload.get("duration")
            star_id = meta_payload.get("star_id")
            mission = meta_payload.get("mission")

            result = engine.infer_one(
                star_id=star_id,
                mission=mission,
                time=np.asarray(curve_arrays["time"], dtype=float),
                flux=np.asarray(curve_arrays["flux"], dtype=float),
                flux_err=np.asarray(curve_arrays.get("flux_err"), dtype=float)
                if curve_arrays.get("flux_err") is not None
                else None,
                period=period,
                epoch=epoch,
                duration=duration,
                options=opts_payload,
            )
            return JSONResponse(content=result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"code": "BAD_INPUT", "message": str(exc)}) from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("predict_endpoint failed")
            raise HTTPException(
                status_code=500,
                detail={"code": "INTERNAL", "message": "error interno"},
            ) from exc

    @app.post("/api/predict/batch", response_model=BatchPredictResponse)
    async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
        if not request.items:
            raise HTTPException(status_code=400, detail={"code": "BAD_INPUT", "message": "lista vacía"})
        if len(request.items) > MAX_BATCH:
            raise HTTPException(
                status_code=400,
                detail={"code": "BAD_INPUT", "message": f"máximo {MAX_BATCH} objetos por lote"},
            )
        items_payload: List[Dict[str, Any]] = []
        for item in request.items:
            if item.period is None or item.t0 is None:
                raise HTTPException(
                    status_code=400,
                    detail={"code": "BAD_INPUT", "message": "cada item requiere period y t0"},
                )
            items_payload.append(
                {
                    "star_id": item.star_id,
                    "mission": item.mission,
                    "time": np.asarray(item.time, dtype=float),
                    "flux": np.asarray(item.flux, dtype=float),
                    "flux_err": np.asarray(item.flux_err, dtype=float)
                    if item.flux_err is not None
                    else None,
                    "period": item.period,
                    "t0": item.t0,
                    "duration": item.duration,
                    "options": item.options,
                }
            )
        try:
            results = engine.infer_batch(items_payload, options=request.options)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"code": "BAD_INPUT", "message": str(exc)}) from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("batch inference failed")
            raise HTTPException(
                status_code=500,
                detail={"code": "INTERNAL", "message": "error interno"},
            ) from exc
        return BatchPredictResponse(results=results)

    return app


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s=%s, using default %.2f", name, value, default)
        return default


def _build_engine(
    model: Path,
    cal: Path,
    vetting: Path,
    metrics: Path,
    reliability: Path,
) -> InferenceEngine:
    engine = InferenceEngine(
        model_path=model,
        calibration_path=cal,
        vetting_config=vetting,
        metrics_path=metrics,
        reliability_reference=reliability,
    )
    cp_high = _float_env("CP_HIGH", engine.decision_thresholds["cp_high"])
    cp_thresh = _float_env("CP_THRESH", engine.decision_thresholds["cp_thresh"])
    engine.set_decision_thresholds(cp_high=cp_high, cp_thresh=cp_thresh)
    return engine


def app() -> FastAPI:  # pragma: no cover - entrypoint for uvicorn --factory
    model = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    cal = Path(os.getenv("CAL_PATH", DEFAULT_CAL_PATH))
    vet = Path(os.getenv("VETTING_CFG", DEFAULT_VET_PATH))
    metrics = Path(os.getenv("METRICS_PATH", "reports/holdout_metrics.json"))
    reliability = Path(os.getenv("RELIABILITY_PATH", "data/processed/lightgbm/lightgbm_oof.npz"))
    engine = _build_engine(model, cal, vet, metrics, reliability)
    return create_app(engine)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--cal", type=Path, default=DEFAULT_CAL_PATH)
    parser.add_argument("--vetting", type=Path, default=DEFAULT_VET_PATH)
    parser.add_argument("--metrics", type=Path, default=Path("reports/holdout_metrics.json"))
    parser.add_argument("--reliability", type=Path, default=Path("data/processed/lightgbm/lightgbm_oof.npz"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    engine = _build_engine(args.model, args.cal, args.vetting, args.metrics, args.reliability)
    service = create_app(engine)

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise SystemExit("uvicorn is required to run the API server") from exc

    uvicorn.run(service, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
