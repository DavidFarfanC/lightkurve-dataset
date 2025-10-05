# Hybrid Exoplanet Classification Pipeline

This repository hosts the data preparation layer for a hybrid system that combines a feature-based gradient boosting model (LightGBM) with a morphology-based 1-D CNN. The focus is on preparing consistent inputs sourced from NASA's Kepler/K2/TESS missions via the Lightkurve toolkit.

## Tabla de contenidos

- [Repository Layout](#repository-layout)
- [Quickstart](#quickstart)
- [Cómo funciona el pipeline (detalle)](#cómo-funciona-el-pipeline-detalle)
- [Resumen de scripts y módulos](#resumen-de-scripts-y-módulos)
- [Entrenamiento de modelos](#entrenamiento-de-modelos)
- [Métricas oficiales](#métricas-oficiales)
- [Ensamble con garantías “do-no-harm”](#ensamble-con-garantías-do-no-harm)
- [Pipeline de evaluación final](#pipeline-de-evaluación-final)
- [Interfaces de predicción](#interfaces-de-predicción)
- [API de inferencia rica (v1)](#api-de-inferencia-rica-v1)
- [Arquitectura de servicio](#arquitectura-de-servicio)
- [Desplegar la API](#desplegar-la-api)
- [Despliegue en Vercel](#despliegue-en-vercel)
- [Próximos pasos sugeridos](#próximos-pasos-sugeridos)

## Repository Layout

- `src/lightkurve_dataset/` – reusable Python package containing modules for downloading, cleaning, phase-folding, and feature generation.
- `scripts/prepare_datasets.py` – CLI orchestrator that produces both LightGBM feature tables and CNN tensors.
- `data/raw/` – Light curve FITS files downloaded through Lightkurve.
- `data/processed/` – Outputs ready for model training (`lightgbm/features.parquet` and `cnn/cnn_dataset.npz`).
- `data/splits/` – Saved train/hold-out JSON definitions (por estrella).
- `models/` – Boosters finales + calibraciones (`lightgbm_final.*`).
- `reports/` – Métricas, probabilidades y figuras del hold-out + PDF final.
- `cli/`, `api/` – Interfaces de predicción (CLI para jurado, FastAPI para servicio).
- `artifacts/` – Artefactos ligeros necesarios en despliegue (e.g., `lightgbm_oof.npz`).
- `docs/` – Additional technical notes (to be expanded).

## Quickstart

1. Install dependencies (recommended: dedicated virtual environment):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install lightkurve pandas numpy astropy lightgbm torch torchvision scikit-learn streamlit fastapi[all] pyarrow pyyaml
   ```

2. Place the curated metadata catalog at `data/metadata/catalog.csv`. It must contain at least the columns `target_id`, `label`, `period`, `epoch`, `duration`, and optionally `mission`.

3. Run the preprocessing pipeline:

   ```bash
   python scripts/prepare_datasets.py --download --max-targets 100
   ```

   The `--download` flag fetches PDCSAP light curves for the targets listed in the catalog. Omit it if the FITS files are already present in `data/raw/`.

   **Tip for hackathons / quick demos:** coloca un lote pequeño de FITS (por ejemplo, 50–200 curvas ya descargadas) dentro de `data/raw/`, y ejecuta

   ```bash
   python scripts/prepare_datasets.py --max-targets 100 --allow-missing
   ```

   El flag `--allow-missing` descarta objetivos que no tengan archivo asociado y reduce el tiempo de preparación. Puedes ir ampliando el lote conforme dispongas de más datos.

## Resumen de scripts y módulos

- `scripts/prepare_datasets.py` – orquesta descarga, limpieza, extracción de features y generación de tensores/tabla.
- `scripts/make_holdout.py` – construye splits estratificados por estrella (JSON) para reproducir el hold-out final.
- `scripts/train_lightgbm.py` – entrena el modelo físico-ML con StratifiedGroupKFold, aplica calibración y guarda booster + metadatos.
- `scripts/train_cnn.py` – pipeline de la rama CNN (datasets sintéticos en esta versión).
- `scripts/eval_on_holdout.py` – evalúa el LightGBM sobre el hold-out y exporta métricas multiclase/binarias, reliability y probabilidades.
- `scripts/ensemble_blend.py` – ensamble “do-no-harm” con calibración, gating y vetting físico en logit space.
- `scripts/figures.py` – genera figuras oficiales (tablas, matrices de confusión, reliability).
- `scripts/build_report.py` – compila el PDF final (`reports/final_report.pdf`).
- `cli/predict.py` – CLI para producir `pred_{star}.json` con probabilidades calibradas, flags y SHAP.
- `api/serve.py` – servicio FastAPI que expone la inferencia rica y sirve medios bajo `/media`.

El paquete `src/lightkurve_dataset` agrupa:

- `config.py` – dataclasses con los knobs del pipeline.
- `preprocessing/` – limpieza (`cleaning.py`), faseado y vistas (`phase.py`), pipeline end-to-end (`pipeline.py`).
- `features/physical.py` – extracción de features físicos + diagnósticos.
- `inference.py` – motor de inferencia (limpieza, features, LightGBM calibrado, vetting, explicabilidad, plots).
- `datasets/` – utilidades para generar datasets LightGBM/CNN.
- `data/` – carga de catálogos y fetchers MAST.
- `utils/` – logging y helpers compartidos.

## Entrenamiento de modelos

1. **Preparar datos** – Ajusta `data/metadata/catalog.csv` y ejecuta `scripts/prepare_datasets.py` para conseguir `data/processed/lightgbm/features.parquet` y tensores CNN.
2. **Split hold-out** – `scripts/make_holdout.py --star-id target_id --test-size 0.12 --seed 42 --out data/splits/holdout_star42.json` conserva 12 % de estrellas para evaluación final.
3. **Entrenar LightGBM** – `scripts/train_lightgbm.py --features ... --split-json ... --calibrate oof_temperature --save-model models/lightgbm_final.txt --save-cal models/lightgbm_cal.json` produce métricas OOF, probabilidades y el booster definitivo.
4. **Evaluar hold-out** – `scripts/eval_on_holdout.py --vetting-config configs/vetting.yaml --out reports/holdout_metrics.json --probs-out reports/holdout_probs.npz` aplica vetting físico y guarda métricas/curvas.
5. **Figuras + informe** – Ejecuta `scripts/figures.py` y `scripts/build_report.py` para generar `reports/figures/*.png` y `reports/final_report.pdf`.
6. **CNN / Ensamble** – `scripts/train_cnn.py` (opcional) y `scripts/ensemble_blend.py` documentan el marco híbrido y ratifican `α=0`.
7. **Artefactos para despliegue** – Copia `data/processed/lightgbm/lightgbm_oof.npz` a `artifacts/lightgbm_oof.npz` o genera el archivo directamente en `artifacts/` para evitar subir el directorio completo `data/` en entornos serverless.

Todos los artefactos resultantes (modelos, calibraciones, métricas, figuras) se conservan en el repositorio para trazabilidad y para alimentar la API/CLI.

## Cómo funciona el pipeline (detalle)

La lógica completa está centralizada en `scripts/prepare_datasets.py` y el paquete `lightkurve_dataset`. A continuación, un recorrido paso a paso de cada bloque:

1. **Configuración (`lightkurve_dataset.config`)**
   - Define dataclasses con todos los knobs del pipeline: descarga (`DataFetchConfig`), limpieza (`QualityMaskConfig`), faseado (`PhaseFoldConfig`), extracción de features físicos (`FeatureConfig`), generación de tensores para la CNN (`CNNConfig`) y rutas de trabajo (`PipelineConfig`).
   - El script crea una instancia de `PipelineConfig` y permite modificar parámetros vía flags (`--max-targets`, `--download`, etc.).

2. **Carga de catálogo (`lightkurve_dataset.data.catalog`)**
   - `load_metadata_catalog` lee `data/metadata/catalog.csv`, normaliza nombres de columnas a snake_case y asegura que existan los campos clave (`target_id`, `label`, `period`, `epoch`, `duration`, `mission`).
   - Si pasas `--max-targets`, se limita el número de objetivos por misión manteniendo estratificación interna.

3. **Descarga opcional (`lightkurve_dataset.data.fetch`)**
   - Si se usa `--download`, el script itera por misión (`Kepler`, `K2`, `TESS`) y llama a Lightkurve (`lk.search_lightcurve`).
   - Cada misión usa su propia copia de `DataFetchConfig` para ajustar el filtro de búsqueda (autor, cadencia, etc.).
   - Los FITS se guardan bajo `data/raw/mastDownload/<Mission>/...`. Si la red falla, puedes trabajar solo con archivos ya presentes.

4. **Asociación catálogo ↔ FITS (`_attach_lightcurve_paths`)**
   - Recorre `data/raw/` buscando archivos `*.fits` y asigna la ruta correcta a cada `target_id`.
   - Si no encuentra un archivo y usas `--allow-missing`, la fila se elimina silenciosamente. Sin ese flag, el pipeline aborta para evitar datasets incompletos.

5. **Preprocesamiento individual (`lightkurve_dataset.preprocessing`)
   - `pipeline.load_lightcurve` intenta abrir el FITS con Lightkurve; si no reconoce el formato (por ejemplo, curvas sintéticas generadas localmente), cae a un lector manual con `astropy.io.fits` y construye un `LightCurve` a partir de las columnas `TIME`, `FLUX`, `FLUX_ERR`.
   - `clean_lightcurve` aplica:
     - Máscara de calidad (`remove_quality_flags`).
     - Normalización a ppm.
     - Sigma-clipping de outliers.
     - Detrending con Savitzky–Golay (parámetros configurables).
   - `phase.fold_phase` y `phase.create_global_local_views` generan:
     - Vista global: curva faseada en [-0.5, 0.5] con 2001 bins.
     - Vista local: ventana centrada en el tránsito con 201 bins (anchura configurable).
   - El resultado incluye arrays `time`, `flux`, `flux_err`, `phase`, y las vistas global/local usadas por ambas ramas.

6. **Extracción de features físicos (`lightkurve_dataset.features.physical`)**
   - Calcula profundidad del tránsito, SNR, diferencia odd-even, detección secundaria, métrica de forma (V-shape), densidad estelar aparente y parámetros BLS (`BoxLeastSquares`).
   - Si la ventana en tránsito es demasiado corta o ocurre un error, el registro se descarta (se loguea el warning y se continúa). Puedes desactivar el cálculo de BLS estableciendo `FeatureConfig.compute_bls = False` desde el script para acelerar corridas exploratorias.

7. **Construcción de datasets (`lightkurve_dataset.datasets`)**
   - **LightGBM (`datasets.lightgbm`)**: compone un DataFrame con los features anteriores + metadata (`target_id`, `mission`, `label`) y lo guarda como `data/processed/lightgbm/features.parquet`.
   - **CNN (`datasets.cnn`)**: apila los canales `global_flux`, `global_err`, `mask` (y equivalentes locales) en tensores `float32`, los serializa en `cnn_dataset.npz` y exporta `cnn_metadata.csv` + `cnn_config.json` para reproducibilidad.

8. **Generación de datos sintéticos (`scripts/generate_mock_lightcurves.py`)**
   - Utilidad adicional para hackathons: toma el catálogo y crea curvas de luz artificiales con ruido y tránsitos idealizados. Usa Lightkurve para escribir FITS compatibles y permite bootstrapping rápido sin depender de MAST.

9. **Entrenamiento rápido sobre el subset**
   - Con `data/processed/lightgbm/features.parquet` puedes entrenar LightGBM en segundos (`sklearn` + `lgb.train`).
   - Con `data/processed/cnn/cnn_dataset.npz` entrenas una pequeña 1-D CNN (dual-branch global/local). Las probabilidades de ambos modelos se ensamblan posteriormente con un promedio ponderado o un meta-clasificador.

10. **Salida final**
    - Los artefactos clave quedan en:
      - `data/processed/lightgbm/features.parquet` y `lightgbm_mock.txt` (modelo entrenado).
      - `data/processed/cnn/cnn_dataset.npz`, `cnn_metadata.csv`, `cnn_config.json`.
    - Estos sirven de insumo directo para las etapas de entrenamiento, calibración y la interfaz web (Streamlit + FastAPI) planteada en el diseño general.

## Workflow sugerido en hackathon

1. Ejecuta `scripts/generate_mock_lightcurves.py --per-label 40` para generar curvas sintéticas (opcional si ya tienes FITS reales).
2. Corre el pipeline: `python scripts/prepare_datasets.py --max-targets 120 --allow-missing`.
3. Entrena LightGBM rápido y guarda métricas (ver ejemplo en este README).
4. Entrena la CNN sobre `cnn_dataset.npz` (unas pocas épocas con PyTorch o TensorFlow).
5. Implementa el ensamble (`alpha * LightGBM + (1 - alpha) * CNN`), aplica reglas físicas (odd-even, duraciones, densidad) y genera visualizaciones/interpretabilidad (SHAP, saliency).
6. Conecta el backend (FastAPI) y la interfaz (Streamlit) para consumir los modelos y mostrar resultados en la demo.

## Entrenamiento con validación robusta

Con los scripts `scripts/train_lightgbm.py` y `scripts/train_cnn.py` puedes reproducir los bloques A/B con validación estratificada y focal loss.

- **LightGBM (rama física)**
  ```bash
  python scripts/train_lightgbm.py \
      --features data/processed/lightgbm/features.parquet \
      --folds 5 --repeats 3 --learning-rate 0.05 --num-leaves 63
  ```
  - Usa `StratifiedGroupKFold` (5×3=15 corridas) agrupando por `target_id`.
  - Reporta media ± σ de Macro-F1, log-loss y ECE en `data/processed/lightgbm/lightgbm_mock_cv.json`.
  - Entrena un modelo final sobre todo el dataset (`lightgbm_mock.txt`), guarda el mapping de etiquetas (`.labels.json`) y exporta probabilidades out-of-fold (`lightgbm_oof.npz`). Durante la preparación se descarta `target_id=11853255` (la curva falló al extraer features); conviene usar la metadata del CNN para reconstruir el `target_id` y alinear las ramas.

- **CNN dual (rama morfológica)**
  ```bash
  python scripts/train_cnn.py \
      --dataset data/processed/cnn/cnn_dataset.npz \
      --folds 5 --repeats 2 --epochs 20 --batch-size 64 --gamma 2.0
  ```
  - Implementa StratifiedKFold (proxy cuando cada curva pertenece a una estrella única).
  - Incluye focal loss + class weights, augmentaciones sencillas y early stopping basado en macro-F1.
  - Genera métricas por fold y su resumen (`cnn_mock.metrics.json`), exporta probabilidades OOF (`cnn_oof.npz`) y entrena un modelo final (`cnn_mock.pt`).
  - Nota: si tu entorno bloquea memoria compartida (`OMP Error #179: Can't open SHM2`), ejecuta este script en un entorno local sin sandbox (por ejemplo, en tu terminal habitual o en Google Colab). Copia el archivo `data/processed/cnn/cnn_dataset.npz` y el script; las salidas usan la misma estructura descrita arriba.

## Métricas oficiales

Los números de referencia para la entrega final se calculan sobre 201 estrellas de entrenamiento y un hold-out por estrella del 12 % (27 objetos):

- **LightGBM (OOF estratificado por estrella, 5×3)**
  - Macro-F1 `0.980 ± 0.022`
  - Log-loss `0.082 ± 0.066`
  - Expected Calibration Error `0.029 ± 0.015`
  - Resultados completos en `data/processed/lightgbm/lightgbm_mock_cv.json` y probabilidades en `data/processed/lightgbm/lightgbm_oof.npz`.
- **Hold-out por estrella (post-calibración + vetting físico)**
  - Macro-F1 `0.921`
  - Log-loss `0.167`
  - Expected Calibration Error `0.035`
  - Brier score `0.072`
  - Matriz de confusión y reliability en `reports/holdout_metrics.json`.
- **Binario CP vs (FP+PC)**
  - Kepler: ROC-AUC `1.000`, PR-AUC `1.000`, Precision@Recall≥0.96 `1.000`, Recall@Precision≥0.90 `1.000`.
- **CNN**
  - Se mantiene como componente híbrido opcional. Sus OOF actuales no aportan lift (Macro-F1 ≈ `0.31`, Log-loss ≈ `1.04`), por lo que no interviene en el score final pero se documenta para futuras iteraciones (ver `data/processed/cnn/cnn_oof.npz`).

## Ensamble con garantías “do-no-harm”

Evaluamos el marco híbrido con:

```bash
python scripts/ensemble_blend.py \
    --features data/processed/lightgbm/features.parquet \
    --lightgbm-oof data/processed/lightgbm/lightgbm_oof.npz \
    --cnn-oof data/processed/cnn/cnn_oof.npz \
    --alpha-max 0.25 \
    --output data/processed/ensemble/blend_probs.npz
```

El script calibra cada rama (temperature scaling), busca `α ≤ α_max` que minimiza el log-loss sin empeorar al LGBM, aplica gating dependiente de SNR/periodo/duración y finalmente ajusta logits con vetting físico (V-shape, odd-even, secundaria, duración, densidad). También emite métricas y deja un resumen JSON (`data/processed/ensemble/blend_probs.json`).

### Resultados del ensamble 2024

- Temperaturas OOF: LightGBM `T≈0.84`, CNN `T≈1.05`.
- `α` óptimo (cota 0.25) = **0.000** → el blend se mantiene neutro y respalda la decisión “modelo físico primero”.
- Métricas sobre el conjunto común (201 curvas):
  - LightGBM calibrado: Macro-F1 `0.980`, LogLoss `0.075`, ECE `0.014`.
  - CNN calibrado: Macro-F1 `0.306`, LogLoss `1.043`, ECE `0.045`.
  - Blend (α=0.0): Macro-F1 `0.980`, LogLoss `0.129`, ECE `0.051`.
  - Blend + vetting físico: Macro-F1 `0.980`, LogLoss `0.129`, ECE `0.050`.
- Conclusión: la infraestructura híbrida queda habilitada (documenta transparencia y vetting), pero el score final recae en el LightGBM calibrado hasta disponer de más ejemplos FP/PC para la CNN.

## Pipeline de evaluación final

1. **Split hold-out por estrella**

   ```bash
   python scripts/make_holdout.py \
       --input data/processed/lightgbm/features.parquet \
       --star-id target_id \
       --test-size 0.12 \
       --seed 42 \
       --out data/splits/holdout_star42.json
   ```

   El JSON resultante lista `train_ids` y `holdout_ids`. Puedes proporcionar `--metadata-csv` si extraes los ID desde otro archivo.

2. **Entrenamiento + calibración (OOF temperature scaling)**

   ```bash
   python scripts/train_lightgbm.py \
       --features data/processed/lightgbm/features.parquet \
       --split-json data/splits/holdout_star42.json \
       --calibrate oof_temperature \
       --save-model models/lightgbm_final.txt \
       --save-cal models/lightgbm_cal.json
   ```

   Esto persiste el booster final, el mapa de etiquetas y la lista de features (`models/lightgbm_final.*`). Las probabilidades OOF quedan en `data/processed/lightgbm/lightgbm_oof.npz`.

3. **Evaluación hold-out con vetting físico**

   ```bash
   python scripts/eval_on_holdout.py \
       --features data/processed/lightgbm/features.parquet \
       --split-json data/splits/holdout_star42.json \
       --model models/lightgbm_final.txt \
       --cal models/lightgbm_cal.json \
       --label-map models/lightgbm_final.labels.json \
       --vetting-config configs/vetting.yaml \
       --out reports/holdout_metrics.json \
       --probs-out reports/holdout_probs.npz
   ```

   El reporte JSON incluye métricas multiclase, binario por misión, matriz de confusión, reliability y temperatura aplicada.

4. **Figuras oficiales**

   ```bash
   MPLCONFIGDIR=/tmp/mplcache \
   python scripts/figures.py \
       --oof data/processed/lightgbm/lightgbm_oof.npz \
       --holdout reports/holdout_metrics.json \
       --holdout-probs reports/holdout_probs.npz \
       --label-map models/lightgbm_final.labels.json \
       --outdir reports/figures/
   ```

   Genera `metrics_multiclass.png`, `metrics_binary.png`, `confusion_matrices.png` y `reliability.png`.

5. **Ensamble híbrido (opcional / documentación)**

   ```bash
   python scripts/ensemble_blend.py \
       --features data/processed/lightgbm/features.parquet \
       --lightgbm-oof data/processed/lightgbm/lightgbm_oof.npz \
       --cnn-oof data/processed/cnn/cnn_oof.npz \
       --alpha-max 0.25 \
       --output data/processed/ensemble/blend_probs.npz
   ```

6. **Reporte PDF automatizado**

   ```bash
   python scripts/build_report.py \
       --metrics-oof data/processed/ensemble/blend_probs.json \
       --metrics-holdout reports/holdout_metrics.json \
       --fig-dir reports/figures \
       --blend-npz data/processed/ensemble/blend_probs.npz \
       --out reports/final_report.pdf
   ```

## Interfaces de predicción

- **CLI jurado / lotes:**

  ```bash
  MPLCONFIGDIR=/tmp/mplcache \
  python cli/predict.py \
      --curves data/curves/*.csv \
      --model models/lightgbm_final.txt \
      --cal models/lightgbm_cal.json \
      --vetting configs/vetting.yaml \
      --outdir predictions/
  ```

  Para cada registro genera `pred_{star}.json` con probabilidades calibradas, flags físicos y top-SHAP. Si tu CSV incluye columnas `phase_global`/`phase_local` como listas, se guardan mini-plots en `predictions/`.

- **API FastAPI (serve.py):**

  ```bash
  PYTHONPATH=src \
  python api/serve.py \
      --model models/lightgbm_final.txt \
      --cal models/lightgbm_cal.json \
      --vetting configs/vetting.yaml \
      --host 0.0.0.0 --port 8000
  ```

  La API monta `/media` para figuras y expone `GET /healthz`, `GET /api/models/info`, `POST /api/predict` y `POST /api/predict/batch`.

## API de inferencia rica (v1)

La nueva API expone inferencia “plug-and-play” a partir de curvas de luz crudas. Entradas posibles: multipart (`curve` en CSV/FITS + `meta` JSON) o JSON puro con arrays `time`/`flux`.

### Endpoints principales

- `POST /api/predict` — inferencia rica para una curva.
- `POST /api/predict/batch` — lote pequeño (≤50 objetos) con el mismo contrato que la versión unitary.
- `GET /api/models/info` — metadatos del modelo (hash, métricas hold-out, umbrales de decisión, features).
- `GET /healthz` — chequeo de salud.

Los artefactos (PNGs) se sirven bajo `/media/plots/*` gracias a `StaticFiles`.

### Contrato de `POST /api/predict`

Entrada multipart típica:

```bash
curl -s -X POST http://localhost:8000/api/predict \
  -F "curve=@data/curves/ejemplo.csv" \
  -F 'meta={"star_id":"KIC 123","period":3.1416,"t0":2457000.123,"mission":"TESS"}' \
  -F 'options={"return_plots":true,"return_plot_data":true}' | jq .
```

Entrada JSON equivalente:

```bash
curl -s -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
        "star_id": "KIC 123456",
        "mission": "TESS",
        "time": [...],
        "flux": [...],
        "period": 3.1416,
        "t0": 2457000.123,
        "options": {"return_plots": true, "return_plot_data": true}
      }' | jq .
```

Salida resumida:

```json
{
  "star_id": "KIC 123456",
  "mission": "TESS",
  "version": {
    "model_hash": "a1b2c3d4",
    "trained_at": "2024-03-08T11:22:33Z",
    "features": ["depth_ppm", "snr_in_transit", ...],
    "holdout_metrics": {"macro_f1": 0.921, "logloss": 0.167, "ece": 0.035, "brier": 0.072},
    "decision_thresholds": {"cp_high": 0.90, "cp_thresh": 0.50}
  },
  "probs": {"CP": 0.83, "PC": 0.12, "FP": 0.05},
  "prediction": "CP",
  "decision": {
    "label": "candidate",
    "rule": "p(CP) >= 0.50",
    "score": 0.83,
    "confidence": {"entropy": 0.65, "margin": 0.71, "confidence_index": 0.82}
  },
  "physics_fit": {
    "model": "MandelAgol2002",
    "params": {"Rp_Rs": 0.05, "b": 0.40, "a_Rs": 12.3, "u1": 0.3, "u2": 0.2},
    "derived": {"rho_star_phot": 1.1, "delta": 0.0025, "T14": 2.1},
    "goodness": {"bic": 1234.5, "delta_bic_vs_notransit": -120.0, "snr": 14.3},
    "odd_even": 0.01,
    "secondary": {"snr": 1.2, "phase": 0.5},
    "status": "estimated"
  },
  "vetting": {
    "enabled": true,
    "penalties": {"v_shape": 0.0, "odd_even": 0.0, "secondary": 0.1, "duration": 0.0, "rho": 0.0},
    "flags": ["secondary"],
    "post_penalty_probs": {"CP": 0.81, "PC": 0.14, "FP": 0.05}
  },
  "explainability": {
    "top_shap": [
      {"feature": "depth_ppm", "shap": 0.42},
      {"feature": "duration_days", "shap": 0.31},
      {"feature": "bls_power", "shap": 0.18}
    ]
  },
  "plots": {
    "phase_global": "/media/plots/KIC123_global.png",
    "phase_local": "/media/plots/KIC123_local.png",
    "residuals": "/media/plots/KIC123_resid.png",
    "reliability": "/media/plots/KIC123_reliab.png"
  },
  "plot_data": {
    "phase": {"x": [...], "y": [...], "y_model": [...]},
    "local": {"x": [...], "y": [...], "y_model": [...]},
    "residuals": {"x": [...], "r": [...]},
    "reliability": {"confidence": [...], "accuracy": [...], "counts": [...]}
  }
}
```

Reglas de decisión de negocio (`cp_thresh=0.5`, `cp_high=0.9`) se pueden sobreescribir vía variables de entorno `CP_THRESH` y `CP_HIGH`.

Errores devuelven un payload consistente `{"error": {"code": "BAD_INPUT", "message": "..."}}` con códigos `BAD_INPUT`, `FIT_FAIL`, `MODEL_ERROR` o `INTERNAL`.

## Arquitectura de servicio

- **Ingesta → Preprocesamiento**: `InferenceEngine` recibe la curva, aplica el mismo pipeline de limpieza que se usó en entrenamiento (máscara de calidad, sigma-clipping, detrending y faseo) para garantizar distribuciones consistentes.
- **Modelo físico-ML**: LightGBM calibrado (`models/lightgbm_final.*`) + vetting físico (penalización en logit space usando umbrales de `configs/vetting.yaml`).
- **Explicabilidad y física**: se calculan SHAP top-k y métricas físicas aproximadas (parámetros tipo Mandel & Agol, delta BIC, SNR, odd-even, secundaria).
- **Artefactos**: figuras (fase global/local, residuales, reliability) en `media/plots/`, arrays listos para el frontend en la respuesta JSON, métricas históricas en `reports/holdout_metrics.json`.
- **Interfaces**:
  - CLI (`cli/predict.py`) para lotes locales/competencia.
  - API (`api/serve.py`) basada en FastAPI con `/api/predict`, `/api/predict/batch`, `/api/models/info`, `/healthz` y estáticos en `/media`.
- **Persistencia**: modelos bajo `models/`, configuraciones en `configs/`, datos procesados en `data/processed/`, medios estáticos en `media/`.

## Desplegar la API

1. **Requisitos**: Python ≥3.10 con dependencias del Quickstart (`lightkurve`, `pyarrow`, `pyyaml`, `fastapi`, `uvicorn`, etc.).
2. **Variables opcionales**:
   - `MODEL_PATH`, `CAL_PATH`, `VETTING_CFG` si tus artefactos viven fuera de la ruta por defecto.
   - `METRICS_PATH` (por defecto `reports/holdout_metrics.json`).
   - `RELIABILITY_PATH` apuntando a `artifacts/lightgbm_oof.npz` (ver nota en la sección anterior).
   - `CP_THRESH`, `CP_HIGH` para ajustar la lógica de decisión (default 0.50 / 0.90).
   - `MPLCONFIGDIR` si deseas mover el caché de Matplotlib.
3. **Arrancar servidor**:

   ```bash
   export PYTHONPATH=src
   uvicorn api.serve:app --factory --host 0.0.0.0 --port 8000
   ```

   o bien `python api/serve.py --host 0.0.0.0 --port 8000`.
4. **Verificar**: `curl http://localhost:8000/healthz` y `curl http://localhost:8000/api/models/info`.
5. **Probar inferencia**: usar los ejemplos `curl` (multipart o JSON) o la CLI (`cli/predict.py`).
6. **Endurecer producción** (opcional): sirve `/media` detrás de CDN/S3, agrega autenticación (API key/JWT), configura HTTPS vía proxy (Traefik/Nginx) y fija límites de carga (`--limit-concurrency`, `--limit-max-requests`, tamaño máx. de archivo).

## Despliegue en Vercel

La API puede ejecutarse como función serverless en Vercel.

1. **Prerrequisitos**
   - Cuenta en [vercel.com](https://vercel.com/) y CLI instalada (`npm i -g vercel`).
   - Archivo `requirements.txt` (incluido) para que Vercel instale dependencias.
   - `vercel.json` con la configuración de la función y `api/index.py` como entrypoint serverless.
2. **Archivos clave**
   - `vercel.json`

     ```json
     {
       "functions": {
         "api/index.py": {
           "runtime": "python3.10",
           "maxDuration": 60
         }
       },
       "routes": [{"src": "/(.*)", "dest": "api/index.py"}]
     }
     ```

   - `api/index.py`

     ```python
     from api.serve import app as create_app

     app = create_app()
     ```

3. **Variables de entorno**
   - Define en el panel de Vercel (o con `vercel env`) las rutas si difieren de las por defecto: `MODEL_PATH`, `CAL_PATH`, `VETTING_CFG`, `METRICS_PATH`, `RELIABILITY_PATH`, `CP_THRESH`, `CP_HIGH`.
   - Considera subir los artefactos del modelo (`models/lightgbm_final.*`, `configs/vetting.yaml`, `reports/holdout_metrics.json`, `artifacts/lightgbm_oof.npz`) al repo o a storage privado accesible mediante variable de entorno.
4. **Despliegue**
   ```bash
   vercel login               # una vez
   vercel link                # asociar el repo al proyecto
   vercel --prod              # despliegue producción
   ```
5. **Verificación**
   - `vercel open /healthz`
   - `vercel open /api/models/info`
   - Ejecuta una inferencia con `curl` apuntando al dominio generado.

`/.vercelignore` excluye directorios pesados (`.venv`, `data/`, `logs/`, etc.) para cumplir el límite de 100 MB del plan gratuito; asegúrate de mantener los artefactos mínimos en `models/`, `configs/`, `reports/` y `artifacts/`.

### Despliegue desde la web de Vercel

1. Inicia sesión en [vercel.com](https://vercel.com/) y pulsa **Add New → Project**.
2. Autoriza el acceso a tu repositorio `lightkurve-dataset` (si no aparece, usa **Import Git Repository** e introduce la URL).
3. En la pantalla de configuración inicial:
   - **Framework Preset**: selecciona “Python”.
   - **Root Directory**: deja la raíz del repo.
   - **Build & Output Settings**: Vercel detecta automáticamente la función `api/index.py`; no necesita build command.
4. En la sección **Environment Variables**, agrega las mismas claves usadas en CLI:
   - `MODEL_PATH = models/lightgbm_final.txt`
   - `CAL_PATH = models/lightgbm_cal.json`
   - `VETTING_CFG = configs/vetting.yaml`
   - `METRICS_PATH = reports/holdout_metrics.json`
   - `RELIABILITY_PATH = artifacts/lightgbm_oof.npz`
   - `CP_THRESH = 0.5`
   - `CP_HIGH = 0.9`
5. Haz clic en **Deploy**. Tras unos segundos se mostrará el dominio público. Verifica `https://<tu-dominio>/healthz` y `https://<tu-dominio>/api/models/info` para confirmar que la API responde.

Recuerda que Vercel es stateless: `media/` y archivos generados se guardan en almacenamiento efímero. Para compartir plots deberás copiar la respuesta (`plots` URLs) inmediatamente o subir los PNG a un bucket externo durante la llamada.

## Próximos pasos sugeridos

- Recolectar más curvas FP/PC reales o augmentations para que la CNN aporte lift y revaluar el blend.
- Integrar monitoreo de drift/transfer tests usando nuevos splits (TESS/K2) antes de inferencia en producción.
- Añadir empaquetado ligero (Docker o scripts de despliegue) que unan API + CLI + reporte para entregas recurrentes.
