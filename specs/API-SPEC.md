# API-SPEC — Backend API Specification

## 1. Framework
FastAPI + uvicorn. Base URL `http://localhost:8000`. CORS enabled for the Vite dev server.

## 2. Model Selection
Default served model = `augmented`. Optional `model` query/field to choose `baseline`.

## 3. Endpoints

### POST /predict
- **Body**: multipart/form-data, field `file` (image).
- **Response 200**:
```json
{
  "prediction": "Malignant",
  "probability": 0.87,
  "model": "augmented"
}
```
- **Errors**: 400 invalid/missing file, 415 unsupported type.

### POST /heatmap
- **Body**: multipart/form-data, field `file` (image).
- **Response 200**: `image/png` Grad-CAM overlay (or JSON with base64 `overlay`).
- Per [GRADCAM-SPEC.md](GRADCAM-SPEC.md).

### GET /metrics
- **Query**: `exp` = `baseline` | `augmented` (default augmented).
- **Response 200**:
```json
{
  "overall": {"accuracy":0.0,"precision":0.0,"recall":0.0,"f1":0.0,"roc_auc":0.0},
  "confusion_matrix": [[0,0],[0,0]],
  "skin_tone": {
    "Light": {"accuracy":0.0,"count":0},
    "Medium": {"accuracy":0.0,"count":0},
    "Dark": {"accuracy":0.0,"count":0}
  },
  "fairness_gap": 0.0
}
```
- Reads precomputed `artifacts/metrics/<exp>_metrics.json`.

### GET /health
- Returns `{"status":"ok"}`.

## 4. Validation & Security
- Accept only `image/jpeg`, `image/png`; max size 10 MB.
- No path traversal; never execute uploaded content.
- Return typed Pydantic response models.

## 5. Startup
- Load model checkpoint(s) once at startup into memory.
