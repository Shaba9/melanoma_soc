"""FastAPI backend per API-SPEC.md.

Serves melanoma classification (`/predict`), Grad-CAM overlays (`/heatmap`), and
precomputed evaluation metrics (`/metrics`). Trained checkpoints are loaded once
and cached in memory. Default served model is `augmented`.
"""
import io
import json
import logging
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError

from .. import config
from ..data import get_preprocess_transform
from ..evaluation import load_experiment_model
from ..gradcam import generate_overlay
from ..training.train import get_device
from .schemas import (
    HealthResponse,
    MetricsResponse,
    OverallMetrics,
    PredictResponse,
    SkinToneMetric,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "augmented"
VALID_MODELS = {"baseline", "augmented"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# Human-readable captions for generated figures (FIGURE-SPEC.md).
FIGURE_TITLES = {
    "architecture.png": "Model Architecture",
    "augmentation_examples.png": "Augmentation Examples",
    "training_curves.png": "Training Curves",
    "roc_curve.png": "ROC Curve (DDI)",
    "confusion_matrices.png": "Confusion Matrices (DDI)",
    "gradcam_examples.png": "Grad-CAM Examples",
}

app = FastAPI(title="Melanoma SOC API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_device = get_device()
_model_cache: dict[str, torch.nn.Module] = {}


def _get_model(name: str) -> torch.nn.Module:
    """Return a cached trained model, loading its checkpoint on first use."""
    if name not in VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model '{name}'.")
    if name not in _model_cache:
        try:
            model, _ = load_experiment_model(name, _device)
        except FileNotFoundError:
            raise HTTPException(status_code=503,
                                detail=f"Model '{name}' is not trained yet.")
        model.eval()
        _model_cache[name] = model
    return _model_cache[name]


async def _read_image(file: UploadFile) -> Image.Image:
    """Validate upload type/size and decode to an RGB PIL image."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415,
                            detail="Only image/jpeg and image/png are supported.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty or missing file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit.")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL),
) -> PredictResponse:
    image = await _read_image(file)
    net = _get_model(model)
    tensor = get_preprocess_transform()(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        probs = torch.softmax(net(tensor), dim=1)[0]
    pred_class = int(probs.argmax().item())
    return PredictResponse(
        prediction=config.CLASS_NAMES[pred_class],
        probability=float(probs[config.MALIGNANT].item()),
        model=model,
    )


@app.post("/heatmap")
async def heatmap(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL),
) -> StreamingResponse:
    image = await _read_image(file)
    net = _get_model(model)
    overlay, _ = generate_overlay(net, image, _device)
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@app.get("/metrics", response_model=MetricsResponse)
def metrics(exp: str = Query(DEFAULT_MODEL)) -> MetricsResponse:
    if exp not in VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown experiment '{exp}'.")
    path = config.METRICS_DIR / f"{exp}_metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404,
                            detail=f"Metrics for '{exp}' not found.")
    data = json.loads(path.read_text())
    overall = data["overall"]
    skin_tone = {
        group: SkinToneMetric(accuracy=vals["accuracy"], count=vals["count"])
        for group, vals in data.get("skin_tone", {}).items()
    }
    return MetricsResponse(
        overall=OverallMetrics(
            accuracy=overall["accuracy"],
            precision=overall["precision"],
            recall=overall["recall"],
            f1=overall["f1"],
            roc_auc=overall["roc_auc"],
        ),
        confusion_matrix=overall["confusion_matrix"],
        skin_tone=skin_tone,
        fairness_gap=data.get("fairness_gap", 0.0),
    )


@app.get("/experiments")
def experiments() -> dict:
    """List experiments that have saved metrics (for UI toggles)."""
    available = [m for m in sorted(VALID_MODELS)
                 if (config.METRICS_DIR / f"{m}_metrics.json").exists()]
    return {"experiments": available, "default": DEFAULT_MODEL}


@app.get("/figures")
def list_figures() -> dict:
    """List generated figure PNGs available in `artifacts/figures/`."""
    figures_dir = config.FIGURES_DIR
    names = sorted(p.name for p in figures_dir.glob("*.png")) if figures_dir.exists() else []
    items = [{"name": n, "title": FIGURE_TITLES.get(n, n)} for n in names]
    return {"figures": items}


@app.get("/figures/{name}")
def get_figure(name: str) -> FileResponse:
    """Serve a single figure PNG by filename (no path traversal)."""
    # Reject any path separators or parent references.
    if name != Path(name).name or not name.endswith(".png"):
        raise HTTPException(status_code=400, detail="Invalid figure name.")
    path = config.FIGURES_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Figure '{name}' not found.")
    return FileResponse(path, media_type="image/png")