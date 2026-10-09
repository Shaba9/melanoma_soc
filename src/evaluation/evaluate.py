"""DDI evaluation per EXP-SPEC.md: run inference, compute overall metrics, persist JSON.

Raw predictions (with skin-tone groups) are saved alongside metrics so fairness
(Prompt 6) and figures (Prompt 8) can reuse them without re-running inference.
"""
import json
import logging

import torch

from .. import config
from ..data import build_ddi_loader, get_preprocess_transform
from ..models import build_model
from ..training.metrics import classification_metrics
from ..training.train import get_device, set_seed

logger = logging.getLogger(__name__)


@torch.no_grad()
def run_inference(model, loader, device) -> dict:
    """Run inference over a loader; return labels, preds, P(Malignant), skin-tone groups."""
    model.eval()
    labels_all, preds_all, probs_all, tones = [], [], [], []
    for images, labels, meta in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, config.MALIGNANT]
        preds = logits.argmax(dim=1)
        labels_all += labels.tolist()
        preds_all += preds.cpu().tolist()
        probs_all += probs.cpu().tolist()
        tones += list(meta.get("skin_tone_group", []))
    return {
        "labels": labels_all,
        "preds": preds_all,
        "probs": probs_all,
        "skin_tone_groups": tones,
    }


def load_experiment_model(experiment: str, device):
    """Load a trained checkpoint into a fresh EfficientNet-B0."""
    ckpt_path = config.MODELS_DIR / f"{experiment}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device), ckpt


def evaluate_experiment(experiment: str) -> dict:
    """Evaluate a trained model on the full DDI set; save `<exp>_metrics.json`."""
    set_seed()
    device = get_device()
    model, _ = load_experiment_model(experiment, device)
    loader = build_ddi_loader(get_preprocess_transform())
    preds = run_inference(model, loader, device)
    overall = classification_metrics(preds["labels"], preds["preds"], preds["probs"])

    result = {"experiment": experiment, "overall": overall, "predictions": preds}
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.METRICS_DIR / f"{experiment}_metrics.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Saved DDI metrics to %s", out_path)
    return result
