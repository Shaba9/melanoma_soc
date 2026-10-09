"""Training loop with class weighting, checkpointing, metrics tracking, and early
stopping per MODEL-SPEC.md.
"""
import json
import logging
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .. import config
from ..data import (
    build_ham_loaders,
    compute_class_weights,
    get_preprocess_transform,
    get_train_transform,
    load_ham_metadata,
    split_ham,
)
from ..models import build_model
from .metrics import classification_metrics

logger = logging.getLogger(__name__)


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_epoch(model, loader, criterion, device, optimizer=None) -> dict:
    """Run one train (optimizer given) or eval epoch; return metrics incl. loss."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, labels_all, preds_all, probs_all = 0.0, [], [], []
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1)[:, config.MALIGNANT]
            preds = logits.argmax(dim=1)
            labels_all += labels.cpu().tolist()
            preds_all += preds.cpu().tolist()
            probs_all += probs.detach().cpu().tolist()
    metrics = classification_metrics(labels_all, preds_all, probs_all)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train_model(experiment: str, augment: bool, epochs: int = config.EPOCHS,
                patience: int = 3, lr: float = config.LEARNING_RATE):
    """Train EfficientNet-B0 on HAM10000; checkpoint best model by validation F1.

    Returns (checkpoint_path, best_val_metrics, history).
    """
    set_seed()
    device = get_device()
    logger.info("Training '%s' (augment=%s) on %s", experiment, augment, device)

    train_df, _ = split_ham(load_ham_metadata())
    class_weights = compute_class_weights(train_df).to(device)
    train_loader, val_loader = build_ham_loaders(
        get_train_transform(augment), get_preprocess_transform()
    )

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    best_f1, best_state, best_val, no_improve, history = -1.0, None, None, 0, []
    for epoch in range(1, epochs + 1):
        tr = _run_epoch(model, train_loader, criterion, device, optimizer)
        va = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(va["f1"])
        history.append({
            "epoch": epoch,
            "train_loss": tr["loss"], "train_f1": tr["f1"],
            "val_loss": va["loss"], "val_f1": va["f1"],
        })
        logger.info("Epoch %d | train_loss=%.4f val_loss=%.4f val_f1=%.4f",
                    epoch, tr["loss"], va["loss"], va["f1"])
        if va["f1"] > best_f1:
            best_f1 = va["f1"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_val = va
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d (no val F1 improvement)", epoch)
                break

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.MODELS_DIR / f"{experiment}_best.pt"
    torch.save({
        "model_state": best_state,
        "experiment": experiment,
        "augment": augment,
        "config": {
            "seed": config.SEED, "lr": lr, "batch_size": config.BATCH_SIZE,
            "epochs": epochs, "image_size": config.IMAGE_SIZE,
        },
        "val_metrics": best_val,
        "history": history,
    }, ckpt_path)
    (config.METRICS_DIR / f"{experiment}_history.json").write_text(json.dumps(history, indent=2))
    logger.info("Saved checkpoint to %s (best val F1=%.4f)", ckpt_path, best_f1)
    return ckpt_path, best_val, history
