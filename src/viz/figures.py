"""Deterministic figure generation per FIGURE-SPEC.md.

All figures are rebuilt from saved artifacts (`artifacts/metrics/*.json`,
trained checkpoints) and written to `artifacts/figures/` at 300 DPI.
"""
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .. import config

logger = logging.getLogger(__name__)

EXPERIMENTS = ["baseline", "augmented"]
# Colorblind-friendly (Okabe-Ito) for baseline vs augmented.
COLORS = {"baseline": "#0072B2", "augmented": "#D55E00"}


def _load_metrics(experiment: str) -> dict:
    path = config.METRICS_DIR / f"{experiment}_metrics.json"
    return json.loads(path.read_text())


def _load_history(experiment: str) -> list:
    path = config.METRICS_DIR / f"{experiment}_history.json"
    return json.loads(path.read_text())


def _experiments_with(suffix: str) -> list:
    """Experiments that have a given artifact (e.g. '_metrics.json')."""
    return [e for e in EXPERIMENTS if (config.METRICS_DIR / f"{e}{suffix}").exists()]


def figure_architecture():
    """Fig 1: EfficientNet-B0 pipeline diagram."""
    stages = [
        "Input\n224x224x3",
        "EfficientNet-B0\nBackbone",
        "Global\nAvg Pool",
        "Linear\n1280 -> 2",
        "Softmax\nBenign / Malignant",
    ]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, len(stages) * 3)
    ax.set_ylim(0, 3)
    ax.axis("off")
    for i, label in enumerate(stages):
        x = i * 3 + 0.3
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 0.9), 2.2, 1.2, boxstyle="round,pad=0.1",
            edgecolor="black", facecolor="#E8F0FE"))
        ax.text(x + 1.1, 1.5, label, ha="center", va="center", fontsize=10)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 2.7, 1.5), xytext=(x + 2.2, 1.5),
                        arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_title("EfficientNet-B0 Classification Pipeline", fontsize=13)
    _save(fig, "architecture.png")


def figure_augmentation():
    """Fig 2: augmentation examples (delegates to the data preview generator)."""
    from ..data.augment_preview import generate
    out = generate()
    logger.info("Augmentation examples at %s", out)
    return out


def figure_training_curves():
    """Fig 3: loss & F1 vs epoch (train vs val) for each experiment."""
    exps = _experiments_with("_history.json")
    if not exps:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for exp in exps:
        history = _load_history(exp)
        epochs = [h["epoch"] for h in history]
        color = COLORS[exp]
        axes[0].plot(epochs, [h["train_loss"] for h in history],
                     color=color, linestyle="-", label=f"{exp} train")
        axes[0].plot(epochs, [h["val_loss"] for h in history],
                     color=color, linestyle="--", label=f"{exp} val")
        axes[1].plot(epochs, [h["train_f1"] for h in history],
                     color=color, linestyle="-", label=f"{exp} train")
        axes[1].plot(epochs, [h["val_f1"] for h in history],
                     color=color, linestyle="--", label=f"{exp} val")
    axes[0].set(title="Loss vs Epoch", xlabel="Epoch", ylabel="Cross-Entropy Loss")
    axes[1].set(title="F1 vs Epoch", xlabel="Epoch", ylabel="F1 Score")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    _save(fig, "training_curves.png")


def figure_roc():
    """Fig 4: ROC curves (baseline vs augmented) on DDI."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for exp in _experiments_with("_metrics.json"):
        preds = _load_metrics(exp)["predictions"]
        labels, probs = preds["labels"], preds["probs"]
        if len(set(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, color=COLORS[exp], label=f"{exp} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Chance")
    ax.set(title="ROC Curve (DDI)", xlabel="False Positive Rate",
           ylabel="True Positive Rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, "roc_curve.png")


def figure_confusion_matrices():
    """Fig 5: 2x2 confusion matrices for baseline and augmented."""
    exps = _experiments_with("_metrics.json")
    if not exps:
        return
    fig, axes = plt.subplots(1, len(exps), figsize=(5 * len(exps), 5), squeeze=False)
    for ax, exp in zip(axes[0], exps):
        cm = np.array(_load_metrics(exp)["overall"]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="black", fontsize=12)
        ax.set(title=f"{exp.capitalize()} (DDI)",
               xlabel="Predicted", ylabel="Actual",
               xticks=[0, 1], yticks=[0, 1])
        ax.set_xticklabels(config.CLASS_NAMES)
        ax.set_yticklabels(config.CLASS_NAMES)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "confusion_matrices.png")


def figure_gradcam():
    """Fig 6: Grad-CAM examples (delegates to the explainability module)."""
    from ..gradcam import generate_examples
    trained = [e for e in EXPERIMENTS if (config.MODELS_DIR / f"{e}_best.pt").exists()]
    if not trained:
        return None
    experiment = "augmented" if "augmented" in trained else trained[0]
    out = generate_examples(experiment)
    logger.info("Grad-CAM examples at %s", out)
    return out


def _save(fig, name: str):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved figure %s", out_path)
    return out_path


def generate_all(include_gradcam: bool = True):
    """Generate every figure from saved artifacts."""
    figure_architecture()
    figure_augmentation()
    figure_training_curves()
    figure_roc()
    figure_confusion_matrices()
    if include_gradcam:
        figure_gradcam()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_all()


if __name__ == "__main__":
    main()
