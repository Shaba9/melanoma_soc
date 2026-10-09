"""Grad-CAM explainability per GRADCAM-SPEC.md.

Custom forward/backward hooks on EfficientNet-B0's last conv block
(`model.features[-1]`). Produces a [0,1] heatmap upsampled to 224x224 and a
JET-colormap overlay alpha-blended over the input lesion. Reused by the API
`/heatmap` endpoint (Prompt 9) and the Grad-CAM Examples figure (Prompt 8).
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
from PIL import Image

from .. import config
from ..data import get_preprocess_transform
from ..models import build_model, gradcam_target_layer

logger = logging.getLogger(__name__)

ALPHA = 0.4


class GradCAM:
    """Grad-CAM for a single target layer via forward/backward hooks."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inp, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __call__(self, input_tensor: torch.Tensor, target_class: int | None = None):
        """Return (heatmap [H,W] in [0,1], predicted_class, P(Malignant))."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        prob_malignant = torch.softmax(logits, dim=1)[0, config.MALIGNANT].item()
        pred_class = int(logits.argmax(dim=1).item())
        target = pred_class if target_class is None else target_class
        logits[0, target].backward()

        # Grad-CAM weights = GAP of gradients over spatial dims.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy(), pred_class, prob_malignant


def compute_gradcam(model, input_tensor, target_class=None, target_layer=None):
    """Run Grad-CAM; return (heatmap, predicted_class, P(Malignant))."""
    layer = target_layer if target_layer is not None else gradcam_target_layer(model)
    cam = GradCAM(model, layer)
    try:
        return cam(input_tensor, target_class)
    finally:
        cam.remove()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = ALPHA) -> Image.Image:
    """Alpha-blend a JET-colored heatmap over an RGB image (both resized to 224)."""
    base = image.convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    colored = matplotlib.colormaps["jet"](heatmap)[:, :, :3]  # drop alpha channel
    heat_img = Image.fromarray((colored * 255).astype(np.uint8))
    return Image.blend(base, heat_img, alpha)


def generate_overlay(model, image: Image.Image, device, target_class=None):
    """Produce a Grad-CAM overlay for a PIL image.

    Returns (overlay PIL image, prediction dict with class index, name, prob).
    """
    tensor = get_preprocess_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    heatmap, pred_class, prob = compute_gradcam(model, tensor, target_class)
    overlay = overlay_heatmap(image, heatmap)
    prediction = {
        "predicted_class": pred_class,
        "class_name": config.CLASS_NAMES[pred_class],
        "prob_malignant": prob,
    }
    return overlay, prediction


def generate_examples(experiment: str = "augmented", per_cell: int = 1):
    """Save a Grad-CAM Examples figure across skin tones and both classes.

    Picks DDI lesions for each (skin tone x class) cell, overlays Grad-CAM from a
    trained model, and writes `artifacts/figures/gradcam_examples.png`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..data import load_ddi_metadata
    from ..evaluation import load_experiment_model
    from ..training.train import get_device, set_seed

    set_seed()
    device = get_device()
    model, _ = load_experiment_model(experiment, device)
    df = load_ddi_metadata()

    tones = list(config.SKIN_TONE_MAP.values())
    classes = [config.BENIGN, config.MALIGNANT]
    rows = [(tone, cls) for tone in tones for cls in classes]

    fig, axes = plt.subplots(len(rows), per_cell + 1,
                             figsize=(3 * (per_cell + 1), 3 * len(rows)))
    axes = np.atleast_2d(axes)

    for r, (tone, cls) in enumerate(rows):
        subset = df[(df["skin_tone_group"] == tone) & (df["label"] == cls)]
        samples = subset.sample(n=min(per_cell, len(subset)),
                                random_state=config.SEED) if len(subset) else subset
        image_path = samples.iloc[0]["image_path"] if len(samples) else None

        ax_orig = axes[r, 0]
        if image_path is not None:
            img = Image.open(image_path).convert("RGB").resize(
                (config.IMAGE_SIZE, config.IMAGE_SIZE))
            ax_orig.imshow(img)
        ax_orig.set_ylabel(f"{tone}\n{config.CLASS_NAMES[cls]}", fontsize=10)
        ax_orig.set_xticks([]); ax_orig.set_yticks([])
        if r == 0:
            ax_orig.set_title("Original")

        for c in range(per_cell):
            ax = axes[r, c + 1]
            ax.set_xticks([]); ax.set_yticks([])
            if c < len(samples):
                pil = Image.open(samples.iloc[c]["image_path"]).convert("RGB")
                overlay, pred = generate_overlay(model, pil, device)
                ax.imshow(overlay)
                ax.set_xlabel(f"{pred['class_name']} ({pred['prob_malignant']:.2f})",
                              fontsize=9)
            if r == 0:
                ax.set_title("Grad-CAM")

    fig.suptitle(f"Grad-CAM Examples ({experiment})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FIGURES_DIR / "gradcam_examples.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved Grad-CAM examples to %s", out_path)
    return out_path


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_examples()


if __name__ == "__main__":
    main()
