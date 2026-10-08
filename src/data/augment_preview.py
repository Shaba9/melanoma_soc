"""Augmentation preview figure per AUGMENTATION-SPEC.md / FIGURE-SPEC.md.

Renders one training lesion as original + several augmented variants and saves
``artifacts/figures/augmentation_examples.png``.

Run with: ``python -m src.data.augment_preview``.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .. import config
from .loaders import load_ham_metadata
from .transforms import get_augmented_preview_transform

N_VARIANTS = 5


def generate(seed: int = config.SEED, out_path=None):
    import torch

    out_path = out_path or (config.FIGURES_DIR / "augmentation_examples.png")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_ham_metadata()
    # Deterministic sample: first malignant lesion if present, else first row.
    mel = df[df["label"] == config.MALIGNANT]
    row = (mel.iloc[0] if len(mel) else df.iloc[0])
    image = Image.open(row["image_path"]).convert("RGB")

    preview_tf = get_augmented_preview_transform()
    torch.manual_seed(seed)

    fig, axes = plt.subplots(1, N_VARIANTS + 1, figsize=(3 * (N_VARIANTS + 1), 3))
    axes[0].imshow(image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))
    axes[0].set_title("Original")
    axes[0].axis("off")
    for i in range(1, N_VARIANTS + 1):
        axes[i].imshow(preview_tf(image))
        axes[i].set_title(f"Augmented {i}")
        axes[i].axis("off")

    fig.suptitle("Augmentation Examples (flip, rotation, color jitter)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    path = generate()
    print(f"Augmentation preview written to {path}")


if __name__ == "__main__":
    main()
