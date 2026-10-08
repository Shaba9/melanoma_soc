"""Preprocessing transforms per DATA-SPEC.md.

Augmented training transforms are added in a later step (AUGMENTATION-SPEC.md).
"""
from torchvision import transforms

from .. import config


def get_preprocess_transform():
    """Deterministic preprocessing: resize -> tensor -> ImageNet normalize.

    Used for validation and DDI evaluation, and as the baseline training transform.
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
