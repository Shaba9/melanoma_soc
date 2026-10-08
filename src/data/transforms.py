"""Preprocessing and augmentation transforms per DATA-SPEC.md and AUGMENTATION-SPEC.md."""
from torchvision import transforms

from .. import config


def get_preprocess_transform():
    """Deterministic preprocessing: resize -> tensor -> ImageNet normalize.

    Used for validation, DDI evaluation, and as the baseline training transform.
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def get_augmented_transform():
    """Augmented training transform (photometric + geometric) per AUGMENTATION-SPEC.md."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def get_train_transform(augment: bool = False):
    """Select the training transform: augmented pipeline or baseline preprocessing."""
    return get_augmented_transform() if augment else get_preprocess_transform()


def get_augmented_preview_transform():
    """Augmentation ops without ToTensor/Normalize, for visualization (returns PIL)."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    ])
