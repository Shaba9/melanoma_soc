"""EfficientNet-B0 classifier per MODEL-SPEC.md."""
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

NUM_CLASSES = 2
_IN_FEATURES = 1280


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """EfficientNet-B0 with an ImageNet backbone and a `Linear(1280 -> num_classes)` head."""
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(_IN_FEATURES, num_classes)
    return model


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Last convolutional block used as the Grad-CAM target (see GRADCAM-SPEC.md)."""
    return model.features[-1]
