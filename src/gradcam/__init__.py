"""Grad-CAM explainability (GRADCAM-SPEC.md)."""
from .gradcam import (
    ALPHA,
    GradCAM,
    compute_gradcam,
    generate_examples,
    generate_overlay,
    overlay_heatmap,
)

__all__ = [
    "ALPHA",
    "GradCAM",
    "compute_gradcam",
    "generate_examples",
    "generate_overlay",
    "overlay_heatmap",
]
