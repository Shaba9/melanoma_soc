# GRADCAM-SPEC — Explainability Specification

## 1. Goal
Produce Grad-CAM heatmaps that localize image regions driving the Malignant/Benign prediction, overlaid on the input lesion image.

## 2. Method
- Grad-CAM on EfficientNet-B0.
- **Target layer**: last conv block `model.features[-1]`.
- Target class: predicted class (or Malignant logit).
- Library: `pytorch-grad-cam` (preferred) or custom forward/backward hooks.

## 3. Pipeline
1. Preprocess input image ([DATA-SPEC.md](DATA-SPEC.md)).
2. Forward pass, get prediction + probability.
3. Compute Grad-CAM weights → coarse heatmap.
4. Upsample heatmap to 224×224, normalize to [0,1].
5. Apply colormap (JET) and alpha-blend over the original image (alpha ≈ 0.4).
6. Return/save overlay PNG.

## 4. Outputs
- API `/heatmap` returns overlay image (see [API-SPEC.md](API-SPEC.md)).
- "Grad-CAM Examples" figure: several lesions with overlays, ideally across skin tones and both classes ([FIGURE-SPEC.md](FIGURE-SPEC.md)).

## 5. Interpretation Notes
- Grad-CAM is **localization**, not segmentation; it indicates influential regions only.
  Stated in [LIMITATIONS-SPEC.md](LIMITATIONS-SPEC.md).
