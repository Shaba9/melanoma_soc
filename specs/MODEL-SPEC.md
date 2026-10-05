# MODEL-SPEC — Model Specification

## 1. Architecture
- **Backbone**: EfficientNet-B0, pretrained on ImageNet (`torchvision.models.efficientnet_b0`).
- **Head**: replace classifier with `Linear(1280 → 2)` (or `→ 1` with sigmoid). Use 2-class
  softmax for consistency across metrics.
- **Input**: 224×224×3 normalized tensor.
- **Output**: logits → softmax → P(Malignant).

## 2. Training Configuration
| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 32 |
| Epochs | 15 (early stop on val F1, patience 3) |
| Loss | Cross-entropy with class weights |
| Scheduler | ReduceLROnPlateau (optional) |
| Seed | 42 |
| Device | CUDA if available else CPU |

## 3. Class Weighting
Compute weights inversely proportional to class frequency in HAM10000 train split,
passed to `CrossEntropyLoss(weight=...)`.

## 4. Fine-Tuning Strategy
- Train the full network (backbone + head). Pretrained weights as initialization.
- Optionally freeze early blocks for the first epoch; not required.

## 5. Checkpointing
- Save best model by validation F1 to `artifacts/models/<experiment>_best.pt`.
- Save final metrics and config alongside the checkpoint.

## 6. Two Models
- `baseline`: trained per [AUGMENTATION-SPEC.md](AUGMENTATION-SPEC.md) with augmentation disabled.
- `augmented`: trained with the augmentation pipeline enabled.
Both share identical architecture and hyperparameters; only augmentation differs.

## 7. Inference
- Load checkpoint, eval mode, apply preprocessing from [DATA-SPEC.md](DATA-SPEC.md).
- Return predicted class + probability.

## 8. Grad-CAM Target Layer
- Target the last convolutional block (`features[-1]`) for Grad-CAM
  (see [GRADCAM-SPEC.md](GRADCAM-SPEC.md)).
