# EXP-SPEC

## Research Question
Does skin-tone-aware augmentation improve melanoma classification performance using a pretrained model?

## Hypothesis
H0: No improvement.
H1: Augmentation improves robustness and classification metrics.

## Dataset
Primary: HAM10000
Optional: DDI

## Experiment A - Baseline
Transforms:
- Resize
- Normalize

Model:
- EfficientNet-B0

Training:
- Adam
- LR=1e-4
- Batch size=32
- Epochs=10-20
- Early stopping patience=3

Outputs:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion Matrix

## Experiment B - Augmented
Same settings as Experiment A.

Extra transforms:
```python
RandomHorizontalFlip(0.5)
RandomRotation(15)
ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.05)
```

## Explainability
Grad-CAM required.
Generate examples for:
- True Positive
- True Negative
- False Positive
- False Negative

## Figures
1. System architecture
2. Dataset summary
3. Training loss curves
4. Validation curves
5. ROC curves
6. Confusion matrices
7. Grad-CAM examples
8. Sample augmentations

## CSV Exports
baseline_metrics.csv
augmented_metrics.csv
predictions.csv

## Model Checkpoints
Save:
- best_model.pt
- final_model.pt

## Optional Fairness Analysis
Only if DDI available.
Compare subgroup metrics by Fitzpatrick category.
