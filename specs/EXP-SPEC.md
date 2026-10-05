# EXP-SPEC — Experiment Specification

## 1. Experiments
Two experiments with identical architecture/hyperparameters, differing only in augmentation.

### 1.1 Baseline
- EfficientNet-B0 pretrained, trained on HAM10000, **no augmentation**.
- Evaluated on DDI.

### 1.2 Augmented
- EfficientNet-B0 pretrained, trained on HAM10000 **with augmentation**
  ([AUGMENTATION-SPEC.md](AUGMENTATION-SPEC.md)).
- Evaluated on DDI.

## 2. Procedure
1. Load + split HAM10000 ([DATA-SPEC.md](DATA-SPEC.md)).
2. Train model per [MODEL-SPEC.md](MODEL-SPEC.md), logging train/val loss & metrics per epoch.
3. Select best checkpoint by validation F1.
4. Evaluate on full DDI test set.
5. Compute overall and per-skin-tone metrics ([FAIRNESS-SPEC.md](FAIRNESS-SPEC.md)).
6. Persist metrics JSON, training curves, ROC, confusion matrix.

## 3. Metrics (computed on DDI)
- Accuracy, Precision, Recall, F1, ROC-AUC.
- Confusion matrix (2×2).
- Per-skin-tone accuracy (Light/Medium/Dark).
- Light-vs-Dark accuracy gap.

Positive class = **Malignant**.

## 4. Reproducibility
- Global seed 42 (python, numpy, torch).
- Config-driven (see [IMPLEMENTATION.md](IMPLEMENTATION.md)); config saved with results.

## 5. Outputs (per experiment)
```
artifacts/
  models/<exp>_best.pt
  metrics/<exp>_metrics.json
  figures/<exp>_training_curves.png
  figures/<exp>_roc.png
  figures/<exp>_confusion_matrix.png
```

## 6. Comparison Deliverables
- Baseline vs Augmented metrics tables.
- Skin-tone metrics and fairness-gap table comparing both models
  (see [TABLE-SPEC.md](TABLE-SPEC.md)).
- Interpretation feeds Results & Discussion in [REPORT-SPEC.md](REPORT-SPEC.md).
