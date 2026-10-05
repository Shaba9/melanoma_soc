# TABLE-SPEC — Table Specification

All tables saved to `artifacts/tables/` (CSV + Markdown) via `src/viz/tables.py`.

## 1. Dataset Summary
Columns: Dataset, Total Images, Malignant, Benign, Skin-tone breakdown (DDI only).
Rows: HAM10000 (train/val), DDI (Light/Medium/Dark).

## 2. Baseline Metrics
Columns: Metric, Value. Rows: Accuracy, Precision, Recall, F1, ROC-AUC (on DDI).

## 3. Augmented Metrics
Same structure as Baseline, for the augmented model.

## 4. Skin Tone Metrics
Columns: Skin Tone, Count, Accuracy, Precision, Recall, F1.
One block per model (baseline, augmented).

## 5. Fairness Gap Analysis
Columns: Model, Acc(Light), Acc(Medium), Acc(Dark), Light−Dark Gap, Max−Min Gap.
Rows: Baseline, Augmented.

## Requirements
- Values sourced from `artifacts/metrics/*.json` (no manual entry).
- Round metrics to 3 decimals.
- Markdown versions embedded in the report; CSV retained for reproducibility.
- Each table referenced and discussed in [REPORT-SPEC.md](REPORT-SPEC.md).
