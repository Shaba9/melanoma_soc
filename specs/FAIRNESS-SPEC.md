# FAIRNESS-SPEC — Fairness Evaluation Specification

## 1. Goal
Quantify whether classifier performance differs across skin-tone groups on DDI, and whether
augmentation reduces that disparity.

## 2. Groups
From DDI `skin_tone` ([DATA-SPEC.md](DATA-SPEC.md)):
- Light (12), Medium (34), Dark (56).

## 3. Metrics per Group
For each group compute: Accuracy, Precision, Recall, F1 (positive class = Malignant),
and sample count.

## 4. Fairness Gap
- **Primary metric**: `gap = Accuracy(Light) − Accuracy(Dark)`.
- Report for both baseline and augmented models.
- Also report max−min accuracy across all three groups.
- **Success signal**: augmented model shows a smaller gap than baseline.

## 5. Procedure
1. Run inference on full DDI.
2. Join predictions with skin-tone groups.
3. Compute overall and per-group metrics.
4. Compute gaps for both models; tabulate side by side.

## 6. Outputs
- `artifacts/metrics/<exp>_fairness.json` (per-group metrics + gap).
- "Skin Tone Metrics" and "Fairness Gap Analysis" tables ([TABLE-SPEC.md](TABLE-SPEC.md)).
- Narrative interpretation in Results & Discussion ([REPORT-SPEC.md](REPORT-SPEC.md)).

## 7. Caveats
- DDI is small; per-group counts may be low → report counts alongside metrics and avoid
  over-claiming. Documented in [LIMITATIONS-SPEC.md](LIMITATIONS-SPEC.md).
