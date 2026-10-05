# FIGURE-SPEC — Figure Specification

All figures saved to `artifacts/figures/` at ≥300 DPI, PNG, with axis labels and captions.

| # | Figure | Source | Notes |
|---|--------|--------|-------|
| 1 | Architecture | diagram | EfficientNet-B0 pipeline: input → backbone → head → Malignant/Benign. |
| 2 | Augmentation Examples | one lesion + transforms | Show flip, rotation, color jitter variants. |
| 3 | Training Curves | train logs | Loss & F1 vs epoch, train vs val, per experiment. |
| 4 | ROC Curve | DDI eval | ROC with AUC; baseline vs augmented on one plot. |
| 5 | Confusion Matrices | DDI eval | 2×2 for baseline and augmented. |
| 6 | Grad-CAM Examples | explain module | Overlays across skin tones / both classes. |

## Requirements
- Deterministic generation from saved artifacts (`src/viz/figures.py`).
- Clear titles, axis labels with units/words (per IEEE guidance), legends.
- Colorblind-friendly palette where practical.
- Each figure referenced in the report ([REPORT-SPEC.md](REPORT-SPEC.md)).

## Mapping to Report
- Fig 1 → Methods. Fig 2 → Methods/Setup. Fig 3–5 → Results. Fig 6 → Results/Discussion.
