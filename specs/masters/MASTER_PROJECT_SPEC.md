# MASTER_PROJECT_SPEC

Project: Improving Malignant-vs-Benign Skin Lesion Classification Across Diverse Skin Tones Through Image Augmentation

## Goal
Build a web-based computational imaging application using EfficientNet-B0, HAM10000 for training, DDI for evaluation, Grad-CAM explainability, fairness evaluation by skin tone, and IEEE-report outputs.

Refer to the documents in the /rubrics directory for guidelines on final report structure.

Refer to the /data directory for the HAM10000 and DDI dataset structures.

## Datasets
### HAM10000
Training dataset.
Schema:
- lesion_id
- image_id
- dx
- dx_type
- age
- sex
- localization
- dataset

Binary mapping:
- mel => Malignant
- all remaining classes => Benign

### DDI
Evaluation dataset.
Schema:
- DDI_ID
- DDI_file
- skin_tone
- malignant
- disease

Skin tone mapping:
- 12 = Light
- 34 = Medium
- 56 = Dark

Labels:
- malignant=True => Malignant
- malignant=False => Benign

## Experiments
### Baseline
EfficientNet-B0 pretrained.
Train on HAM10000.
No augmentation.
Evaluate on DDI.

### Augmented
EfficientNet-B0 pretrained.
Train on HAM10000 with:
- ColorJitter
- Brightness
- Contrast
- Saturation
- Hue
- Horizontal Flip
- Rotation

Evaluate on DDI.

## Metrics
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion Matrix

Fairness:
- Accuracy by skin tone
- Light vs Dark performance gap

## Computational Imaging Components
- Preprocessing
- Classification
- Grad-CAM localization
- Explainability overlays

## Frontend
- React
- Vite
- Upload page
- Results page
- Metrics dashboard

## Backend
- FastAPI
- Prediction endpoint
- Heatmap endpoint
- Metrics endpoint

## Required Outputs
Figures:
- Architecture
- Augmentation Examples
- Training Curves
- ROC Curve
- Confusion Matrices
- Grad-CAM Examples

Tables:
- Dataset Summary
- Baseline Metrics
- Augmented Metrics
- Skin Tone Metrics
- Fairness Gap Analysis

## IEEE Report Sections
1. Abstract
2. Introduction
3. Related Work
4. Methods
5. Experimental Setup
6. Results
7. Discussion
8. Limitations
9. Conclusion
10. References

## Limitations
- Different source datasets
- No skin-tone labels in HAM10000
- Photometric augmentation is not real SOC data
- DDI is relatively small
- Grad-CAM is localization not segmentation

## Generate These Files
- TRD.md
- DATA-SPEC.md
- EXP-SPEC.md
- IMPLEMENTATION.md
- API-SPEC.md
- UI-SPEC.md
- MODEL-SPEC.md
- AUGMENTATION-SPEC.md
- FAIRNESS-SPEC.md
- GRADCAM-SPEC.md
- REPO-SPEC.md
- REPORT-SPEC.md
- FIGURE-SPEC.md
- TABLE-SPEC.md
- LIMITATIONS-SPEC.md
- IEEE-PAPER-SPEC.md
- PROMPTS.md
