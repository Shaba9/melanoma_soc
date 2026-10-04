# Technical Requirements Document (TRD)
# Project: Improving Melanoma Classification Across Diverse Skin Tones Through Skin-Tone-Aware Image Augmentation

## 1. Project Objective

Develop a web-based computational imaging application that:

1. Allows a user to upload a skin lesion image.
2. Uses a pretrained melanoma classification model.
3. Predicts Melanoma vs Benign.
4. Displays prediction confidence.
5. Generates a Grad-CAM heatmap showing image regions used in classification.
6. Evaluates performance differences across skin-tone groups.
7. Tests whether skin-tone-aware augmentation improves fairness and classification performance.

---

## 2. Functional Requirements

### FR-1 Image Upload
- User can upload JPG, JPEG, PNG files.
- Validate file type and size.
- Display uploaded image preview.

### FR-2 Classification
- Use existing pretrained model.
- Output:
  - Predicted class
  - Confidence score

### FR-3 Explainability
- Generate Grad-CAM heatmap.
- Overlay heatmap on original image.
- Display resulting visualization.

### FR-4 Experiment Mode
- Run baseline model evaluation.
- Run augmented model evaluation.
- Compare performance metrics.

### FR-5 Metrics Dashboard
Display:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC (optional)

### FR-6 Fairness Evaluation
Report metrics by:
- Light skin group
- Dark skin group

Calculate:
- Accuracy gap
- Recall gap
- F1 gap

---

## 3. Non-Functional Requirements

### Performance
- Classification response under 5 seconds.

### Usability
- Single-page interface.
- Upload and prediction workflow.

### Maintainability
- Modular architecture.
- Separate model, preprocessing, augmentation, evaluation, and web UI.

### Reproducibility
- Random seed support.
- Store experiment configuration.

---

## 4. Technology Stack

### Frontend
- React
- Vite
- Material UI (optional)

### Backend
- Python
- Flask or FastAPI

### Machine Learning
- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-learn

### Visualization
- Matplotlib
- Grad-CAM

---

## 5. Datasets

### Primary Dataset
DDI (Diverse Dermatology Images)

Dataset fields:
- Image
- Diagnosis label
- Skin-tone metadata

### Optional Dataset
ISIC Dataset (future extension only)

---

## 6. Baseline Model

### Selected Model
EfficientNet-B0

Reason:
- Pretrained availability
- Lightweight
- Strong image classification performance

Implementation:

```python
from torchvision.models import efficientnet_b0
model = efficientnet_b0(weights='DEFAULT')
```

Replace final classification layer:

```python
Melanoma
Benign
```

---

## 7. Augmentation Strategy

### Baseline Training
No augmentation except resize and normalization.

### Experimental Training
Apply:

- Random Horizontal Flip
- Random Rotation
- Brightness Adjustment
- Contrast Adjustment
- Saturation Adjustment
- Hue Adjustment
- Color Jitter

Example:

```python
ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    hue=0.05
)
```

Goal:
Increase appearance diversity while preserving lesion characteristics.

---

## 8. Computational Imaging Components

### Image Preprocessing
- Resize images
- Normalize pixel values
- Data augmentation

### Classification
- EfficientNet-B0 inference

### Localization
- Grad-CAM heatmap

Output:
- Original image
- Heatmap
- Overlay image

### Explainability
Verify model focuses on lesion regions rather than artifacts.

---

## 9. Experimental Design

### Experiment A
Baseline Model

Input:
- Original DDI images

Output:
- Performance metrics

### Experiment B
Augmented Model

Input:
- DDI images with augmentation pipeline

Output:
- Performance metrics

### Comparison
Compare:
- Accuracy
- Precision
- Recall
- F1 Score
- Accuracy by Skin Tone

---

## 10. Evaluation Metrics

### Classification Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC (optional)

### Fairness Metrics
- Light Skin Accuracy
- Dark Skin Accuracy
- Performance Gap

Formula:

Performance Gap = Light Accuracy - Dark Accuracy

Target:
Reduced gap after augmentation.

---

## 11. Application Workflow

1. User uploads image.
2. Backend preprocesses image.
3. Model performs inference.
4. Prediction generated.
5. Grad-CAM heatmap generated.
6. Results displayed.

Displayed Results:
- Prediction
- Confidence
- Heatmap
- Overlay image

---

## 12. Repository Structure

```text
project/
│
├── frontend/
├── backend/
├── datasets/
├── models/
├── augmentation/
├── evaluation/
├── gradcam/
├── experiments/
├── reports/
└── README.md
```

---

## 13. Deliverables

### Deliverable 1
Web Application

### Deliverable 2
Baseline Model Results

### Deliverable 3
Augmented Model Results

### Deliverable 4
Grad-CAM Visualization Examples

### Deliverable 5
Performance Comparison Tables

### Deliverable 6
IEEE Final Report

---

## 14. Success Criteria

Success is achieved if:

1. Web application successfully classifies lesion images.
2. Grad-CAM visualizations are generated.
3. Baseline and augmented experiments are completed.
4. Metrics are reported by skin-tone group.
5. Augmentation improves overall performance and/or reduces performance disparities across skin tones.
