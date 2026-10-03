# Experiment Specification (EXP-SPEC)

# Project
Improving Melanoma Classification Across Diverse Skin Tones Through Skin-Tone-Aware Image Augmentation

## 1. Research Question

Does skin-tone-aware image augmentation improve melanoma classification performance on underrepresented skin tone groups while maintaining overall classification accuracy?

---

## 2. Hypothesis

H0: Skin-tone-aware augmentation does not significantly improve performance across skin-tone groups.

H1: Skin-tone-aware augmentation improves performance and reduces the accuracy gap between lighter and darker skin tone groups.

---

## 3. Dataset

### Primary Dataset
DDI (Diverse Dermatology Images)

### Required Metadata
- Image file
- Melanoma/benign label
- Skin tone category

### Data Split
- Train: 70%
- Validation: 15%
- Test: 15%

Use stratified splitting where possible.

Random seed: 42

---

## 4. Model Selection

### Baseline Model
EfficientNet-B0

### Training Strategy
Transfer learning.

Replace final classification head with:
- Melanoma
- Benign

---

## 5. Image Preprocessing

Applied to all images:

- Resize: 224x224
- RGB conversion
- Pixel normalization

---

## 6. Experiment A: Baseline

### Dataset
Original DDI images only.

### Augmentation
None except resize and normalization.

### Outputs
- Accuracy
- Precision
- Recall
- F1
- Confusion Matrix
- ROC-AUC (optional)

---

## 7. Experiment B: Augmented

### Dataset
DDI images with runtime augmentation.

### Augmentations
- RandomHorizontalFlip
- RandomRotation (±15°)
- Brightness adjustment
- Contrast adjustment
- Saturation adjustment
- Hue adjustment
- ColorJitter

Suggested parameters:

```python
ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    hue=0.05
)
```

---

## 8. Training Parameters

### Optimizer
Adam

### Learning Rate
0.0001

### Batch Size
16 or 32

### Epochs
10-20

### Loss Function
CrossEntropyLoss

### Early Stopping
Patience = 3

---

## 9. Fairness Evaluation

### Group A
Lighter skin-tone group

### Group B
Darker skin-tone group

Calculate separately for each group:

- Accuracy
- Precision
- Recall
- F1

### Gap Metric

Performance Gap = Light Accuracy − Dark Accuracy

Goal:
Reduce gap after augmentation.

---

## 10. Explainability Evaluation

### Technique
Grad-CAM

Generate heatmaps for:
- Correct melanoma prediction
- Correct benign prediction
- Incorrect melanoma prediction
- Incorrect benign prediction

Qualitative objective:
Verify model focuses on lesion region.

---

## 11. Required Figures

Figure 1 - System Architecture

Figure 2 - Sample Augmented Images

Figure 3 - Training/Validation Curves

Figure 4 - Baseline Confusion Matrix

Figure 5 - Augmented Confusion Matrix

Figure 6 - Grad-CAM Heatmap Examples

---

## 12. Required Tables

### Table 1
Dataset Summary

### Table 2
Overall Model Performance

| Metric | Baseline | Augmented |
|----------|----------|----------|
| Accuracy | | |
| Precision | | |
| Recall | | |
| F1 | | |

### Table 3
Performance by Skin Tone

| Group | Baseline Accuracy | Augmented Accuracy |
|---------|---------|---------|
| Light | | |
| Dark | | |

### Table 4
Performance Gap Analysis

| Metric | Baseline Gap | Augmented Gap |
|----------|----------|----------|
| Accuracy Gap | | |
| Recall Gap | | |
| F1 Gap | | |

---

## 13. Success Criteria

Minimum Success:
- Working web application
- Baseline experiment completed
- Augmented experiment completed
- Grad-CAM visualizations generated

Target Success:
- Improved overall performance and/or reduced skin-tone performance gap

Stretch Goal:
- Demonstrate measurable fairness improvement while maintaining classification performance.

---

## 14. Final IEEE Report Structure

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
