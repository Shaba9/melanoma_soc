# IMPLEMENTATION-SPEC

## Project
Improving Melanoma Classification Through Skin-Tone-Aware Image Augmentation

## 1. Project Goal
Build a web application that uploads skin lesion images, classifies Melanoma vs Not Melanoma using EfficientNet-B0, generates Grad-CAM visualizations, and compares baseline vs augmented training experiments.

## 2. Dataset Requirements
Primary Dataset: HAM10000
Required Files:
- HAM10000_images_part_1.zip
- HAM10000_images_part_2.zip
- HAM10000_metadata.tab

Optional Dataset:
- DDI (if access approved)

## 3. Dataset Structure
```text
project/
├── datasets/
│   └── ham10000/
│       ├── HAM10000_metadata.tab
│       └── images/
```

## 4. Metadata Processing
Required columns:
- image_id
- dx

Label mapping:
```python
label = 1 if dx == 'mel' else 0
```

## 5. Data Split
- Train 70%
- Validation 15%
- Test 15%
- Stratified
- Seed 42

## 6. Class Imbalance
Use Weighted CrossEntropyLoss or WeightedRandomSampler.

## 7. Baseline Transforms
Resize(224,224)
Normalize(ImageNet)

## 8. Augmented Transforms
RandomHorizontalFlip(0.5)
RandomRotation(15)
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

## 9. Model
EfficientNet-B0
IMAGENET1K_V1 weights
2-class output layer

## 10. Training
Adam
LR=1e-4
Batch Size=32
Epochs=20
EarlyStopping=3

Save:
- best_model.pt
- final_model.pt

## 11. Metrics
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion Matrix

## 12. Exports
baseline_metrics.csv
augmented_metrics.csv
predictions.csv

## 13. Grad-CAM
Use pytorch-grad-cam.
Save original, heatmap, overlay images.

## 14. FastAPI Endpoints
POST /predict
GET /metrics
POST /train/baseline
POST /train/augmented

## 15. Frontend
React + Vite
Pages:
- Upload
- Results
- Dashboard

## 16. Deliverables
- Web app
- Baseline model
- Augmented model
- Metrics report
- Grad-CAM visualizations
- IEEE report
