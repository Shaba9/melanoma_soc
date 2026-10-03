# DATA-SPEC.md

# Dataset Specification

## Project
Improving Melanoma Classification Through Skin-Tone-Aware Image Augmentation

## Primary Dataset
HAM10000

## Optional Dataset
DDI (only if access is approved)

## Required Downloads
- HAM10000_images_part_1.zip
- HAM10000_images_part_2.zip
- HAM10000_metadata.tab

## Expected Structure
```text
project/
└── datasets/
    └── ham10000/
        ├── HAM10000_metadata.tab
        └── images/
            ├── ISIC_xxx.jpg
```

## Metadata Columns Used
Required:
- image_id
- dx

Optional:
- age
- sex
- localization
- dx_type

## Label Mapping
Positive (1): mel
Negative (0): nv, bkl, bcc, akiec, df, vasc

```python
label = 1 if dx == 'mel' else 0
```

## Image Processing
- RGB conversion
- Resize 224x224
- ImageNet normalization

Mean=[0.485,0.456,0.406]
Std=[0.229,0.224,0.225]

## Splits
- Train 70%
- Validation 15%
- Test 15%
- Stratified by label
- Random seed 42

## Class Imbalance Requirements
Implement at least one:
- Weighted CrossEntropyLoss
- WeightedRandomSampler

Record class counts before training.

## Augmentation Dataset
Training only:
- Horizontal Flip
- Rotation ±15°
- Brightness 0.2
- Contrast 0.2
- Saturation 0.2
- Hue 0.05

Never augment validation or test data.

## Required Outputs
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion Matrix
- Training Curves
- Classification Report

## Optional DDI Evaluation
If DDI available:
- Read ddi_metadata.csv
- Evaluate by Fitzpatrick groups
- Compare subgroup metrics
