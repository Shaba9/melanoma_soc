# DATA-SPEC.md

# Dataset Specification

## Project
Improving Melanoma Classification Across Diverse Skin Tones Through Skin-Tone-Aware Image Augmentation

## Primary Dataset
DDI (Diverse Dermatology Images)

## Source
Stanford Center for Artificial Intelligence in Medicine & Imaging (AIMI)

## Dataset Purpose
Evaluate melanoma/skin lesion classification performance across diverse skin tones.

## Dataset Contents
- Clinical skin lesion images (.png)
- Metadata file (.csv)
- Diagnosis labels
- Fitzpatrick skin type annotations

## Dataset Size
- 656 images
- 570 unique patients

## Expected Directory Structure

DDI/
├── metadata.csv
├── 000001.png
├── 000002.png
├── ...

## Fields To Extract

Required:
- image_id
- diagnosis
- benign_or_malignant
- fitzpatrick_skin_type

Optional:
- age
- sex
- anatomical_site

## Label Mapping

Binary Classification

Class 0 = Benign
Class 1 = Malignant / Melanoma

## Skin Tone Mapping

Light Skin Group
- Fitzpatrick I
- Fitzpatrick II

Medium Skin Group (optional)
- Fitzpatrick III
- Fitzpatrick IV

Dark Skin Group
- Fitzpatrick V
- Fitzpatrick VI

Recommended Fairness Evaluation

Group A = Fitzpatrick I-II
Group B = Fitzpatrick V-VI

## Train/Validation/Test Split

- Train = 70%
- Validation = 15%
- Test = 15%

Random Seed = 42

Use stratified split.

## Preprocessing

- RGB format
- Resize 224x224
- Normalize using ImageNet statistics

## Baseline Dataset Version

No augmentation.

## Experimental Dataset Version

Runtime augmentation only.

Images should not be permanently modified.

Augmentations:
- Horizontal Flip
- Rotation
- Brightness
- Contrast
- Saturation
- Hue
- ColorJitter

## Fairness Evaluation Outputs

For each skin-tone group:

- Accuracy
- Precision
- Recall
- F1

Performance Gap:

Gap = Accuracy(FST I-II) - Accuracy(FST V-VI)

Success Criterion:

Reduced gap after augmentation.
