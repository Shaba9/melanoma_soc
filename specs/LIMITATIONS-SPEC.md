# LIMITATIONS-SPEC — Limitations Specification

Document these limitations in the report (Limitations section) and reference them where relevant in Discussion.

## 1. Different Source Datasets
HAM10000 (train) and DDI (test) differ in acquisition, population, and label distribution;
cross-dataset shift may affect absolute metrics.

## 2. No Skin-Tone Labels in HAM10000
Training data lacks skin-tone annotations, so fairness cannot be optimized directly during
training — only evaluated on DDI.

## 3. Photometric Augmentation ≠ Real SOC Data
Color/brightness jitter approximates but does not replicate real skin-of-color imagery;
it is a proxy, not ground-truth diversity.

## 4. DDI Is Small
Limited DDI size, especially per skin-tone group, yields high-variance fairness estimates;
report counts and avoid over-claiming.

## 5. Grad-CAM Is Localization, Not Segmentation
Heatmaps indicate influential regions, not precise lesion boundaries; not a diagnostic
segmentation.

## 6. Research-Only
Not a medical device; outputs must not be used for clinical diagnosis.
