**Prompt 1**
Read the DDI metadata.csv and tell me:

1. Column names.
2. Which column contains the image paths.
3. Which column contains diagnosis labels.
4. Which column contains Fitzpatrick skin types.
5. Recommend label mappings for binary melanoma classification.

**Prompt 2**
Build a PyTorch DDI data loader using EfficientNet-B0.

Requirements:
- Train/validation/test split
- Stratified split
- Baseline transforms only
- Binary melanoma classification

**Prompt 3**
Add augmentation pipeline:

- brightness
- contrast
- saturation
- hue
- random rotation
- horizontal flip

Generate a second training experiment using identical model settings.