# DATA-SPEC — Data Specification

## 1. Datasets

### 1.1 HAM10000 (Training)
- Location: `data/HAM10000/`
- Metadata: `HAM10000_metadata.csv`
- Images: `images/HAM10000_images_part_1/`, `images/HAM10000_images_part_2/` (`.jpg`, named `<image_id>.jpg`).
- Columns: `lesion_id, image_id, dx, dx_type, age, sex, localization, dataset`.

**Binary label mapping (`dx` → `label`):**
- `mel` → **Malignant (1)**
- all other classes (`nv, bkl, bcc, akiec, vasc, df`) → **Benign (0)**

### 1.2 DDI (Evaluation)
- Location: `data/DDI/`
- Metadata: `ddi_metadata.csv`
- Images: `images/` (filename from `DDI_file`).
- Columns: `DDI_ID, DDI_file, skin_tone, malignant, disease`.

**Label mapping:**
- `malignant == True` → **Malignant (1)**
- `malignant == False` → **Benign (0)**

**Skin-tone mapping (`skin_tone` → group):**
- `12` → **Light**
- `34` → **Medium**
- `56` → **Dark**

## 2. Preprocessing
- Resize to **224×224**.
- Convert to RGB.
- Normalize with ImageNet stats: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
- To tensor (C,H,W), float32.

## 3. Splits
- HAM10000: stratified **80/20** train/val split on binary label, seed=42.
  Split by `lesion_id` to prevent leakage (same lesion not in both splits).
- DDI: used entirely as the held-out **test** set (no training).

## 4. Class Imbalance
- HAM10000 is benign-heavy. Apply **weighted sampling** or **class-weighted loss**
  (see [MODEL-SPEC.md](MODEL-SPEC.md)).

## 5. Data Loading Contract
- `Dataset` returns `(image_tensor, label, metadata_dict)`.
- `metadata_dict` for DDI includes `skin_tone_group` for fairness analysis.
- Missing/unreadable images are logged and skipped.

## 6. Outputs
- Dataset summary table (counts per class, per dataset, per skin tone) →
  see Table "Dataset Summary" in [TABLE-SPEC.md](TABLE-SPEC.md).
