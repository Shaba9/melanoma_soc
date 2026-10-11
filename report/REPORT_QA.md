# Report Q&A — Outline for Final Report

## 1. Which ML model should I use? Why?

EfficientNet-B0 — strong balance of accuracy, computational efficiency, and ease of
implementation. Widely used in medical image classification, supports transfer learning
(ImageNet pretrained), and reaches competitive performance without heavy compute. In this
repo it is loaded from torchvision with the final classifier replaced by a 2-class head
(`Linear(1280 → 2)`); the ImageNet backbone is reused and fine-tuned.

## 2. Why Grad-CAM heatmap?

Grad-CAM provides model interpretability by highlighting the image regions that most
influenced a prediction. It demonstrates the computational-imaging / explainability
component and helps verify the model focuses on the lesion rather than background, without
needing a full segmentation model.

## 3. Why the DDI dataset for evaluation?

Released publicly by the Stanford group (Daneshjou et al., *Science Advances*, 2022) as part
of their study on dermatology-AI disparities. It is biopsy-confirmed and deliberately spans
diverse skin tones (Fitzpatrick I–VI), so results can be compared directly against published
fairness findings. The reviewed literature is built on DDI.

## 4. Why the HAM10000 dataset for training?

Publicly available dermatology dataset of 10,015 dermoscopic lesion images with biopsy-
confirmed malignancies. Large enough to fine-tune and stabilize training, and reproducible.
(Note: HAM10000 is dermoscopic and lighter-skinned on average; DDI is clinical photography —
a deliberate train/test domain shift, see Limitations.)

## 5. What image-augmentation approach?

Photometric (color-based) augmentation plus light geometric jitter, applied to the **training
set only** in the augmented experiment:
- Brightness, contrast, saturation, hue (`ColorJitter`)
- Random horizontal flip, random rotation (±20°)

Rationale: simulate lighting/appearance variability as a proxy for darker-skin appearance,
since HAM10000 lacks skin-tone labels.

## 6. What is the experiment process?

1. Fine-tune EfficientNet-B0 on original HAM10000 (baseline).
2. Evaluate on DDI; record metrics.
3. Apply color + geometric augmentations to training data.
4. Retrain on the augmented data (same eval procedure).
5. Evaluate on DDI; record metrics.
6. Generate Grad-CAM visualizations.
7. Compare baseline vs augmented (performance + fairness).
8. Compare against published dermatology-fairness literature (DDI study).

## 7. How do I evaluate the model?

Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, and **accuracy by skin-tone
group** (Light / Medium / Dark) plus a Light–Dark fairness gap. Grad-CAM overlays provide
qualitative explainability. Together these cover predictive performance, fairness, and
interpretability.

## 8. AUGMENTATION-SPEC transform/parameters chart — valid for darkening?

The chart gives each transform and its jitter magnitude. For `ColorJitter(brightness=0.2)`,
each image gets a random brightness factor `b ~ Uniform(0.8, 1.2)`; `b < 1` darkens, `b > 1`
brightens, so not every image is adjusted the same way (variability during training).
These are valid parameters for *mild* darkening. **⚠ Caveat:** ±0.2 only darkens up to ~20%,
a weak proxy for true dark-skin appearance — worth noting as a limitation.

## 9. FAIRNESS-SPEC grouping (Light / Medium / Dark)

Pre-determined from the DDI dataset's Fitzpatrick labels, mapped to three groups:
Light = FST I–II (code 12), Medium = FST III–IV (code 34), Dark = FST V–VI (code 56). The
project does **not** create new labels or assign Fitzpatrick scores itself — it reuses the
dataset's `skin_tone` field (see `SKIN_TONE_MAP` in `src/config.py`).

## 10. LIMITATIONS — why heatmap, not segmentation? What would segmentation take?

**Why not segmentation.** The research question is a *classification* problem — is a lesion
benign or malignant — not a localization problem. Segmentation answers a different question
(*which exact pixels belong to the lesion*), which is not required to produce or explain a
benign/malignant decision. More importantly, segmentation is **supervised**: it requires
ground-truth lesion masks to train against, and neither HAM10000 nor DDI provides them.
Creating those masks would mean having clinicians hand-trace lesion boundaries on thousands
of images — a large, costly annotation effort outside this project's scope and timeline.
Grad-CAM gives the explainability we need (showing the model attends to the lesion, not
background or artifacts) for free, directly from the trained classifier, with no extra labels.

**What implementing segmentation would take.**
1. **Mask labels:** pixel-level lesion masks for every training image (expert annotation, or a
   dataset that already ships masks — e.g., ISIC/PH2 provide lesion masks; HAM10000 and DDI do
   not).
2. **A segmentation architecture:** an encoder–decoder such as U-Net, DeepLab, or Mask R-CNN
   that outputs a per-pixel mask instead of a single class.
3. **A segmentation loss + metrics:** pixel losses like Dice or IoU/Jaccard (plus boundary or
   cross-entropy terms), evaluated with Dice/IoU rather than accuracy/F1.
4. **A different pipeline:** mask-aware data loading and augmentation (transform the mask
   alongside the image), plus optionally a two-stage design (segment the lesion, then classify
   the cropped region).

Grad-CAM therefore shows *attention* (influential regions), not precise lesion boundaries;
segmentation would add boundary precision at a substantial data-and-engineering cost.

---

## 11. What is the code doing? (high-level, computational-imaging focus)

Pipeline: load metadata → transform images → fine-tune EfficientNet-B0 on HAM10000 →
evaluate on DDI → compute fairness → generate Grad-CAM → serve via API/UI.

- **Data** (`src/data/`): `loaders.py` maps metadata to binary labels and image paths, does a
  lesion-safe 80/20 HAM split, and builds DataLoaders. `transforms.py` defines the
  preprocessing and augmentation pipelines.
- **Image augmentation** (`src/data/transforms.py`, `get_augmented_transform`): a
  `torchvision.transforms.Compose` of `RandomHorizontalFlip(0.5)` → `RandomRotation(20)` →
  `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)` → `Resize(224)` →
  `ToTensor` → `Normalize(ImageNet mean/std)`. Baseline uses the same minus the random ops.
- **Model** (`src/models/efficientnet.py`): `efficientnet_b0(weights=DEFAULT)` with the head
  swapped to `Linear(1280 → 2)`.
- **Training** (`src/training/train.py`): class-weighted cross-entropy, Adam (lr 1e-4),
  `ReduceLROnPlateau`, checkpoints best model by validation F1, early-stops when val F1 stops
  improving.
- **Evaluation** (`src/evaluation/evaluate.py`): runs inference over all DDI images, computes
  overall metrics, and saves per-image predictions with skin-tone groups.
- **Grad-CAM** (`src/gradcam/gradcam.py`): see §13.
- **API/UI** (`src/api/`, `frontend/`): FastAPI serves predictions, heatmaps, metrics, and
  figures; React UI consumes them (see §16).

## 12. Math/concepts for darkening the training data

`ColorJitter` samples factors from uniform ranges and applies, in random order:
- **Brightness:** $\text{out} = b \cdot \text{in}$, $b \sim U(0.8, 1.2)$. $b<1$ darkens.
- **Contrast:** $\text{out} = c\,(\text{in} - \bar{g}) + \bar{g}$, $c \sim U(0.8, 1.2)$, where
  $\bar{g}$ is the mean gray (luminance) of the image.
- **Saturation:** $\text{out} = s\,\text{in} + (1-s)\,\text{gray}$, $s \sim U(0.8, 1.2)$
  (blend between grayscale and original).
- **Hue:** convert RGB→HSV, shift $H$ by $h \sim U(-0.1, 0.1)$ of the full circle
  (≈ ±36°), convert back.

Images are then resized to 224×224 and normalized:
$\hat{x} = (x - \mu)/\sigma$ with ImageNet $\mu,\sigma$. Darkening comes specifically from the
brightness factor $b<1$ (and, secondarily, reduced contrast/saturation).

## 13. Math/concepts for the Grad-CAM heatmap

For target class $c$ and the last conv block's activation maps $A^k$ (channel $k$):

1. Backprop the class score $y^c$ to get gradients $\partial y^c / \partial A^k_{ij}$.
2. Channel weights = global-average-pool of gradients:
   $\alpha_k^c = \frac{1}{Z}\sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}$.
3. Weighted combination + ReLU:
   $L^c = \mathrm{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)$.
4. Bilinearly upsample $L^c$ to 224×224 and min–max normalize to $[0,1]$.
5. Overlay: alpha-blend a JET colormap of the heatmap over the image,
   $\text{out} = (1-\alpha)\,\text{image} + \alpha\,\text{jet}(L^c)$, with $\alpha = 0.4$.

In code: forward/backward hooks on `model.features[-1]` capture activations/gradients;
`weights = gradients.mean(dim=(2,3))`; `cam = relu((weights*activations).sum(1))`;
`F.interpolate(...)`; normalize; `Image.blend`.

## 14. How do I interpret the results? What are the output files?

All artifacts land under `artifacts/`:
- `artifacts/models/<exp>_best.pt` — trained checkpoint (best val-F1 weights + metadata).
- `artifacts/metrics/<exp>_metrics.json` — overall DDI metrics + per-image predictions
  (label, pred, P(malignant), skin-tone group). Source of truth for tables/figures/fairness.
- `artifacts/tables/` — `dataset_summary`, `<exp>_metrics`, `skin_tone_metrics`,
  `fairness_gap` as both `.csv` and `.md`.
- `artifacts/figures/` (300 DPI PNG) — `architecture`, `augmentation_examples`,
  `training_curves`, `roc_curve`, `confusion_matrices`, `gradcam_examples`.
- `artifacts/logs/` — run logs.

Interpretation: higher Recall/F1/ROC-AUC = better malignant detection; the confusion matrix
shows error types; per-skin-tone accuracy + the Light–Dark gap show fairness; Grad-CAM shows
*where* the model looked.

## 15. Augmentation experiment — how does the model avoid augmenting existing skin-of-color images?

The premise doesn't apply: augmentation is applied uniformly and randomly to *every*
training image; it does not detect or skip skin-of-color images. HAM10000 has **no skin-tone
labels**, so there is nothing to target or exclude — and it contains very few dark-skin
images to begin with. DDI (the evaluation set) is **never** augmented. The augmentation is a
blanket photometric proxy, not a skin-tone-aware operation.

## 16. Significance of the UI; what each page does; terminal-only workflow?

The UI is a demonstration/validation layer; it does not train. Pages:
- **Upload** — drag/drop or pick an image, choose model (baseline/augmented), run analysis.
- **Results** — original image vs Grad-CAM overlay, predicted class + P(malignant).
- **Metrics** — per-experiment metric cards, confusion matrix, per-skin-tone accuracy chart,
  fairness gap, and the figures gallery.

Yes — you can run everything from the terminal (`python run_experiment.py --exp ...`) and the
experiment results are written to `artifacts/`. The UI simply reads those artifacts through
the FastAPI backend. The UI adds interactive single-image inference + a visual dashboard but
is optional for producing/viewing results.

## 17. Why train the baseline at all?

A plain ImageNet model recognizes generic objects, not lesions. Fine-tuning on dermatology
images teaches lesion-specific features for benign-vs-malignant classification. The baseline
(no augmentation) is the control against which the augmentation experiment is measured.

## 18. How is the training data split? Why did the terminal show 251 / 63?

The split is the **full HAM10000**, not a 251/63 subset. The 251 and 63 you see in the terminal progress bar are **batch
counts**, not image counts: the tqdm progress bar is labeled `unit="batch"`, and with
`BATCH_SIZE = 32`:
- Training: 8,012 images ÷ 32 = **251 batches/epoch**.
- Validation: 2,003 images ÷ 32 = **63 batches/epoch**.

So each epoch really processes all 8,012 train / 2,003 val images — it just advances the bar
once per 32-image batch. The split itself is a stratified 80/20 split by `lesion_id` (to
prevent lesion leakage), seed 42; DDI (656 images) is the held-out test set
(`artifacts/tables/dataset_summary.md`). Each epoch is slow because it is CPU-bound, not
because the dataset is small.

## 19. How does the model work? What does importing a pretrained model mean? Any changes after the experiments?

EfficientNet-B0 is a CNN that learns hierarchical image features (edges → textures → lesion
structure). "Importing a pretrained model" means loading weights already trained on ImageNet
(general visual features) and reusing them as a starting point (transfer learning). We replace
the final classifier with a 2-class head and fine-tune on melanoma data (Benign / Malignant).
No architectural changes were made after the experiments — only fine-tuning and training on
domain data.

## 20. Original DDI study scores; comparison to my experiment

From Daneshjou et al. (2022), ROC-AUC on DDI:

| Model (study) | Whole DDI | Light (FST I–II) | Dark (FST V–VI) |
|---|---|---|---|
| HAM10000 (baseline) | 0.67 | 0.72 | 0.57 |
| DeepDerm (baseline) | 0.56 | 0.61 | 0.50 |
| ModelDerm (baseline) | 0.65 | 0.64 | 0.55 |
| HAM10000 **fine-tuned on DDI** | — | **0.77** | **0.78** |
| DeepDerm **fine-tuned on DDI** | — | 0.73 | 0.76 |

My experiment (EfficientNet-B0 on DDI): baseline ROC-AUC **0.554**, augmented **0.615**.

- **Methodology differences:** Study used Inception V3 at 299×299, balanced sampling, Adam
  lr 1e-4, augmentation (rotation/flip/crop); fairness improvement came from **fine-tuning on
  DDI itself** (60-20-20 split, Adam lr 5e-2, weight decay 1e-4, color jitter + Gaussian blur
  + Mixup, **up to 500 epochs**, 20 seeds, val-loss model selection). My setup: EfficientNet-B0
  at 224×224, class-weighted CE, Adam lr 1e-4, early stopping on val F1, single seed, and —
  critically — **my augmented model is never trained on DDI**; it uses photometric
  augmentation of HAM10000 as a proxy.
- **Why my baseline AUC (0.554) < study's HAM10000 (0.67):** different architecture, single
  seed, early stopping, class-weighted loss vs balanced sampling, different preprocessing/
  decision threshold, and no DDI fine-tuning. The study's big gains (≈0.77–0.78) came from
  fine-tuning *on DDI*, which my design deliberately avoids, so I should not expect to match
  them.
- **Baseline epochs (study):** not explicitly stated for the base models; the fine-tuning
  used up to 500 epochs. My baseline early-stopped around epoch 6 (best at epoch 3).

## 21. Final results — was augmentation successful? Limitations? Future work?

**Results (DDI):**

| Metric | Baseline | Augmented |
|---|---|---|
| Accuracy | 0.745 | 0.741 |
| Precision | 1.000 | 0.512 |
| Recall | 0.023 | 0.123 |
| F1 | 0.046 | 0.198 |
| ROC-AUC | 0.554 | 0.615 |

**Fairness (accuracy by group):**

| Model | Light | Medium | Dark | Light−Dark gap | Max−Min |
|---|---|---|---|---|---|
| Baseline | 0.764 | 0.701 | 0.778 | −0.013 | 0.077 |
| Augmented | 0.774 | 0.726 | 0.725 | +0.049 | 0.049 |

**Verdict: partially successful.** Augmentation clearly improved malignant detection
(Recall 0.023→0.123, F1 0.046→0.198, ROC-AUC 0.554→0.615) and **narrowed the overall max−min
accuracy spread** (0.077→0.049). However, it did **not** reduce the primary Light–Dark gap —
Dark-group accuracy actually fell (0.778→0.725) while Light rose, so on that specific metric
fairness slightly worsened. So the fairness hypothesis is only partially supported.

**Likely reasons / limitations:**
- Photometric jitter (±0.2) is a weak proxy for real dark-skin imagery; it changes pixel
  intensities but not lesion morphology or clinical-photo characteristics.
- Train/test domain shift (HAM10000 dermoscopic → DDI clinical photos).
- No skin-tone labels in training, so fairness can't be optimized directly — only measured.
- Small per-group DDI counts (≈207–241) → high-variance fairness estimates.
- Single seed; severe class imbalance; CPU-limited training.

**Future work:** fine-tune on genuinely diverse data (as the DDI study did) or add synthetic
dark-skin generation; stronger/targeted augmentation + Mixup; multi-seed runs with CIs; train
on the full dataset on GPU; explore fairness-aware losses (GroupDRO/CORAL); larger backbones.
Look into datasets that provide pixel-level lesion masks (e.g., ISIC/PH2) so segmentation can
be used for more precise lesion localization and classification instead of the Grad-CAM
heatmap.

## 22. Acknowledgements

This was a solo project (no other team members). Tools and resources used:
- **Claude (Anthropic)** — AI coding assistant used for implementation, debugging, and drafting.
- **GitHub** — version control and code hosting.
- **PyTorch / torchvision** — model, training, and augmentation framework (EfficientNet-B0,
  ImageNet pretrained weights).
- **scikit-learn** — evaluation metrics (ROC-AUC, confusion matrix).
- **FastAPI + Uvicorn** — backend API; **React + Vite** — frontend UI.
- **matplotlib / pandas** — figure and table generation.
- **Datasets:** HAM10000 (Tschandl et al., 2018) and DDI (Daneshjou et al., 2022).

## 23. Methodology outline

- **Dataset selection:** DDI reviewed in the literature; HAM10000 chosen for training.
- **Data preparation:** RGB convert, resize 224×224, ImageNet normalize; binary label
  (mel → Malignant); lesion-safe stratified 80/20 train/val split.
- **Baseline experiment:** fine-tune EfficientNet-B0 (ImageNet init, 2-class head),
  class-weighted CE, Adam lr 1e-4; checkpoint best val-F1; early stop on val F1.
- **Augmentation experiment:** add flip, rotation, and brightness/contrast/saturation/hue
  jitter to training only; retrain with identical evaluation.
- **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, per-skin-tone
  accuracy + Light–Dark gap on all 656 DDI images; Grad-CAM overlays.
- **Early stopping:** stop when validation F1 stops improving (baseline stopped ≈epoch 6,
  best at epoch 3) — guards against overfitting.
- **Computational cost:** CPU-bound; each full epoch is slow (~tens of minutes).
- **Comparison:** baseline vs augmented on performance and fairness; contextualized against
  the DDI study.


