# Improving Melanoma Classification Across Diverse Skin Tones Through Photometric Image Augmentation

**Author:** Shaba Rahavi
**Affiliation:** Department of Biomedical Engineering, University of North Dakota

## Abstract

Automated melanoma classifiers can match dermatologist-level accuracy on
curated benchmarks, yet they frequently underperform on darker skin tones because public
dermatology datasets lack skin-tone diversity. This work investigates whether skin-tone-aware
photometric image augmentation can improve classification robustness and fairness without
designing a new architecture. An EfficientNet-B0 classifier, pretrained on ImageNet and
fine-tuned on HAM10000, is evaluated as a cross-dataset generalization test on the Diverse
Dermatology Images (DDI) set, which provides Fitzpatrick skin-tone labels. Two models are
compared: a baseline trained with preprocessing only, and an augmented model trained with
brightness, contrast, saturation, hue, flip, and rotation jitter. Grad-CAM overlays provide
explainability. On DDI, augmentation improved malignant detection (F1 0.046 → 0.198, ROC-AUC
0.554 → 0.615) and narrowed the maximum–minimum per-group accuracy spread (0.077 → 0.049),
but it did not reduce the Light-versus-Dark accuracy gap. The results suggest photometric
augmentation is a weak proxy for genuinely diverse training data and that fairness gains most
likely require representative images rather than synthetic color perturbation.

*Index Terms*—melanoma classification, skin-tone fairness, data augmentation, EfficientNet,
Grad-CAM, computational imaging, medical image analysis.

---

## I. Introduction

### A. Problem and Motivation

Melanoma is one of the most severe forms of skin cancer, and early diagnosis significantly
improves patient outcomes [1]. Advances in computational imaging and deep learning have
enabled automated classification of skin lesions from clinical and dermoscopic images, and
convolutional neural networks (CNNs) have become the dominant approach, demonstrating
diagnostic performance comparable to dermatologists in some studies [1].

Despite these advances, existing skin-lesion classifiers often fail to generalize across
diverse patient populations. A major limitation is the lack of skin-tone diversity in publicly
available dermatology datasets [2]. Darker skin tones are frequently underrepresented, making
it difficult to evaluate whether models perform consistently across all populations [2]. As a
result, many systems exhibit reduced accuracy on skin-of-color (SOC) populations [2], [3].
Because patients with skin of color are already diagnosed at later stages, these disparities
carry real clinical consequences, motivating the need for more equitable dermatology AI.

Recent work shows that state-of-the-art dermatology AI models perform worse on diverse
clinical image sets containing darker skin tones, and that fine-tuning with more
representative data reduces this gap [4]. This suggests improving dataset diversity may be as
important as improving the model itself. Rather than developing a new classifier, this project
investigates whether **skin-tone-aware image augmentation** — generating realistic variation
in skin pigmentation and imaging conditions while preserving lesion characteristics — can
improve melanoma classification across skin-tone subgroups.

### B. Contributions

1. A reproducible cross-dataset pipeline that fine-tunes EfficientNet-B0 on HAM10000 and
   evaluates on the skin-tone-labeled DDI benchmark.
2. A controlled comparison of a baseline versus a photometrically augmented model on both
   overall performance and per-skin-tone fairness.
3. Grad-CAM–based explainability to verify the model attends to lesions, and a web
   application for interactive inference and results visualization.

---

## II. Related Work

**CNNs for skin-lesion classification.** Deep CNNs trained on dermoscopic datasets have
achieved dermatologist-comparable accuracy for melanoma recognition [1]. The HAM10000 dataset
[6] has become a standard training corpus of multi-source dermatoscopic images with
biopsy-confirmed malignancies.

**Dataset bias and skin-tone diversity.** Reviews of skin-lesion datasets show pervasive
under-representation of darker skin tones, limiting fairness evaluation [2]. Broader analyses
of AI in dermatology for skin of color highlight both the promise and the persistent
challenges of equitable performance [3].

**Fairness evaluation by skin tone.** The Diverse Dermatology Images (DDI) benchmark [4], [5]
was curated specifically to expose these disparities. It is biopsy-confirmed and matched
across Fitzpatrick skin types, and demonstrates that leading models — including one trained on
HAM10000 — lose substantial ROC-AUC on Fitzpatrick V–VI skin, while fine-tuning on diverse
data closes the gap [4].

**Explainability.** Grad-CAM produces class-discriminative localization maps from a trained
classifier without extra labels, and is widely used to confirm that medical-imaging models
focus on clinically relevant regions rather than artifacts [7].

---

## III. Methods

### A. Datasets and Labels

HAM10000 is used for training/validation and DDI [5] as a held-out test set — a
deliberate train/test domain shift (dermoscopic → clinical photography). Labels are binarized
to **Benign** vs **Malignant** (HAM10000 `dx == mel` → Malignant; DDI `malignant` field). DDI
Fitzpatrick codes are mapped to three groups: Light (FST I–II), Medium (FST III–IV), and Dark
(FST V–VI). Table I summarizes the splits.

**TABLE I. Dataset Summary**

| Dataset | Total | Malignant | Benign | Skin tone |
|---|---:|---:|---:|---|
| HAM10000 (train) | 8012 | 884 | 7128 | — |
| HAM10000 (val) | 2003 | 229 | 1774 | — |
| DDI (Light) | 208 | 49 | 159 | FST I–II |
| DDI (Medium) | 241 | 74 | 167 | FST III–IV |
| DDI (Dark) | 207 | 48 | 159 | FST V–VI |

Images are converted to RGB, resized to 224×224, and normalized with ImageNet statistics
(mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]). The HAM10000 train/validation split
is a stratified 80/20 split performed by **`lesion_id`** so that no lesion appears in both
splits (preventing leakage), with seed 42.

### B. Model and Training
The classifier is EfficientNet-B0 [8], initialized with weights pretrained on the ImageNet dataset [9], which enables transfer learning by leveraging visual features such as edges, textures, shapes, and color patterns learned from millions of labeled natural images. The final classification layer was then replaced with a two-class head, Linear(1280 → 2), and fine-tuned for benign-versus-malignant skin lesion classification (Fig. 1). Only the head is new; the backbone
is fine-tuned. Training uses class-weighted cross-entropy (inverse-frequency weights to
counter benign-heavy imbalance), the Adam optimizer (learning rate 1×10⁻⁴), and a
`ReduceLROnPlateau` scheduler on validation F1. The best checkpoint is selected by validation
F1, and training early-stops when validation F1 stops improving.

![Model architecture: preprocessing → EfficientNet-B0 backbone → 2-class head, with a Grad-CAM branch on the last convolutional block.](../artifacts/figures/architecture.png)
**Fig. 1.** End-to-end pipeline: image preprocessing, EfficientNet-B0 backbone, two-class
head, and the Grad-CAM branch on the final convolutional block.

### C. Skin-Tone-Aware Augmentation

Augmentation is applied **to the training set only** in the augmented experiment, before
normalization, in random order: `RandomHorizontalFlip(0.5)`, `RandomRotation(±20°)`, and
`ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)`. The color operations
sample factors from uniform ranges and transform the image as follows, where `in`/`out` are
pixel values, $\bar{g}$ is mean luminance, and `gray` is the grayscale image:

$$ \text{brightness: } \text{out} = b\cdot\text{in}, \quad b \sim U(0.8,\,1.2) \tag{1} $$

$$ \text{contrast: } \text{out} = c\,(\text{in}-\bar{g})+\bar{g}, \quad c \sim U(0.8,\,1.2) \tag{2} $$

$$ \text{saturation: } \text{out} = s\,\text{in}+(1-s)\,\text{gray}, \quad s \sim U(0.8,\,1.2) \tag{3} $$

$$ \text{hue: shift } H \text{ in HSV by } h \sim U(-0.1,\,0.1)\ (\approx \pm 36^{\circ}) \tag{4} $$

Darkening arises specifically from a brightness factor $b<1$ (and, secondarily, reduced
contrast/saturation), approximating darker-skin appearance in the absence of skin-tone labels
in HAM10000. Representative outputs are shown in Fig. 2.

![Augmentation examples: one lesion under flip, rotation, and color jitter.](../artifacts/figures/augmentation_examples.png)
**Fig. 2.** One training lesion rendered under several augmentations (flip, rotation, color
jitter).

### D. Grad-CAM Explainability

Grad-CAM is computed on the last convolutional block (`model.features[-1]`). For a target
class $c$ with activation maps $A^k$ (channel $k$), the gradient of the class score $y^c$ is
global-average-pooled to channel weights $\alpha_k^c$, combined, and rectified:

$$ \alpha_k^c = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A^k_{ij}} \tag{5} $$

$$ L^c = \mathrm{ReLU}\!\Big(\sum_k \alpha_k^c A^k\Big) \tag{6} $$

$L^c$ is bilinearly upsampled to 224×224, min–max normalized to $[0,1]$, and alpha-blended
over the input as $(1-\alpha)\,\text{image} + \alpha\,\text{jet}(L^c)$ with $\alpha=0.4$.

---

## IV. Experimental Setup

Two experiments are run with identical evaluation: **baseline** (preprocessing-only training)
and **augmented** (Section III-C). Both are trained on HAM10000 and evaluated on all 656 DDI
images. Hyperparameters: batch size 32, Adam learning rate 1×10⁻⁴, seed 42, up to 15 epochs
with early stopping on validation F1. With batch size 32, one epoch is 251 training batches
(8012 images) and 63 validation batches (2003 images). Metrics are Accuracy, Precision,
Recall, F1, and ROC-AUC (positive class = Malignant), plus per-skin-tone accuracy and a
Light–Dark fairness gap. Training was CPU-bound; the baseline early-stopped near epoch 6 (best
validation F1 at epoch 3). All artifacts (checkpoints, metrics JSON, figures, tables) are
regenerated deterministically from the pipeline.

---

## V. Results

### A. Overall Performance

On the DDI test set, augmentation substantially improved malignant detection. The baseline is
extremely conservative — perfect precision but near-zero recall — whereas the augmented model
trades some precision for much higher recall, F1, and ROC-AUC (Table II). ROC curves and
confusion matrices are shown in Fig. 4 and Fig. 5; training curves in Fig. 3.

**TABLE II. Overall DDI Metrics (Baseline vs Augmented)**

| Metric | Baseline | Augmented |
|---|---:|---:|
| Accuracy | 0.745 | 0.741 |
| Precision | 1.000 | 0.512 |
| Recall | 0.023 | 0.123 |
| F1 | 0.046 | 0.198 |
| ROC-AUC | 0.554 | 0.615 |

![Training and validation loss/F1 curves across epochs.](../artifacts/figures/training_curves.png)
**Fig. 3.** Training/validation loss and F1 across epochs.

![ROC curves on DDI for baseline and augmented models.](../artifacts/figures/roc_curve.png)
**Fig. 4.** ROC curves (DDI) for the baseline and augmented models.

![Confusion matrices on DDI for baseline and augmented models.](../artifacts/figures/confusion_matrices.png)
**Fig. 5.** Confusion matrices (DDI) for the baseline and augmented models.

### B. Fairness Across Skin Tones

Per-group results are reported in Table III, and the fairness gaps in Table IV. Augmentation
raised Light and Medium accuracy but **lowered** Dark accuracy (0.778 → 0.725). As a result,
the overall max–min spread narrowed (0.077 → 0.049), but the primary Light–Dark gap did not
improve and in fact became positive (−0.013 → +0.049). Per-group recall/F1 (Table III) did
rise for every group, consistent with the overall sensitivity gain.

**TABLE III. Per-Skin-Tone Metrics**

| Model | Skin tone | Count | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Light | 208 | 0.764 | 0.000 | 0.000 | 0.000 |
| Baseline | Medium | 241 | 0.701 | 1.000 | 0.027 | 0.053 |
| Baseline | Dark | 207 | 0.778 | 1.000 | 0.042 | 0.080 |
| Augmented | Light | 208 | 0.774 | 0.750 | 0.061 | 0.113 |
| Augmented | Medium | 241 | 0.726 | 0.722 | 0.176 | 0.283 |
| Augmented | Dark | 207 | 0.725 | 0.263 | 0.104 | 0.149 |

**TABLE IV. Fairness Gap (Accuracy by Group)**

| Model | Light | Medium | Dark | Light−Dark | Max−Min |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.764 | 0.701 | 0.778 | −0.013 | 0.077 |
| Augmented | 0.774 | 0.726 | 0.725 | +0.049 | 0.049 |

### C. Explainability

Grad-CAM overlays (Fig. 6) across skin tones and both classes confirm the model generally
concentrates activation on the lesion region rather than surrounding skin or imaging artifacts,
supporting that predictions are lesion-driven.

![Grad-CAM overlays across skin tones and classes.](../artifacts/figures/gradcam_examples.png)
**Fig. 6.** Grad-CAM examples across skin tones (Light/Medium/Dark) and classes
(Benign/Malignant).

---

## VI. Discussion

Augmentation clearly improved the model's ability to *detect* malignancy: recall rose from
0.023 to 0.123, F1 from 0.046 to 0.198, and ROC-AUC from 0.554 to 0.615. The baseline behaved
as a near-constant "benign" predictor (perfect precision, negligible recall), which inflates
accuracy on a benign-heavy test set while being clinically useless; augmentation produced a
more balanced operating point.

The fairness picture is mixed. The max–min accuracy spread narrowed, but the targeted
Light–Dark accuracy gap did **not** improve — Dark-group accuracy actually fell even though
Dark-group recall and F1 increased. This apparent tension is a consequence of the benign-heavy
class balance: accuracy and sensitivity move differently as the decision threshold shifts. The
most plausible explanation is that photometric jitter changes pixel intensities but not lesion
morphology or the characteristics of real skin-of-color clinical photography, so it is a weak
proxy for genuine diversity.

This aligns with the DDI study [4], where the large fairness gains came specifically from
**fine-tuning on diverse DDI data** (HAM10000 Light/Dark ROC-AUC rising to ≈0.77/0.78), not
from color augmentation of a non-diverse corpus. Our augmented model never trains on DDI, so
it should not be expected to reproduce those gains. Our baseline DDI ROC-AUC (0.554) is also
lower than the study's HAM10000 baseline (0.67), attributable to a different architecture
(EfficientNet-B0 vs Inception V3), a single seed, class-weighted loss versus balanced
sampling, early stopping, and different preprocessing/decision thresholds.

An additional observation comes from the Grad-CAM visualizations. While several Light-skin
examples show model attention concentrated around the lesion itself, some Medium- and
Dark-skin examples exhibit activations that are partially displaced from the lesion region and
extend into surrounding skin or background structures. This suggests that the model may not
always localize diagnostically relevant features consistently across skin-tone groups.
Although the number of Grad-CAM examples is too small to draw definitive conclusions,
inconsistent localization could help explain why fairness improvements were limited despite
improvements in overall sensitivity. If the model learns different visual cues for different
skin-tone groups, performance disparities may persist even when overall classification metrics
improve.

---

## VII. Limitations

Several factors limit the scope and interpretation of these results.

The most fundamental limitation is that the training and test data come from different sources.
HAM10000 consists primarily of dermoscopic images, whereas DDI contains clinical photographs.
These datasets differ in acquisition conditions, image quality, patient populations, and label
distributions. Consequently, the reported performance reflects both classifier quality and the
difficulty of cross-dataset generalization.

A second limitation is the absence of skin-tone labels in HAM10000. Because skin-tone
information is unavailable during training, the model cannot explicitly learn fairness-aware
representations or be optimized for equitable performance across demographic groups. Fairness
can only be assessed afterward using DDI.

The augmentation strategy itself is also limited. Photometric jitter modifies pixel intensities
but does not reproduce the anatomical, textural, and photographic characteristics of genuine
skin-of-color clinical images. In particular, the ±0.2 brightness adjustment used in this
project darkens images only modestly and may not adequately represent the appearance
variability observed in real-world dermatology datasets. As a result, augmentation serves only
as a weak proxy for true dataset diversity.

An additional limitation is revealed by the Grad-CAM visualizations. Although many examples
show attention focused on lesion regions, several Medium- and Dark-skin examples exhibit
activation patterns that are partially shifted away from the lesion and into surrounding skin
or unrelated image regions. These observations suggest that lesion localization may be less
consistent across skin-tone groups. Because accurate classification depends on identifying
clinically relevant structures, inconsistent localization could contribute to the remaining
fairness gap and may help explain why augmentation improved overall metrics while failing to
consistently improve Dark-group performance. The current analysis includes only a small number
of explainability examples, so this interpretation should be considered exploratory rather than
conclusive.

Evaluation is further constrained by the relatively small size of DDI. Each skin-tone subgroup
contains only approximately 200–240 images, which increases the variance of subgroup
performance estimates and makes fairness metrics sensitive to small numbers of classification
errors. In addition, all experiments were performed using a single random seed and CPU-limited
training, preventing a more comprehensive statistical analysis.

Finally, Grad-CAM itself has limitations as an explainability tool. Its heatmaps indicate
influential image regions but do not provide precise lesion boundaries and should not be
interpreted as segmentation masks. The system is intended solely for research purposes and is
not a medical device. Its predictions should not be used for clinical diagnosis or treatment
decisions.

---

## VIII. Conclusion and Future Work

This project asked whether skin-tone-aware image augmentation can improve melanoma
classification across diverse skin tones. Using EfficientNet-B0 trained on HAM10000 and
evaluated on the skin-tone-labeled DDI benchmark, augmentation improved overall malignant
detection (F1 and ROC-AUC) and narrowed the max–min per-group accuracy spread, but did **not**
reduce the targeted Light–Dark gap. The hypothesis is therefore only partially supported:
photometric augmentation helps robustness but is insufficient, on its own, to close skin-tone
fairness gaps — consistent with prior findings that representative data is the key lever [4].

Future work could fine-tune on genuinely diverse data (as in [4]), apply stronger augmentation techniques, run multiple random seeds with confidence intervals, train on the full dataset with GPU acceleration, and investigate domain-adaptation approaches such as Deep CORAL [10] to reduce the distribution shift between HAM10000 and DDI. Fairness-aware optimization methods such as GroupDRO [11] may also be explored if skin-tone labels become available during training. Another promising direction is adopting datasets with pixel-level lesion masks, such as ISIC and PH2, to support segmentation-based approaches rather than Grad-CAM-only localization [12].

---

## Acknowledgment

This was a solo project with no additional team members. The author used **Claude (Anthropic)**
as an AI coding assistant for implementation, debugging, and drafting; **GitHub** for version
control; **PyTorch/torchvision** for the model, training, and augmentation (EfficientNet-B0 with
ImageNet weights); **scikit-learn** for evaluation metrics; **FastAPI/Uvicorn** and
**React/Vite** for the web application; and **matplotlib/pandas** for figures and tables. The
work uses the publicly available HAM10000 [6] and DDI [5] datasets.

---

## References

[1] H. Li, Y. Pan, J. Zhao, and L. Zhang, "Skin disease diagnosis with deep learning: a review," 2020. [Online]. Available: https://arxiv.org/abs/2011.05627.

[2] N. Alipour, T. Burke, and J. Courtney, "Skin Type Diversity in Skin Lesion Datasets: A Review," *Current Dermatology Reports*, vol. 13, no. 3, pp. 198–210, Aug. 2024, doi: 10.1007/s13671-024-00440-0. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC11343783/.

[3] R. Fliorent et al., "Artificial Intelligence in dermatology: advancements and challenges in skin of color," *International Journal of Dermatology*, vol. 63, no. 4, pp. 455–461, Apr. 2024, doi: 10.1111/ijd.17076. [Online]. Available: https://onlinelibrary.wiley.com/doi/10.1111/ijd.17076.

[4] R. Daneshjou et al., "Disparities in dermatology AI performance on a diverse, curated
clinical image set," *Science Advances*, vol. 8, no. 32, Aug. 2022, doi: 10.1126/sciadv.abq6147. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC9374341/.

[5] "DDI — Diverse Dermatology Images," Center for Artificial Intelligence in Medicine & Imaging, 2024. doi: 10.71718/kqee-3z39. [Online]. Available: https://doi.org/10.71718/kqee-3z39.

[6] A. M. Scarlat, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions," *kaggledatasets*, Dec. 2018, doi: 10.7910/DVN/DBW86T. [Online]. Available: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T.

[7] S. Suara, A. Jha, P. Sinha, and S. Arif Ahmed, “Is Grad-CAM Explainable in Medical Images?,” July 20, 2023. doi: 10.1007/978-3-031-58181-6_11. [Online]. Available: https://arxiv.org/pdf/2307.10506v1.

[8] M. Tan and Q. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” 2019. Accessed: Oct. 16, 2026. [Online]. Available: http://proceedings.mlr.press/v97/tan19a/tan19a.pdf.

[9] L. Fei-Fei, J. Deng, and K. Li, “ImageNet: Constructing a Large-Scale Image Database,” Journal of Vision, vol. 9, no. 8, pp. 1037–1037, Mar. 2010, doi: 10.1167/9.8.1037.[Online]. Available: http://image-net.org/static_files/papers/imagenet_cvpr09.pdf.

[10] S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang, "Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization," ICLR, 2020. [Online]. Available: https://openreview.net/pdf?id=ryxGuJrFvS.

[11] B. Sun and K. Saenko, “Deep CORAL: Correlation Alignment for Deep Domain Adaptation,” July 06, 2016. Accessed: Oct. 17, 2026. [Online]. Available: https://arxiv.org/pdf/1607.01719.

[12] R. Bai and M. Zhou, “SL-HarDNet: Skin Lesion Segmentation With HarDNet,” Front Bioeng Biotechnol, vol. 10, p. 1028690, Jan. 2023, doi: 10.3389/fbioe.2022.1028690. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC9849244.