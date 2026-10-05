# REPORT-SPEC — Report Content Specification

Maps project results to the evaluation rubric (Introduction 20, Results & Discussion 40,
Conclusions 10, Writing 30). Formatted per [IEEE-PAPER-SPEC.md](IEEE-PAPER-SPEC.md).

## 1. Abstract
Problem, approach (EfficientNet-B0 + augmentation), datasets (HAM10000→DDI), key result
(metrics + fairness gap change), ~150–200 words.

## 2. Introduction (Rubric: 20)
- **Problem & motivation (10)**: skin-cancer classifiers underperform on darker skin tones;
  clinical importance of equitable detection.
- **Literature / prior work (10)**: deep learning for dermoscopy, dataset bias, fairness in
  medical imaging, HAM10000 and DDI background. Cite prior work.

## 3. Related Work
Brief survey: CNNs for skin lesion classification, augmentation for robustness, fairness
evaluation by skin tone, explainability (Grad-CAM).

## 4. Methods (Rubric: Results & Discussion — methods 20)
- Datasets and label/skin-tone mappings ([DATA-SPEC.md](DATA-SPEC.md)).
- Model + training ([MODEL-SPEC.md](MODEL-SPEC.md)).
- Augmentation pipeline ([AUGMENTATION-SPEC.md](AUGMENTATION-SPEC.md)).
- Grad-CAM ([GRADCAM-SPEC.md](GRADCAM-SPEC.md)). Include Fig 1, Fig 2.

## 5. Experimental Setup
Splits, hyperparameters, metrics, hardware, reproducibility ([EXP-SPEC.md](EXP-SPEC.md)).

## 6. Results (Rubric: Results & Discussion — analysis 20)
- Baseline vs augmented metrics (Tables 2–3).
- ROC (Fig 4), confusion matrices (Fig 5), training curves (Fig 3).
- Fairness: per-skin-tone metrics (Table 4) and gap analysis (Table 5).
- Grad-CAM examples (Fig 6).

## 7. Discussion
Interpret whether augmentation reduced the Light-vs-Dark gap; trade-offs; what Grad-CAM
reveals; cross-dataset generalization.

## 8. Limitations
Summarize [LIMITATIONS-SPEC.md](LIMITATIONS-SPEC.md).

## 9. Conclusion (Rubric: 10)
Restate problem, findings, and connection to the research goal; future work.

## 10. References
IEEE numbered style; all claims cited.

## Writing (Rubric: 30)
Coherent organization, no spelling/grammar errors, defined terminology, proper IEEE
formatting, and a team-member contribution statement.
