# TRD — Technical Requirements Document

## 1. Purpose
Define the technical requirements for a web application that classifies skin lesions as
**Malignant** vs **Benign**, evaluates fairness across skin tones, and explains predictions
with Grad-CAM. Trains on HAM10000, evaluates on DDI.

## 2. Scope
- Two training experiments: **Baseline** (no augmentation) and **Augmented** (photometric + geometric).
- Fairness evaluation on DDI by skin tone (Light / Medium / Dark).
- Explainability via Grad-CAM overlays.
- React + Vite frontend; FastAPI backend.
- IEEE-formatted report with required figures and tables.

## 3. Functional Requirements
| ID | Requirement |
|----|-------------|
| FR-1 | Train EfficientNet-B0 (ImageNet-pretrained) on HAM10000 with binary labels. |
| FR-2 | Support two training modes: baseline and augmented. |
| FR-3 | Evaluate trained models on DDI and compute all metrics. |
| FR-4 | Compute per-skin-tone accuracy and Light-vs-Dark gap. |
| FR-5 | Generate Grad-CAM heatmaps for any input image. |
| FR-6 | Serve `/predict`, `/heatmap`, `/metrics` endpoints. |
| FR-7 | Frontend provides upload, results (with overlay), and metrics dashboard pages. |
| FR-8 | Export all required figures and tables to disk. |
| FR-9 | Produce an IEEE-format report following the rubric. |

## 4. Non-Functional Requirements
- **Reproducibility**: fixed random seed (42); config-driven runs.
- **Performance**: single prediction < 2 s on CPU.
- **Portability**: runs on CPU or CUDA; dependencies pinned.
- **Maintainability**: modular code (data, model, train, eval, explain, api).

## 5. Technology Stack
- Python 3.11+, PyTorch, torchvision, scikit-learn, numpy, pandas, matplotlib, pillow.
- Grad-CAM: `pytorch-grad-cam` or custom hook implementation.
- Backend: FastAPI + uvicorn.
- Frontend: React + Vite + Axios + a chart library (Recharts/Chart.js).

## 6. Acceptance Criteria
- Both experiments run end-to-end and emit metrics JSON.
- Fairness gap reported for both models.
- All figures/tables in [FIGURE-SPEC.md](FIGURE-SPEC.md) and [TABLE-SPEC.md](TABLE-SPEC.md) generated.
- API endpoints return valid responses; UI renders prediction + heatmap.
- Report satisfies [IEEE-PAPER-SPEC.md](IEEE-PAPER-SPEC.md) and the evaluation rubric.

## 7. Related Specs
See [DATA-SPEC.md](DATA-SPEC.md), [MODEL-SPEC.md](MODEL-SPEC.md),
[EXP-SPEC.md](EXP-SPEC.md), [IMPLEMENTATION.md](IMPLEMENTATION.md).
