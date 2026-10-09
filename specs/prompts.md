# PROMPTS — Implementation Prompt Playbook

Ordered prompts to drive autonomous implementation. Each references the governing spec.

## 1. Scaffold
> Create the repo structure in [REPO-SPEC.md](REPO-SPEC.md): `src/`, `frontend/`,
> `artifacts/`, `report/`, `requirements.txt`, `.gitignore`, `README.md`.
> Do not implement ML functionality yet.

## 2. Data Layer
> Implement `src/data/` per [DATA-SPEC.md](DATA-SPEC.md): HAM10000/DDI datasets, binary label.
> Create dataset loaders for HAM10000 and DDI.
> mapping, skin-tone mapping, stratified lesion-safe split, preprocessing transforms.
> Generate dataset validation reports.

## 3. Augmentation
> Implement baseline and augmented transforms per [AUGMENTATION-SPEC.md](AUGMENTATION-SPEC.md).

## 4. Model
> Implement `build_model()` (EfficientNet-B0, 2-class head) and training loop per
> [MODEL-SPEC.md](MODEL-SPEC.md) with class weighting, checkpointing, seed=42.

## 5. Experiments
> Implement `run_experiment.py --exp {baseline,augmented}` per [EXP-SPEC.md](EXP-SPEC.md):
> train, select best, evaluate on DDI, save metrics JSON.
> Support baseline and augmented modes.

## 6. Fairness
> Implement per-skin-tone metrics and gap per [FAIRNESS-SPEC.md](FAIRNESS-SPEC.md).

## 7. Grad-CAM
> Implement Grad-CAM overlay per [GRADCAM-SPEC.md](GRADCAM-SPEC.md).

## 8. Figures & Tables
> Implement `src/viz/figures.py` and `src/viz/tables.py` per [FIGURE-SPEC.md](FIGURE-SPEC.md)
> and [TABLE-SPEC.md](TABLE-SPEC.md).

## 9. Backend
> Implement FastAPI app per [API-SPEC.md](API-SPEC.md): `/predict`, `/heatmap`, `/metrics`,
> `/health`.

## 10. Frontend
> Implement React + Vite app per [UI-SPEC.md](UI-SPEC.md): Upload, Results, Metrics pages.

## 11. Report
> Draft the IEEE report per [REPORT-SPEC.md](REPORT-SPEC.md) and
> [IEEE-PAPER-SPEC.md](IEEE-PAPER-SPEC.md); embed figures/tables; include limitations and
> team contribution statement.

## Conventions
- Follow [TRD.md](TRD.md) and [IMPLEMENTATION.md](IMPLEMENTATION.md).
- Keep changes minimal and config-driven; ensure reproducibility (seed=42).
