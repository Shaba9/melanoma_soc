# PROJECT_EXECUTION_CHECKLIST.md

# End-to-End Execution Checklist

## Phase 0 - Verify Inputs

### Required Assets
- HAM10000 images
- HAM10000 metadata CSV
- DDI images
- DDI metadata CSV
- All specification files

### Validation
- Confirm image paths are correct.
- Confirm metadata schemas match DATA-SPEC.md.
- Confirm DDI skin_tone values contain only 12, 34, 56.

---

# Phase 1 - Generate Codebase

Provide the following prompt to your coding agent:

## Prompt 1: Scaffold
Create the repository structure defined in REPO-SPEC.md.
Generate all folders, requirements.txt, README.md, .gitignore, and configuration templates.
Do not implement ML functionality yet.

## Prompt 2: Data Layer
Implement DATA-SPEC.md completely.
Create dataset loaders for HAM10000 and DDI.
Implement binary label mapping.
Implement skin-tone mapping.
Implement train/validation splitting.
Generate dataset validation reports.

## Prompt 3: Augmentation
Implement AUGMENTATION-SPEC.md.
Create baseline transforms.
Create augmented transforms.
Generate augmentation preview figures.

## Prompt 4: Model
Implement MODEL-SPEC.md.
Create EfficientNet-B0 model.
Implement training loop.
Implement checkpointing.
Implement metrics tracking.
Implement early stopping.

## Prompt 5: Experiments
Implement EXP-SPEC.md.
Create run_experiment.py.
Support baseline and augmented modes.
Generate metrics JSON outputs.

## Prompt 6: Fairness
Implement FAIRNESS-SPEC.md.
Compute per-group metrics.
Compute Light-Dark performance gap.
Generate fairness report artifacts.

## Prompt 7: Grad-CAM
Implement GRADCAM-SPEC.md.
Generate heatmaps.
Generate overlays.
Create export workflow.

## Prompt 8: Figures and Tables
Implement FIGURE-SPEC.md and TABLE-SPEC.md.
Generate all required publication figures and tables.

## Prompt 9: Backend
Implement FastAPI backend.
Create endpoints:
- /predict
- /heatmap
- /metrics
- /health

## Prompt 10: Frontend
Implement React frontend.
Create Upload page.
Create Results page.
Create Metrics Dashboard.

---

# Phase 2 - Review Generated Code

Verify:

- Code compiles.
- Dependencies install.
- Dataset loads.
- Model initializes.
- Tests pass.
- API starts.
- Frontend builds.

---

# Phase 3 - Environment Setup

## Create Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Phase 4 - Dataset Validation

Run:

```bash
python src/data/validate_dataset.py
```

Confirm:

- No missing files
- No duplicate IDs
- Correct class mapping
- Correct DDI tone mapping

---

# Phase 5 - Run Baseline Experiment

```bash
python run_experiment.py --exp baseline
```

Expected Outputs:

- checkpoints/baseline/
- metrics/baseline_metrics.json
- figures/baseline_confusion_matrix.png
- figures/baseline_roc_curve.png

---

# Phase 6 - Run Augmented Experiment

```bash
python run_experiment.py --exp augmented
```

Expected Outputs:

- checkpoints/augmented/
- metrics/augmented_metrics.json
- figures/augmented_confusion_matrix.png
- figures/augmented_roc_curve.png

---

# Phase 7 - Generate Fairness Metrics

```bash
python src/fairness/run_fairness.py
```

Verify:

- Accuracy by Light
- Accuracy by Medium
- Accuracy by Dark
- Gap analysis

Artifact:

```text
artifacts/metrics/fairness.json
```

---

# Phase 8 - Generate Grad-CAM Examples

```bash
python src/gradcam/generate_gradcam.py
```

Verify:

- Original image
- Heatmap image
- Overlay image

---

# Phase 9 - Generate Paper Figures

```bash
python src/viz/figures.py
```

Required:

- Architecture
- Augmentation examples
- Training curves
- ROC curves
- Confusion matrices
- Grad-CAM examples
- Fairness plots

---

# Phase 10 - Generate Tables

```bash
python src/viz/tables.py
```

Required:

- Dataset summary
- Hyperparameters
- Baseline metrics
- Augmented metrics
- Fairness metrics
- Gap analysis

---

# Phase 11 - Start Backend

```bash
uvicorn src.api.main:app --reload
```

Verify:

```text
http://localhost:8000/docs
```

Test:

- /predict
- /heatmap
- /metrics
- /health

---

# Phase 12 - Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Verify:

- Upload image
- View prediction
- View heatmap
- View metrics

---

# Phase 13 - Research Analysis

Compare:

- Baseline Accuracy vs Augmented Accuracy
- Baseline F1 vs Augmented F1
- Baseline ROC-AUC vs Augmented ROC-AUC
- Baseline Fairness Gap vs Augmented Fairness Gap

Answer:

1. Did augmentation improve performance?
2. Did augmentation improve fairness?
3. Did Grad-CAM focus on lesions?

---

# Phase 14 - Generate IEEE Paper

Agent Prompt:

Draft the IEEE paper using IEEE-PAPER-SPEC.md and REPORT-SPEC.md.
Populate all sections using actual experiment results.
Insert generated figures and tables.
Include limitations, fairness discussion, and Grad-CAM discussion.
Do not use placeholder values.

---

# Project Completion Criteria

The project is complete only when:

- Both experiments finished.
- Fairness analysis finished.
- Grad-CAM outputs generated.
- Figures generated.
- Tables generated.
- Backend operational.
- Frontend operational.
- IEEE report completed.
- Results interpreted.
