**Do not paste all prompts at once. Run them sequentially and commit after Prompts 1, 4, 9, 12, 15, and 17. That will make it much easier to recover if the agent introduces errors.**


################################################################################
MASTER PROMPT 0 — READ THE PROJECT SPECIFICATIONS FIRST
################################################################################

Read and fully understand the following files before writing any code:

- DATA-SPEC.md
- EXP-SPEC.md
- TRD.md
- IMPLEMENTATION-SPEC.md

Requirements:

1. Summarize all project requirements.
2. Identify:
   - Dataset requirements
   - Model requirements
   - Training requirements
   - Augmentation requirements
   - Explainability requirements
   - Backend requirements
   - Frontend requirements
3. Identify missing details or implementation risks.
4. Create an implementation plan.
5. Create a dependency graph.
6. Create a development roadmap.

Output analysis only.

DO NOT generate any source code yet.

################################################################################
MASTER PROMPT 1 — CREATE PROJECT STRUCTURE
################################################################################

Using the specification documents:

- DATA-SPEC.md
- EXP-SPEC.md
- TRD.md
- IMPLEMENTATION-SPEC.md

Create the complete repository structure.

Required structure:

frontend/
backend/
datasets/
models/
augmentation/
experiments/
evaluation/
gradcam/
outputs/
reports/
tests/
configs/
scripts/
docs/

Requirements:

1. Create all folders.
2. Create README.md in every folder explaining purpose.
3. Create root README.md.
4. Create requirements.txt.
5. Create .gitignore.
6. Create project architecture diagram.

Do not implement any model code yet.

################################################################################
MASTER PROMPT 2 — ANALYZE HAM10000 DATASET
################################################################################

Read HAM10000_metadata.tab.

Tasks:

1. Detect delimiter automatically.
2. List all columns.
3. Display first 10 rows.
4. Generate dataset summary.
5. Calculate diagnosis distribution.
6. Create binary labels:
   mel = 1
   all others = 0
7. Calculate:
   - melanoma count
   - non-melanoma count
   - class imbalance ratio
8. Generate visualizations:
   - diagnosis distribution
   - binary class distribution
9. Create dataset report.

Export:

outputs/dataset_summary.csv
outputs/dataset_report.md
outputs/diagnosis_distribution.png
outputs/binary_distribution.png

################################################################################
MASTER PROMPT 3 — BUILD DATASET LOADER
################################################################################

Build the complete HAM10000 dataset loader according to DATA-SPEC.md and IMPLEMENTATION-SPEC.md.

Requirements:

- Support .tab metadata
- Support .csv metadata
- Read image_id
- Read dx
- Create binary labels
- Construct image paths
- Validate files
- Handle missing images safely

Create:

dataset.py

Implement:

HAM10000Dataset

Return:

image_tensor
label

Generate:

tests/test_dataset.py

Add full docstrings.

################################################################################
MASTER PROMPT 4 — BUILD DATA SPLITTING PIPELINE
################################################################################

Create train/validation/test splitting.

Requirements:

Train = 70%

Validation = 15%

Test = 15%

Use:

Stratified split

Random seed = 42

Export:

splits/train.csv
splits/validation.csv
splits/test.csv

Generate:

split_dataset.py

Display:

- total records
- class counts
- class percentages

################################################################################
MASTER PROMPT 5 — HANDLE CLASS IMBALANCE
################################################################################

Analyze HAM10000 class imbalance.

Implement both:

1. Weighted CrossEntropyLoss
2. WeightedRandomSampler

Tasks:

- Calculate weights
- Compare methods
- Recommend preferred solution

Create:

imbalance.py

Generate report:

outputs/class_imbalance_report.md

################################################################################
MASTER PROMPT 6 — IMPLEMENT BASELINE TRANSFORMS
################################################################################

Create baseline preprocessing transforms.

Requirements:

Resize(224,224)

Convert to Tensor

Normalize using:

mean = [0.485,0.456,0.406]
std = [0.229,0.224,0.225]

No augmentation.

Create:

transforms_baseline.py

################################################################################
MASTER PROMPT 7 — IMPLEMENT AUGMENTATION PIPELINE
################################################################################

Create the augmentation pipeline defined in IMPLEMENTATION-SPEC.md.

Apply only to training images.

Required augmentations:

RandomHorizontalFlip(0.5)

RandomRotation(15)

ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    hue=0.05
)

Validation and test sets must not be augmented.

Create:

transforms_augmented.py

Generate:

outputs/sample_augmentations/

Save:

- original image
- augmented image pairs

Generate augmentation report.

################################################################################
MASTER PROMPT 8 — IMPLEMENT EFFICIENTNET-B0 MODEL
################################################################################

Build EfficientNet-B0 transfer learning model.

Requirements:

Framework:
PyTorch

Architecture:
EfficientNet-B0

Weights:
IMAGENET1K_V1

Replace classifier with:

2 output classes

Mapping:

0 = Not Melanoma

1 = Melanoma

Create:

model.py

Output:

- Total parameters
- Trainable parameters

################################################################################
MASTER PROMPT 9 — BUILD TRAINING PIPELINE
################################################################################

Build production-quality training pipeline.

Requirements:

Model:
EfficientNet-B0

Optimizer:
Adam

Learning Rate:
0.0001

Batch Size:
32

Epochs:
20

Early Stopping:
Patience = 3

Features:

- checkpointing
- resume training
- GPU support
- training history
- validation monitoring

Save:

models/best_model.pt
models/final_model.pt

Export:

training_history.json

Generate:

train.py

################################################################################
MASTER PROMPT 10 — RUN BASELINE EXPERIMENT
################################################################################

Execute Experiment A from EXP-SPEC.md.

Dataset:
Original HAM10000

Transforms:
Baseline only

Compute:

Accuracy

Precision

Recall

F1

ROC-AUC

Confusion Matrix

Classification Report

Export:

outputs/baseline_metrics.csv

outputs/baseline_predictions.csv

outputs/baseline_confusion_matrix.png

outputs/baseline_roc_curve.png

################################################################################
MASTER PROMPT 11 — RUN AUGMENTATION EXPERIMENT
################################################################################

Execute Experiment B from EXP-SPEC.md.

Dataset:
HAM10000

Transforms:
Augmentation pipeline

All training settings must remain identical to baseline.

Compute:

Accuracy

Precision

Recall

F1

ROC-AUC

Confusion Matrix

Classification Report

Export:

outputs/augmented_metrics.csv

outputs/augmented_predictions.csv

outputs/augmented_confusion_matrix.png

outputs/augmented_roc_curve.png

################################################################################
MASTER PROMPT 12 — COMPARE EXPERIMENTS
################################################################################

Compare baseline and augmented experiments.

Generate:

1. Metrics comparison table
2. Improvement table
3. ROC comparison chart
4. Confusion matrix comparison
5. Performance summary

Answer:

- Did augmentation improve performance?
- Which metrics improved?
- Which metrics degraded?

Export:

outputs/experiment_comparison.csv

outputs/comparison_report.md

################################################################################
MASTER PROMPT 13 — IMPLEMENT GRAD-CAM
################################################################################

Implement Grad-CAM using:

pytorch-grad-cam

Requirements:

Support EfficientNet-B0.

Generate examples for:

- True Positive
- True Negative
- False Positive
- False Negative

Save:

outputs/heatmaps/

For each sample save:

original.png

heatmap.png

overlay.png

Create:

gradcam.py

################################################################################
MASTER PROMPT 14 — BUILD INFERENCE PIPELINE
################################################################################

Build inference pipeline.

Input:

single image

Output:

- predicted label
- probability
- confidence
- Grad-CAM image

Load:

models/best_model.pt

Create:

inference.py

Support:

JPG

JPEG

PNG

################################################################################
MASTER PROMPT 15 — BUILD FASTAPI BACKEND
################################################################################

Build FastAPI backend.

Endpoints:

POST /predict

GET /metrics

POST /train/baseline

POST /train/augmented

Requirements:

- logging
- validation
- exception handling
- OpenAPI documentation

Create full backend project.

################################################################################
MASTER PROMPT 16 — BUILD REACT FRONTEND
################################################################################

Build React + Vite frontend.

Use Material UI.

Pages:

1. Upload Page
2. Prediction Page
3. Experiment Dashboard

Features:

- image upload
- prediction display
- confidence display
- Grad-CAM display
- metrics display
- experiment dashboard

Create reusable React components.

################################################################################
MASTER PROMPT 17 — CONNECT FRONTEND AND BACKEND
################################################################################

Connect React frontend and FastAPI backend.

Requirements:

- image upload workflow
- prediction workflow
- heatmap display
- metrics retrieval
- dashboard integration

Verify all API calls work correctly.

################################################################################
MASTER PROMPT 18 — CREATE TEST SUITE
################################################################################

Build full test suite.

Include:

Dataset Tests

Transform Tests

Model Tests

Training Tests

Inference Tests

API Tests

Frontend Tests

Generate:

tests/

TEST_REPORT.md

################################################################################
MASTER PROMPT 19 — END-TO-END VALIDATION
################################################################################

Perform complete application validation.

Verify:

- Dataset loading
- Metadata parsing
- Training
- Inference
- Grad-CAM
- Backend
- Frontend
- Integration

Generate:

VALIDATION_REPORT.md

Include:

Passed Tests

Failed Tests

Recommendations

Known Issues

################################################################################
MASTER PROMPT 20 — GENERATE DOCUMENTATION
################################################################################

Generate:

README.md

INSTALLATION.md

USER_GUIDE.md

DEVELOPER_GUIDE.md

API_DOCUMENTATION.md

EXPERIMENT_RESULTS.md

REPRODUCTION_GUIDE.md

Requirements:

A user must be able to clone the repository and reproduce results.

################################################################################
MASTER PROMPT 21 — GENERATE IEEE REPORT ASSETS
################################################################################

Generate all report figures.

Required Figures:

1. System Architecture
2. Dataset Distribution
3. Sample Augmentations
4. Training Curves
5. Baseline ROC Curve
6. Augmented ROC Curve
7. Baseline Confusion Matrix
8. Augmented Confusion Matrix
9. Grad-CAM Examples

Export to:

reports/final_report_assets/

################################################################################
MASTER PROMPT 22 — GENERATE IEEE REPORT TABLES
################################################################################

Generate:

Table 1 Dataset Summary

Table 2 Diagnosis Distribution

Table 3 Baseline Metrics

Table 4 Augmented Metrics

Table 5 Experiment Comparison

Table 6 Model Configuration

Table 7 Training Parameters

Export all tables as:

CSV

PNG

Markdown

################################################################################
MASTER PROMPT 23 — FINAL PROJECT AUDIT
################################################################################

Perform a complete audit against:

- DATA-SPEC.md
- EXP-SPEC.md
- TRD.md
- IMPLEMENTATION-SPEC.md

Verify every requirement has been implemented.

Generate:

FINAL_AUDIT_REPORT.md

For each requirement:

- Implemented
- Partially Implemented
- Missing

Recommend final improvements before project submission.