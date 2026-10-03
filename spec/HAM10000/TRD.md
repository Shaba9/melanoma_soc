# Technical Requirements Document

## Objective
Build a web application for melanoma classification with Grad-CAM explainability.

## Frontend
React + Vite

Pages:
1. Upload page
2. Prediction results page
3. Experiment dashboard

Display:
- Uploaded image
- Prediction
- Confidence
- Heatmap
- Overlay image

## Backend
FastAPI

Required Endpoints

POST /predict
Returns:
- prediction
- confidence
- heatmap_path

GET /metrics
Returns experiment metrics.

POST /train/baseline
Starts baseline experiment.

POST /train/augmented
Starts augmentation experiment.

## ML Stack
- PyTorch
- Torchvision
- Pandas
- NumPy
- Scikit-Learn

## Data Loader Requirements
- Support .tab and .csv
- Read image_id and dx
- Convert diagnosis to binary labels
- Stratified splits

## Model
EfficientNet-B0

Weights:
IMAGENET1K_V1

Replace classifier with 2-class output layer.

## Training Requirements
Loss:
CrossEntropyLoss

Optimizer:
Adam

Learning Rate:
1e-4

Batch Size:
32

Epochs:
10-20

Early Stopping:
3 epochs

## Augmentation Requirements
Training only.
No test-time augmentation.

## Explainability
Grad-CAM

Save:
- heatmap.png
- overlay.png

## Logging
Save:
- training history
- metrics JSON
- confusion matrices
- ROC values

## Repository Structure
```text
frontend/
backend/
datasets/
models/
experiments/
evaluation/
gradcam/
reports/
```

## Deliverables
1. Working web app
2. Trained baseline model
3. Trained augmented model
4. Metrics report
5. Grad-CAM examples
6. Final IEEE report
