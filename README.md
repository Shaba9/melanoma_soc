# Melanoma SOC

End-to-end melanoma classification system leveraging EfficientNet-B0, HAM10000, DDI, fairness analysis, and Grad-CAM explainability.

## Project Structure
- src/: backend and ML code
- frontend/: React/Vite UI
- artifacts/: generated outputs
- report/: IEEE paper assets

## Setup
python -m venv venv
pip install -r requirements.txt

## Run Experiments
python run_experiment.py --exp baseline
python run_experiment.py --exp augmented

## Backend
uvicorn src.api.main:app --reload

## Frontend
cd frontend
npm install
npm run dev
