# UI-SPEC — Frontend Specification

## 1. Stack
React + Vite. HTTP via Axios. Charts via Recharts (or Chart.js). Plain CSS/CSS modules.

## 2. Pages / Routes
| Route | Page | Purpose |
|-------|------|---------|
| `/` | Upload | Upload a lesion image, trigger prediction. |
| `/results` | Results | Show prediction, probability, Grad-CAM overlay. |
| `/metrics` | Dashboard | Show model metrics, fairness, confusion matrix. |

## 3. Upload Page
- File input (drag-drop or picker), image preview.
- "Analyze" button → `POST /predict` and `POST /heatmap`.
- Loading state; navigate to Results on success.

## 4. Results Page
- Original image + Grad-CAM overlay side by side.
- Prediction label (Malignant/Benign) and probability.
- "Analyze another" button back to Upload.

## 5. Metrics Dashboard
- Experiment toggle: Baseline / Augmented → `GET /metrics?exp=`.
- Display: accuracy, precision, recall, F1, ROC-AUC cards.
- Confusion matrix (2×2) rendering.
- Per-skin-tone accuracy bar chart (Light/Medium/Dark) + fairness gap value.

## 6. API Integration
- Configurable base URL (`VITE_API_URL`, default `http://localhost:8000`).
- Handle and display error states (invalid file, server error).

## 7. UX / Accessibility
- Responsive layout; alt text on images; disabled buttons during loading.
- No medical-advice claims; display a research-only disclaimer.
