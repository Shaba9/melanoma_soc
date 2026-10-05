# IMPLEMENTATION — Implementation Specification

## 1. Repository Layout
See [REPO-SPEC.md](REPO-SPEC.md) for the full tree. Core backend package: `src/`.

```
src/
  config.py          # paths, hyperparameters, seed
  data/
    datasets.py      # HAM10000Dataset, DDIDataset
    transforms.py    # baseline + augmented transforms
    loaders.py       # split + DataLoader factories
  models/
    efficientnet.py  # build_model()
  train.py           # training loop, checkpointing
  evaluate.py        # metrics on DDI + fairness
  explain/gradcam.py # Grad-CAM overlay
  viz/figures.py     # figure generation
  viz/tables.py      # table generation
  api/main.py        # FastAPI app
run_experiment.py    # CLI: baseline | augmented
```

## 2. Configuration
- Single `config.py` (or YAML) with: data paths, seed=42, batch size, lr, epochs, device,
  augmentation flag, artifact dirs.
- Config snapshot saved into each experiment's output.

## 3. CLI
```
python run_experiment.py --exp baseline
python run_experiment.py --exp augmented
python -m src.viz.figures   # generate figures from saved metrics
python -m src.viz.tables    # generate tables
uvicorn src.api.main:app --reload
```

## 4. Determinism
Set seeds for `random`, `numpy`, `torch` (+ `cudnn.deterministic=True`).

## 5. Logging
- Per-epoch train/val loss + metrics to stdout and `artifacts/logs/<exp>.log`.
- Save `metrics.json` after evaluation.

## 6. Error Handling (boundaries only)
- Validate dataset paths exist at startup.
- Skip unreadable images with a logged warning.
- API validates uploaded file type/size.

## 7. Dependencies
`torch, torchvision, scikit-learn, numpy, pandas, matplotlib, pillow, pytorch-grad-cam,
fastapi, uvicorn, python-multipart`. Pin in `requirements.txt`.

## 8. Testing (minimal)
- Smoke test: build model, forward random tensor → shape (N,2).
- Data test: label mapping correctness for both datasets.
- API test: `/predict` returns valid schema on a sample image.
