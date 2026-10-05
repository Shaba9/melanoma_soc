# REPO-SPEC — Repository Structure Specification

## 1. Top-Level Layout
```
melanoma_soc/
  data/
    HAM10000/            # training data (provided)
    DDI/                 # evaluation data (provided)
  specs/                 # engineering specifications
  rubrics/               # grading + IEEE template (provided)
  src/                   # backend + ML code (see IMPLEMENTATION.md)
  frontend/              # React + Vite app (see UI-SPEC.md)
  artifacts/             # generated outputs (gitignored)
    models/
    metrics/
    figures/
    tables/
    logs/
  report/                # IEEE report sources + final PDF
  run_experiment.py
  requirements.txt
  README.md
  .gitignore
```

## 2. artifacts/
All generated models, metrics JSON, figures, tables, and logs. Ignored by git except
`.gitkeep`. Figures/tables used in the report are copied into `report/`.

## 3. frontend/
Standard Vite project (`src/pages`, `src/components`, `src/api.js`).

## 4. report/
Holds the IEEE paper (Markdown/LaTeX/DOCX), figures, tables, and the final PDF per
[IEEE-PAPER-SPEC.md](IEEE-PAPER-SPEC.md).

## 5. Conventions
- Python: modules lowercase_snake; type hints on public functions.
- Config-driven runs; no hardcoded absolute paths (use `config.py`).
- Pinned dependencies in `requirements.txt` and `frontend/package.json`.

## 6. README.md
Must document: setup, data layout, how to run experiments, start API, start frontend,
and how to regenerate figures/tables.
