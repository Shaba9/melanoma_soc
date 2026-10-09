"""Deterministic table generation per TABLE-SPEC.md.

Values are sourced only from `artifacts/metrics/*.json` and dataset metadata
(no manual entry). Each table is written as CSV and Markdown to
`artifacts/tables/`.
"""
import json
import logging

import pandas as pd

from .. import config

logger = logging.getLogger(__name__)

EXPERIMENTS = ["baseline", "augmented"]
GROUPS = ["Light", "Medium", "Dark"]
METRIC_ROWS = [("Accuracy", "accuracy"), ("Precision", "precision"),
               ("Recall", "recall"), ("F1", "f1"), ("ROC-AUC", "roc_auc")]


def _load_metrics(experiment: str) -> dict:
    return json.loads((config.METRICS_DIR / f"{experiment}_metrics.json").read_text())


def _load_fairness(experiment: str) -> dict:
    return json.loads((config.METRICS_DIR / f"{experiment}_fairness.json").read_text())


def _available_experiments() -> list:
    """Experiments that have saved metrics."""
    return [e for e in EXPERIMENTS if (config.METRICS_DIR / f"{e}_metrics.json").exists()]


def _round(value):
    return round(value, 3) if isinstance(value, (int, float)) else value


def table_dataset_summary() -> pd.DataFrame:
    """Table 1: per-dataset image counts and DDI skin-tone breakdown."""
    from ..data import load_ddi_metadata, load_ham_metadata, split_ham

    ham = load_ham_metadata()
    train_df, val_df = split_ham(ham)
    ddi = load_ddi_metadata()

    rows = []
    for name, df in [("HAM10000 (train)", train_df), ("HAM10000 (val)", val_df)]:
        rows.append({
            "Dataset": name,
            "Total Images": len(df),
            "Malignant": int((df["label"] == config.MALIGNANT).sum()),
            "Benign": int((df["label"] == config.BENIGN).sum()),
            "Skin-tone breakdown": "-",
        })
    for group in GROUPS:
        sub = ddi[ddi["skin_tone_group"] == group]
        rows.append({
            "Dataset": f"DDI ({group})",
            "Total Images": len(sub),
            "Malignant": int((sub["label"] == config.MALIGNANT).sum()),
            "Benign": int((sub["label"] == config.BENIGN).sum()),
            "Skin-tone breakdown": group,
        })
    return pd.DataFrame(rows)


def table_metrics(experiment: str) -> pd.DataFrame:
    """Tables 2 & 3: overall DDI metrics (Metric, Value) for one experiment."""
    overall = _load_metrics(experiment)["overall"]
    return pd.DataFrame(
        [{"Metric": label, "Value": _round(overall[key])} for label, key in METRIC_ROWS]
    )


def table_skin_tone() -> pd.DataFrame:
    """Table 4: per-skin-tone metrics, one block per model."""
    rows = []
    for exp in _available_experiments():
        if not (config.METRICS_DIR / f"{exp}_fairness.json").exists():
            continue
        groups = _load_fairness(exp)["skin_tone"]
        for group in GROUPS:
            g = groups.get(group) or {}
            rows.append({
                "Model": exp,
                "Skin Tone": group,
                "Count": g.get("count", 0),
                "Accuracy": _round(g.get("accuracy")),
                "Precision": _round(g.get("precision")),
                "Recall": _round(g.get("recall")),
                "F1": _round(g.get("f1")),
            })
    return pd.DataFrame(rows)


def table_fairness_gap() -> pd.DataFrame:
    """Table 5: per-group accuracy and fairness gaps per model."""
    rows = []
    for exp in _available_experiments():
        if not (config.METRICS_DIR / f"{exp}_fairness.json").exists():
            continue
        fair = _load_fairness(exp)
        groups = fair["skin_tone"]
        rows.append({
            "Model": exp,
            "Acc(Light)": _round((groups.get("Light") or {}).get("accuracy")),
            "Acc(Medium)": _round((groups.get("Medium") or {}).get("accuracy")),
            "Acc(Dark)": _round((groups.get("Dark") or {}).get("accuracy")),
            "Light-Dark Gap": _round(fair["fairness_gap"]),
            "Max-Min Gap": _round(fair["max_min_gap"]),
        })
    return pd.DataFrame(rows)


def _save(df: pd.DataFrame, name: str):
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.TABLES_DIR / f"{name}.csv"
    md_path = config.TABLES_DIR / f"{name}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False))
    logger.info("Saved table %s (.csv/.md)", name)
    return csv_path, md_path


def generate_all():
    """Generate every table from saved metrics/metadata."""
    _save(table_dataset_summary(), "dataset_summary")
    for exp in _available_experiments():
        _save(table_metrics(exp), f"{exp}_metrics")
    _save(table_skin_tone(), "skin_tone_metrics")
    _save(table_fairness_gap(), "fairness_gap")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_all()


if __name__ == "__main__":
    main()
