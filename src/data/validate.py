"""Dataset validation report per DATA-SPEC.md.

Checks schema, image availability, and DDI skin-tone codes; writes a JSON report and
prints a summary. Run with: ``python -m src.data.validate``.
"""
import json
import logging

import pandas as pd

from .. import config
from .loaders import load_ddi_metadata, load_ham_metadata, split_ham

logger = logging.getLogger(__name__)

HAM_COLUMNS = ["lesion_id", "image_id", "dx", "dx_type", "age", "sex",
               "localization", "dataset"]
DDI_COLUMNS = ["DDI_ID", "DDI_file", "skin_tone", "malignant", "disease"]


def _label_counts(df):
    return {config.CLASS_NAMES[k]: int((df["label"] == k).sum())
            for k in (config.BENIGN, config.MALIGNANT)}


def validate() -> dict:
    report = {"ham10000": {}, "ddi": {}, "issues": []}

    # HAM10000
    ham_raw = pd.read_csv(config.HAM_METADATA)
    missing_cols = [c for c in HAM_COLUMNS if c not in ham_raw.columns]
    if missing_cols:
        report["issues"].append(f"HAM10000 missing columns: {missing_cols}")
    ham = load_ham_metadata()
    train_df, val_df = split_ham(ham)
    overlap = set(train_df["lesion_id"]) & set(val_df["lesion_id"])
    if overlap:
        report["issues"].append(f"Lesion leakage across split: {len(overlap)} lesion_ids")
    report["ham10000"] = {
        "total_images": int(len(ham)),
        "label_counts": _label_counts(ham),
        "train_images": int(len(train_df)),
        "val_images": int(len(val_df)),
        "train_label_counts": _label_counts(train_df),
        "val_label_counts": _label_counts(val_df),
    }

    # DDI
    ddi_raw = pd.read_csv(config.DDI_METADATA, index_col=0)
    missing_cols = [c for c in DDI_COLUMNS if c not in ddi_raw.columns]
    if missing_cols:
        report["issues"].append(f"DDI missing columns: {missing_cols}")
    bad_tones = sorted(set(ddi_raw["skin_tone"].astype(int)) - set(config.SKIN_TONE_MAP))
    if bad_tones:
        report["issues"].append(f"DDI unexpected skin_tone codes: {bad_tones}")
    ddi = load_ddi_metadata()
    tone_counts = {g: int((ddi["skin_tone_group"] == g).sum())
                   for g in config.SKIN_TONE_MAP.values()}
    report["ddi"] = {
        "total_images": int(len(ddi)),
        "label_counts": _label_counts(ddi),
        "skin_tone_counts": tone_counts,
    }
    return report


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = validate()
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.LOGS_DIR / "dataset_validation.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nValidation report written to {out_path}")
    if report["issues"]:
        print(f"WARNING: {len(report['issues'])} issue(s) detected.")


if __name__ == "__main__":
    main()
