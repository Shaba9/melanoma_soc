"""Fairness evaluation by skin tone per FAIRNESS-SPEC.md.

Consumes the predictions saved in ``<exp>_metrics.json`` (Prompt 5), computes per-group
metrics and gaps, writes ``<exp>_fairness.json``, and augments the metrics JSON with the
``skin_tone`` and ``fairness_gap`` fields expected by the API.
"""
import json
import logging

from .. import config
from ..training.metrics import classification_metrics

logger = logging.getLogger(__name__)

GROUPS = list(config.SKIN_TONE_MAP.values())  # Light, Medium, Dark


def per_group_metrics(labels, preds, probs, groups) -> dict:
    """Accuracy/precision/recall/F1 and sample count for each skin-tone group."""
    result = {}
    for group in GROUPS:
        idx = [i for i, g in enumerate(groups) if g == group]
        if idx:
            m = classification_metrics([labels[i] for i in idx],
                                       [preds[i] for i in idx],
                                       [probs[i] for i in idx])
            result[group] = {
                "count": len(idx),
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            }
        else:
            result[group] = {"count": 0, "accuracy": None, "precision": None,
                             "recall": None, "f1": None}
    return result


def fairness_gaps(group_metrics: dict) -> dict:
    """Light−Dark accuracy gap and max−min accuracy gap across populated groups."""
    light = group_metrics["Light"]["accuracy"]
    dark = group_metrics["Dark"]["accuracy"]
    light_minus_dark = (light - dark) if (light is not None and dark is not None) else None
    accs = [group_metrics[g]["accuracy"] for g in GROUPS
            if group_metrics[g]["accuracy"] is not None]
    max_minus_min = (max(accs) - min(accs)) if accs else None
    return {"light_minus_dark": light_minus_dark, "max_minus_min": max_minus_min}


def compute_fairness(predictions: dict) -> dict:
    """Assemble per-group metrics and gaps from a predictions dict."""
    group_metrics = per_group_metrics(
        predictions["labels"], predictions["preds"],
        predictions["probs"], predictions["skin_tone_groups"],
    )
    gaps = fairness_gaps(group_metrics)
    return {
        "skin_tone": group_metrics,
        "fairness_gap": gaps["light_minus_dark"],
        "max_min_gap": gaps["max_minus_min"],
    }


def compute_and_save_fairness(experiment: str) -> dict:
    """Load `<exp>_metrics.json`, compute fairness, save `<exp>_fairness.json`, and
    add `skin_tone`/`fairness_gap` to the metrics JSON for the API."""
    metrics_path = config.METRICS_DIR / f"{experiment}_metrics.json"
    data = json.loads(metrics_path.read_text())
    fairness = compute_fairness(data["predictions"])

    fairness_path = config.METRICS_DIR / f"{experiment}_fairness.json"
    fairness_path.write_text(json.dumps({"experiment": experiment, **fairness}, indent=2))

    data["skin_tone"] = {
        g: {"accuracy": fairness["skin_tone"][g]["accuracy"],
            "count": fairness["skin_tone"][g]["count"]}
        for g in GROUPS
    }
    data["fairness_gap"] = fairness["fairness_gap"]
    metrics_path.write_text(json.dumps(data, indent=2))
    logger.info("Saved fairness report to %s", fairness_path)
    return fairness
