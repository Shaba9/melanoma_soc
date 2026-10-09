"""Classification metrics (positive class = Malignant) per EXP-SPEC.md."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(labels, preds, probs=None) -> dict:
    """Accuracy, precision, recall, F1, ROC-AUC, and 2x2 confusion matrix.

    `probs` is P(Malignant); ROC-AUC is NaN when only one class is present.
    """
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, preds, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, preds, pos_label=1, zero_division=0)),
    }
    if probs is not None and len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, np.asarray(probs)))
    else:
        metrics["roc_auc"] = float("nan")
    metrics["confusion_matrix"] = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    return metrics
