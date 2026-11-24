"""
SCoPE — Evaluation Metrics
Jesus Alan Hernandez Galvan
"""

import json
import os
import warnings
from datetime import datetime
from typing import Any

import numpy as np

from ..prediction import DistPrediction, Prediction

try:
    from sklearn.metrics import (
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.utils.extmath import softmax
except ImportError:
    raise ImportError(
        "scikit-learn is required for evaluation metrics. "
        "Install it with: uv add scikit-learn --group eval"
    )


ArrayLike = list | np.ndarray
Report = dict[str, Any]


class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        return super().default(o)


def _safe(fn, *, default: Any = None, label: str = "") -> Any:
    """Run *fn* and return its result; on any exception return *default*."""
    try:
        return fn()
    except Exception as exc:
        if label:
            warnings.warn(f"{label}: {exc}")
        return default


def _to_prob(y_score: np.ndarray, is_binary: bool) -> np.ndarray:
    """
    Normalize raw scores to probabilities.

    For 2-D arrays, apply softmax when rows don't already sum to 1.
    For binary tasks, collapse to the positive-class column.
    """
    if y_score.ndim == 1:
        return y_score

    if not np.allclose(y_score.sum(axis=1), 1.0):
        y_prob = softmax(y_score)
    else:
        y_prob = y_score

    return y_prob[:, 1] if is_binary else y_prob


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Report:
    fpr, tpr, thr = _safe(
        lambda: roc_curve(y_true, y_prob),
        default=(None, None, None),
        label="ROC curve",
    )
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thr,
        "auc_roc": _safe(
            lambda: roc_auc_score(y_true, y_prob, average="macro"),
            default=0.5,
            label="ROC AUC",
        ),
    }


def _shared_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Report:
    return {
        "f1": _safe(
            lambda: f1_score(y_true, y_pred, average="macro"), default=0.0, label="F1"
        ),
        "balanced_accuracy": _safe(
            lambda: balanced_accuracy_score(y_true, y_pred),
            default=0.0,
            label="balanced accuracy",
        ),
        "confusion_matrix": _safe(
            lambda: confusion_matrix(y_true, y_pred),
            default=None,
            label="confusion matrix",
        ),
        "confusion_matrix_normalized": _safe(
            lambda: confusion_matrix(y_true, y_pred, normalize="true"),
            default=None,
            label="normalized confusion matrix",
        ),
        "mcc": _safe(
            lambda: matthews_corrcoef(y_true, y_pred), default=0.0, label="MCC"
        ),
    }


def _save_report(report: Report, directory: str) -> None:
    try:
        os.makedirs(directory, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(directory, f"evaluation_report_{stamp}.json")
        with open(filepath, "w") as fh:
            json.dump(report, fh, cls=NumpyEncoder, indent=4)
    except Exception as exc:
        warnings.warn(f"Could not save report: {exc}")


def make_evaluation_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_score: ArrayLike,
    save_path: str | None = None,
) -> Report:

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score).astype(np.float64)

    n_classes = y_score.shape[1] if y_score.ndim > 1 else 1
    is_binary = n_classes == 2
    y_prob = _to_prob(y_score, is_binary)

    report: Report = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(y_true),
        "n_classes": n_classes,
    }

    if is_binary:
        report.update(_binary_metrics(y_true, y_prob))
    else:
        report.update(
            {
                "fpr": None,
                "tpr": None,
                "thresholds": None,
                "auc_roc": None,
                "ap_score": None,
            }
        )

    report.update(_shared_metrics(y_true, y_pred))

    if save_path:
        _save_report(report, save_path)

    return report


def distance_score(predictions: list[DistPrediction]) -> np.ndarray:
    d = np.array(
        [[p.distances[0], p.distances[1]] for p in predictions]
    )  # (n, 2, n_clf)
    tot = d[:, 0] + d[:, 1]
    r = np.divide(d[:, 0], tot, out=np.full_like(tot, 0.5), where=tot > 0)
    return np.median(r, axis=1)


def predictions_to_report(
    predictions: list[Prediction],
    y_true: list[int],
    save_path: str | None = None,
) -> Report:

    y_pred = np.array([p.predicted_class for p in predictions])

    classes = sorted(predictions[0].scores.keys())
    y_score = np.array([[p.scores[c] for c in classes] for p in predictions])

    report = make_evaluation_report(
        y_true=np.array(y_true),
        y_pred=y_pred,
        y_score=y_score,
    )

    if isinstance(predictions[0], DistPrediction) and len(classes) == 2:
        report["distance_auc"] = _safe(
            lambda: roc_auc_score(y_true, distance_score(predictions)),
            default=0.5,
            label="distance AUC",
        )

    if save_path:
        _save_report(report, save_path)

    return report
