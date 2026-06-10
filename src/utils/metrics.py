"""Evaluation metrics from Wang et al. (2024) arXiv:2405.10523."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_samples": int(len(y_true)),
        "has_probabilities": y_prob is not None,
    }


def ue_rate(
    total_samples: int,
    uncertain_count: int = 0,
    error_count: int = 0,
) -> float | None:
    """Uncertainty/Error rate: (U + E) / N."""
    if total_samples <= 0:
        return None
    return float((uncertain_count + error_count) / total_samples)
