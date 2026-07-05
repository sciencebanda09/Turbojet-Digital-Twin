"""Regression metric suite with per-target breakdown."""

from typing import Any
import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from src.dataset.loader import TARGETS


def regression_metrics(
    truth: np.ndarray, prediction: np.ndarray, target_names: list[str] | None = None
) -> dict[str, Any]:
    """Calculate aggregate and per-target metrics.

    Returns aggregate RMSE, MAE, MAPE, R2, explained variance, and a
    ``per_target`` dict with individual metrics for each target.
    """
    names = target_names or TARGETS
    y, p = np.asarray(truth), np.asarray(prediction)
    denominator = np.maximum(np.abs(y), np.finfo(float).eps)
    per_target = {}
    for i, name in enumerate(names):
        yi, pi = y[:, i], p[:, i]
        denom_i = np.maximum(np.abs(yi), np.finfo(float).eps)
        per_target[name] = {
            "rmse": float(np.sqrt(mean_squared_error(yi, pi))),
            "mae": float(mean_absolute_error(yi, pi)),
            "mape": float(np.mean(np.abs((yi - pi) / denom_i)) * 100),
            "r2": float(r2_score(yi, pi)),
        }
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": float(mean_absolute_error(y, p)),
        "mape": float(np.mean(np.abs((y - p) / denominator)) * 100),
        "r2": float(r2_score(y, p)),
        "explained_variance": float(explained_variance_score(y, p)),
        "per_target": per_target,
    }
