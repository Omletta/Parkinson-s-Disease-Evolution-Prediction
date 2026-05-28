"""Evaluation helpers with masked loss for partially observed UPDRS targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import UPDRS_TARGETS


def masked_mae(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.Series:
    scores = {}
    for col in UPDRS_TARGETS:
        mask = y_true[col].notna()
        if mask.sum() == 0:
            scores[col] = np.nan
            continue
        scores[col] = np.mean(np.abs(y_true.loc[mask, col] - y_pred.loc[mask, col]))
    return pd.Series(scores)


def masked_rmse(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.Series:
    scores = {}
    for col in UPDRS_TARGETS:
        mask = y_true[col].notna()
        if mask.sum() == 0:
            scores[col] = np.nan
            continue
        err = y_true.loc[mask, col] - y_pred.loc[mask, col]
        scores[col] = np.sqrt(np.mean(err**2))
    return pd.Series(scores)


def masked_mean_absolute_error(y_true, y_pred, target_names=UPDRS_TARGETS) -> float:
    """Single scalar for sklearn-style scoring (lower is better)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    total = 0.0
    count = 0
    for idx in range(y_true.shape[1]):
        mask = ~np.isnan(y_true[:, idx])
        if mask.sum() == 0:
            continue
        total += np.abs(y_true[mask, idx] - y_pred[mask, idx]).sum()
        count += mask.sum()
    return total / count if count else np.nan
