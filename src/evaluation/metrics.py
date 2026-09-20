"""
Evaluation metrics and reporting for time-series price/return prediction.

Includes standard regression metrics (MAE, RMSE, MAPE, R^2), directional
accuracy, and -- when evaluating return predictions -- precision/recall/F1
for the UP/DOWN direction classification implied by the sign of the return.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    mape: float
    r2: float
    directional_accuracy: float
    precision_up: Optional[float] = None
    recall_up: Optional[float] = None
    f1_up: Optional[float] = None
    precision_down: Optional[float] = None
    recall_down: Optional[float] = None
    f1_down: Optional[float] = None
    n_samples: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """MAPE, guarded against division-by-zero for near-zero true values
    (common when evaluating return targets, which can legitimately be 0).
    """
    denom = np.where(np.abs(y_true) < epsilon, epsilon, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def directional_accuracy(
    y_true_direction: np.ndarray,
    y_pred_direction: np.ndarray,
) -> float:
    """Fraction of samples where the predicted direction (sign) matches the
    true direction. Inputs should already be signed values (returns, or
    price deltas relative to the current price).
    """
    true_sign = np.sign(y_true_direction)
    pred_sign = np.sign(y_pred_direction)
    return float(np.mean(true_sign == pred_sign))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    direction_true: Optional[np.ndarray] = None,
    direction_pred: Optional[np.ndarray] = None,
    include_direction_classification: bool = False,
) -> RegressionMetrics:
    """Compute the full metrics suite for a set of predictions.

    Args:
        y_true, y_pred: the raw target values (price or return) and predictions.
        direction_true, direction_pred: signed values used purely to derive
            UP/DOWN direction (defaults to y_true/y_pred themselves, which is
            correct when the target IS a return; pass current-price-relative
            deltas separately if evaluating price targets).
        include_direction_classification: if True, also compute
            precision/recall/F1 for the UP/DOWN direction (appropriate when
            the target is a return, per project requirements).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    direction_true = np.asarray(direction_true if direction_true is not None else y_true, dtype=float)
    direction_pred = np.asarray(direction_pred if direction_pred is not None else y_pred, dtype=float)

    metrics = RegressionMetrics(
        mae=mean_absolute_error(y_true, y_pred),
        rmse=root_mean_squared_error(y_true, y_pred),
        mape=mean_absolute_percentage_error(y_true, y_pred),
        r2=r_squared(y_true, y_pred),
        directional_accuracy=directional_accuracy(direction_true, direction_pred),
        n_samples=len(y_true),
    )

    if include_direction_classification:
        true_labels = (direction_true > 0).astype(int)  # 1 = UP, 0 = DOWN
        pred_labels = (direction_pred > 0).astype(int)
        metrics.precision_up = float(precision_score(true_labels, pred_labels, pos_label=1, zero_division=0))
        metrics.recall_up = float(recall_score(true_labels, pred_labels, pos_label=1, zero_division=0))
        metrics.f1_up = float(f1_score(true_labels, pred_labels, pos_label=1, zero_division=0))
        metrics.precision_down = float(precision_score(true_labels, pred_labels, pos_label=0, zero_division=0))
        metrics.recall_down = float(recall_score(true_labels, pred_labels, pos_label=0, zero_division=0))
        metrics.f1_down = float(f1_score(true_labels, pred_labels, pos_label=0, zero_division=0))

    return metrics


def save_metrics_report(metrics_by_model: dict, path: str | Path) -> None:
    """Save a dict of {model_name: RegressionMetrics} to a JSON report file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {name: (m.to_dict() if isinstance(m, RegressionMetrics) else m) for name, m in metrics_by_model.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Saved evaluation report to %s", path)
