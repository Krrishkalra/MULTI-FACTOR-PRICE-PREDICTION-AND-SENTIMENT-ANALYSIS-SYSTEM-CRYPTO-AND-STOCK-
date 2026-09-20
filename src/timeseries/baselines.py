"""
Baseline models the LSTM/Transformer must beat to justify their added
complexity (see project requirement: "Do not assume that an LSTM/Transformer
will automatically perform better").

Baseline 1 -- Naive:
    For the "return" target: predict 0 (i.e. "no change", tomorrow's price
    equals today's price). For the "price" target: predict the current
    price. This is the standard naive/persistence baseline for financial
    time series and is deliberately trivial.

Baseline 2 -- Random Forest on engineered features:
    Uses the same tabular per-row features as the deep models (flattened,
    not sequential) to predict the same target. This tests whether the
    sequence/temporal structure the LSTM exploits actually adds value beyond
    what a simple tree ensemble can already extract from a single row of
    engineered indicators.

Both baselines operate on the same chronological train/val/test split as the
deep models and are evaluated with the same metrics for a fair comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


@dataclass
class NaiveBaseline:
    """Predicts no change: 0 for return targets, current value for price targets."""

    target_type: str = "return"

    def predict(self, current_values: np.ndarray) -> np.ndarray:
        if self.target_type == "return":
            return np.zeros_like(current_values, dtype=float)
        return np.asarray(current_values, dtype=float)


def train_random_forest_baseline(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
    n_estimators: int = 200,
    max_depth: int | None = 8,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train a RandomForestRegressor baseline on tabular (non-sequential) features.

    Args:
        train_df: training split only (chronological -- see dataset.py).
        feature_cols: engineered feature columns (technical + sentiment).
        target_col: name of the target column.

    Returns:
        A fitted ``RandomForestRegressor``.
    """
    X = train_df[feature_cols].to_numpy(dtype=float)
    y = train_df[target_col].to_numpy(dtype=float)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    logger.info("Trained RandomForest baseline on %d rows, %d features", len(train_df), len(feature_cols))
    return model


def predict_random_forest_baseline(model: RandomForestRegressor, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Predict with a fitted RandomForest baseline on any split."""
    X = df[feature_cols].to_numpy(dtype=float)
    return model.predict(X)
