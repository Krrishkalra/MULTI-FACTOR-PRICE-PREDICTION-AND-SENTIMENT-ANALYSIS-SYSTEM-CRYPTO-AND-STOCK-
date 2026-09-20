"""
Time-series dataset construction: chronological train/val/test split,
scaler fitting (train-only), and sliding-window sequence generation for the
LSTM/Transformer models.

Leakage-prevention notes (see README "Data Leakage Prevention" for the full
policy):
    - The split is purely chronological: train = earliest rows, then
      validation, then test = latest rows. No shuffling occurs before the
      split, ever.
    - Scalers (``sklearn.preprocessing.StandardScaler``) are fit ONLY on the
      training slice. The fitted scaler is then applied (``.transform``,
      never ``.fit_transform``) to the validation and test slices.
    - Sequences are built independently within each split after scaling, so
      a sequence's window never straddles the train/val or val/test boundary
      in a way that would let a test-set value influence a training
      sequence's features. Note: the first ``sequence_length - 1`` rows of
      the validation and test splits cannot form a full sequence purely from
      within-split data; rather than leaking across the boundary silently,
      we explicitly carry over the trailing ``sequence_length - 1`` rows of
      the previous split's *raw* (already scaled) feature values so the
      first sequences of val/test are still valid context windows. This is
      standard practice for sequential data and does not leak label
      information -- it only reuses feature rows whose scaling was fit on
      train, which is what we want.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def chronological_split(
    df: pd.DataFrame,
    train_split: float = 0.70,
    validation_split: float = 0.15,
) -> SplitResult:
    """Split a chronologically-sorted DataFrame into train/val/test slices.

    No shuffling is performed. ``train_split + validation_split`` must be
    < 1.0; the remainder is the test split.
    """
    if not 0 < train_split < 1:
        raise ValueError("train_split must be in (0, 1)")
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be in (0, 1)")
    if train_split + validation_split >= 1:
        raise ValueError("train_split + validation_split must be < 1 (remainder is the test split)")

    n = len(df)
    train_end = int(n * train_split)
    val_end = train_end + int(n * validation_split)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    logger.info(
        "Chronological split: train=%d, val=%d, test=%d (total=%d)",
        len(train_df),
        len(val_df),
        len(test_df),
        n,
    )
    return SplitResult(train_df=train_df, val_df=val_df, test_df=test_df)


def fit_scaler(train_df: pd.DataFrame, feature_cols: List[str]) -> StandardScaler:
    """Fit a StandardScaler on the training split ONLY.

    Args:
        train_df: the training slice of the dataset (must already exclude
            validation/test rows).
        feature_cols: columns to scale (should exclude the target and the
            raw timestamp column).

    Returns:
        A fitted ``StandardScaler``. Callers must reuse this same instance
        (via ``.transform``) for validation/test data -- never re-fit it.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].to_numpy(dtype=float))
    return scaler


def apply_scaler(df: pd.DataFrame, feature_cols: List[str], scaler: StandardScaler) -> pd.DataFrame:
    """Apply an already-fitted scaler to ``feature_cols`` of ``df``.

    Returns a new DataFrame; non-feature columns (timestamp, target, etc.)
    are preserved unchanged.
    """
    out = df.copy()
    out[feature_cols] = scaler.transform(df[feature_cols].to_numpy(dtype=float))
    return out


def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sequence_length: int,
    carry_over: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding-window sequences ``X`` and aligned targets ``y`` from a
    (scaled) DataFrame.

    Args:
        df: scaled DataFrame for one split, sorted chronologically, already
            containing the ``target`` column (created before scaling).
        feature_cols: feature columns to include in each sequence step.
        target_col: name of the target column.
        sequence_length: number of consecutive time steps per input sequence.
        carry_over: optional trailing rows (already scaled with the SAME
            scaler as ``df``) taken from the immediately preceding split, used
            purely as left-context so the first sequences of a val/test split
            do not need to be dropped. Must contain the same feature columns.
            Only feature rows are used from ``carry_over``; its targets are
            ignored (this is context only, not additional training labels).

    Returns:
        X: array of shape (n_sequences, sequence_length, n_features)
        y: array of shape (n_sequences,)
    """
    if carry_over is not None and len(carry_over) > 0:
        combined_features = pd.concat([carry_over[feature_cols], df[feature_cols]], axis=0).reset_index(drop=True)
        offset = len(carry_over)
    else:
        combined_features = df[feature_cols].reset_index(drop=True)
        offset = 0

    feature_matrix = combined_features.to_numpy(dtype=float)
    targets = df[target_col].to_numpy(dtype=float)

    n_rows = len(df)
    X_list = []
    y_list = []
    for i in range(n_rows):
        end = offset + i + 1  # inclusive of row i in df's own frame
        start = end - sequence_length
        if start < 0:
            continue  # not enough left-context even with carry_over; skip
        X_list.append(feature_matrix[start:end])
        y_list.append(targets[i])

    if not X_list:
        n_features = len(feature_cols)
        return np.empty((0, sequence_length, n_features)), np.empty((0,))

    return np.stack(X_list), np.array(y_list)


def get_carry_over(df: pd.DataFrame, sequence_length: int) -> pd.DataFrame:
    """Return the trailing ``sequence_length - 1`` rows of ``df`` to be used
    as left-context for the next split's sequence construction.
    """
    n_context = max(sequence_length - 1, 0)
    if n_context == 0:
        return df.iloc[0:0]
    return df.iloc[-n_context:].reset_index(drop=True)
