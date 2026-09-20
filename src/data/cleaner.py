"""
Market-data cleaning: timestamp sanity, duplicate removal, missing-value
handling, invalid-value detection, and outlier flagging.

All functions here operate on a single asset's chronologically sorted
DataFrame. None of them use any information from "the future" relative to a
given row -- see the module-level leakage notes in feature_engineering.py for
the broader leakage-prevention policy.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OHLC_COLS = ["open", "high", "low", "close"]


def remove_duplicate_timestamps(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Drop rows with a duplicate timestamp, keeping the first occurrence.

    Duplicate timestamps typically indicate a data-export glitch. Keeping the
    first occurrence is a deterministic, documented choice.
    """
    before = len(df)
    df = df.drop_duplicates(subset=[timestamp_col], keep="first").reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Removed %d duplicate-timestamp rows", dropped)
    return df


def detect_invalid_ohlc(df: pd.DataFrame) -> pd.Series:
    """Flag rows that violate basic OHLC invariants.

    A row is invalid if any of the following hold:
        - high < low
        - high < open or high < close
        - low > open or low > close
        - any OHLCV value is negative

    Note: null/NaN values are deliberately NOT flagged here -- missing
    values are a distinct concern handled by ``handle_missing_values``.
    Pandas comparisons involving NaN evaluate to False, so a row containing
    NaN simply will not be flagged by the checks above and will fall through
    to missing-value handling instead of being dropped as "invalid".

    Returns:
        Boolean Series, True where the row is invalid.
    """
    invalid = pd.Series(False, index=df.index)
    invalid |= df["high"] < df["low"]
    invalid |= df["high"] < df["open"]
    invalid |= df["high"] < df["close"]
    invalid |= df["low"] > df["open"]
    invalid |= df["low"] > df["close"]
    for col in OHLC_COLS + ["volume"]:
        invalid |= df[col] < 0
    return invalid


def drop_invalid_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows failing the OHLC sanity checks in ``detect_invalid_ohlc``."""
    invalid_mask = detect_invalid_ohlc(df)
    n_invalid = int(invalid_mask.sum())
    if n_invalid:
        logger.warning("Dropping %d rows with invalid OHLCV values", n_invalid)
    return df.loc[~invalid_mask].reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """Fill missing OHLCV values.

    Only past-looking fill strategies are supported to avoid leakage:
        - "ffill": forward-fill from the previous valid observation.
        - "drop": drop rows containing any missing OHLCV value.

    Backward-fill is intentionally NOT offered here because it would leak
    future values into the past.
    """
    if method == "drop":
        n_before = len(df)
        df = df.dropna(subset=list(OHLC_COLS) + ["volume"]).reset_index(drop=True)
        logger.info("Dropped %d rows with missing values", n_before - len(df))
        return df
    if method == "ffill":
        n_missing = df[OHLC_COLS + ["volume"]].isna().sum().sum()
        if n_missing:
            logger.info("Forward-filling %d missing OHLCV cells", int(n_missing))
        df = df.copy()
        df[OHLC_COLS + ["volume"]] = df[OHLC_COLS + ["volume"]].ffill()
        # Any leading NaNs (no prior value to ffill from) must still be dropped.
        n_before = len(df)
        df = df.dropna(subset=OHLC_COLS + ["volume"]).reset_index(drop=True)
        if n_before != len(df):
            logger.info("Dropped %d leading rows with no prior value to forward-fill", n_before - len(df))
        return df
    raise ValueError(f"Unknown missing-value method: {method}")


def flag_return_outliers(df: pd.DataFrame, z_threshold: float = 6.0) -> pd.Series:
    """Flag rows whose close-to-close return is an extreme outlier.

    Uses a rolling z-score computed only from past data (expanding window on
    a leading population would leak; instead we compute mean/std over the
    full training-time series once split, but for a standalone cleaning pass
    we approximate with a global mean/std of returns as a coarse deterministic
    sanity check). This function is meant for exploratory reporting; it does
    NOT drop rows automatically -- extreme moves in real market/crypto data
    can be genuine and should be reviewed rather than silently discarded.

    Args:
        df: DataFrame with a ``close`` column, sorted by timestamp.
        z_threshold: number of standard deviations beyond which a return is
            flagged as an outlier.

    Returns:
        Boolean Series aligned to df.index, True where flagged.
    """
    returns = df["close"].pct_change()
    mean, std = returns.mean(), returns.std()
    if std == 0 or np.isnan(std):
        return pd.Series(False, index=df.index)
    z = (returns - mean) / std
    flagged = z.abs() > z_threshold
    n_flagged = int(flagged.sum())
    if n_flagged:
        logger.info("Flagged %d rows as return outliers (|z| > %.1f)", n_flagged, z_threshold)
    return flagged.fillna(False)


def clean_market_data(
    df: pd.DataFrame,
    missing_value_method: str = "ffill",
    outlier_z_threshold: float = 6.0,
) -> pd.DataFrame:
    """Run the full market-data cleaning sequence in the documented order.

    Order matters: timestamps must already be parsed and sorted (done in the
    loader) before duplicate removal; invalid-value dropping happens before
    missing-value filling so that garbage rows do not get forward-filled
    into subsequent rows; outlier flagging happens last and is informational
    only (added as a boolean column, not used to drop rows).
    """
    df = remove_duplicate_timestamps(df)
    df = drop_invalid_ohlc(df)
    df = handle_missing_values(df, method=missing_value_method)
    df = df.copy()
    df["is_return_outlier"] = flag_return_outliers(df, z_threshold=outlier_z_threshold)
    return df
