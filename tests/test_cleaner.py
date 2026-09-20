"""Tests for src.data.cleaner: dedup, invalid OHLC detection, missing values."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.cleaner import (
    clean_market_data,
    detect_invalid_ohlc,
    drop_invalid_ohlc,
    flag_return_outliers,
    handle_missing_values,
    remove_duplicate_timestamps,
)


def _base_df():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="1h"),
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }
    )


def test_remove_duplicate_timestamps_keeps_first():
    df = _base_df()
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    result = remove_duplicate_timestamps(dup)
    assert len(result) == len(df)


def test_detect_invalid_ohlc_flags_high_less_than_low():
    df = _base_df()
    df.loc[0, "high"] = 50  # now high < low, invalid
    invalid = detect_invalid_ohlc(df)
    assert invalid.iloc[0]
    assert not invalid.iloc[1]


def test_detect_invalid_ohlc_flags_negative_values():
    df = _base_df()
    df.loc[1, "volume"] = -5
    invalid = detect_invalid_ohlc(df)
    assert invalid.iloc[1]


def test_detect_invalid_ohlc_does_not_flag_nan_rows():
    """NaN handling is the responsibility of handle_missing_values, not
    detect_invalid_ohlc -- a NaN row should pass through here untouched so
    it can be forward-filled downstream instead of being dropped outright."""
    df = _base_df()
    df.loc[2, "close"] = np.nan
    invalid = detect_invalid_ohlc(df)
    assert not invalid.iloc[2]


def test_drop_invalid_ohlc_removes_bad_rows():
    df = _base_df()
    df.loc[0, "high"] = 10  # invalid: high < low/open/close
    cleaned = drop_invalid_ohlc(df)
    assert len(cleaned) == 4


def test_handle_missing_values_ffill():
    df = _base_df()
    df.loc[2, "close"] = np.nan
    filled = handle_missing_values(df, method="ffill")
    assert filled.loc[2, "close"] == df.loc[1, "close"]
    assert filled["close"].isna().sum() == 0


def test_handle_missing_values_ffill_drops_leading_nan():
    df = _base_df()
    df.loc[0, "close"] = np.nan  # no prior value to ffill from
    filled = handle_missing_values(df, method="ffill")
    assert len(filled) == 4


def test_handle_missing_values_drop():
    df = _base_df()
    df.loc[2, "close"] = np.nan
    filled = handle_missing_values(df, method="drop")
    assert len(filled) == 4


def test_flag_return_outliers_detects_extreme_move():
    # Use a larger baseline series so the single extreme move doesn't itself
    # dominate the mean/std it's being compared against (a known limitation
    # of the global-mean/std method with very small samples).
    n = 50
    closes = [100 + 0.1 * i for i in range(n)]
    closes[-1] = closes[-2] * 100  # extreme jump on the last row
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1h"),
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1000] * n,
        }
    )
    flagged = flag_return_outliers(df, z_threshold=2.0)
    assert flagged.iloc[-1]


def test_clean_market_data_full_pipeline_no_leakage_of_future_fill():
    """Ensure clean_market_data never back-fills (which would leak future
    values into earlier rows)."""
    df = _base_df()
    df.loc[2, "close"] = np.nan
    original_next_value = df.loc[3, "close"]
    cleaned = clean_market_data(df)
    # forward-filled value should equal the PRIOR row's close, not the next one
    assert cleaned.loc[2, "close"] != original_next_value
    assert cleaned.loc[2, "close"] == df.loc[1, "close"]
    assert "is_return_outlier" in cleaned.columns
