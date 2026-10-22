"""Tests for src.data.feature_engineering, with explicit leakage checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.feature_engineering import (
    add_macd,
    add_moving_averages,
    add_returns,
    add_rsi,
    add_volume_change,
    engineer_features,
    get_engineered_feature_names,
)


def _price_df(n=100, seed=0):
    rng = np.random.default_rng(seed)
    prices = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1h"),
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": rng.integers(1000, 5000, n).astype(float),
        }
    )


def test_add_returns_matches_manual_calc():
    df = _price_df()
    out = add_returns(df)
    manual = df["close"].pct_change()
    pd.testing.assert_series_equal(out["return"], manual, check_names=False)


def test_moving_average_only_uses_past_and_current_rows():
    """A moving average at row i must be invariant to changes made to rows
    AFTER i -- this is the core leakage check for rolling features."""
    df = _price_df()
    out1 = add_moving_averages(df, windows=(5,))

    df2 = df.copy()
    df2.loc[50:, "close"] = df2.loc[50:, "close"] + 1000  # perturb the future
    out2 = add_moving_averages(df2, windows=(5,))

    # Rows strictly before the perturbation must be identical.
    pd.testing.assert_series_equal(
        out1.loc[:48, "ma_5"], out2.loc[:48, "ma_5"], check_names=False
    )


def test_rsi_bounded_between_0_and_100():
    df = _price_df()
    out = add_rsi(df, window=14)
    valid = out["rsi_14"].dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_macd_columns_present():
    df = _price_df()
    out = add_macd(df)
    assert {"macd", "macd_signal", "macd_hist"}.issubset(out.columns)


def test_engineer_features_matches_declared_feature_names():
    df = _price_df()
    out = engineer_features(df)
    names = get_engineered_feature_names()
    for name in names:
        assert name in out.columns, f"{name} missing from engineered output"


def test_engineer_features_no_lookahead_on_full_pipeline():
    """Perturbing the future should not change past feature rows across the
    whole engineered feature set (not just one indicator)."""
    df = _price_df()
    out1 = engineer_features(df)

    df2 = df.copy()
    df2.loc[80:, "close"] = df2.loc[80:, "close"] * 2
    df2.loc[80:, "volume"] = df2.loc[80:, "volume"] * 5
    out2 = engineer_features(df2)

    feature_cols = get_engineered_feature_names()
    pd.testing.assert_frame_equal(
        out1.loc[:78, feature_cols].reset_index(drop=True),
        out2.loc[:78, feature_cols].reset_index(drop=True),
    )


def test_volume_change_from_zero_volume_is_nan_not_inf():
    df = _price_df(n=5)
    df.loc[2, "volume"] = 0.0
    out = add_volume_change(df)
    assert not np.isinf(out["volume_change"]).any()
    assert np.isnan(out.loc[3, "volume_change"])
