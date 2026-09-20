"""Tests for src.data.alignment: target creation and warm-up row dropping,
with explicit checks that the target uses FUTURE data intentionally and only
in the one sanctioned place."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.alignment import add_target, drop_warmup_rows


def _df_with_close(closes):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(closes), freq="1h"),
            "close": closes,
        }
    )


def test_add_target_return_matches_manual_formula():
    closes = [100.0, 102.0, 101.0, 105.0]
    df = _df_with_close(closes)
    out = add_target(df, target_type="return", horizon=1)
    # Row 0: (102-100)/100 = 0.02 ; Row 1: (101-102)/102 ; Row 2: (105-101)/101
    expected = [(102 - 100) / 100, (101 - 102) / 102, (105 - 101) / 101]
    np.testing.assert_allclose(out["target"].to_numpy(), expected)
    # Last row (no future close) must be dropped.
    assert len(out) == 3


def test_add_target_price_is_next_close():
    closes = [100.0, 102.0, 101.0]
    df = _df_with_close(closes)
    out = add_target(df, target_type="price", horizon=1)
    assert list(out["target"]) == [102.0, 101.0]
    assert len(out) == 2


def test_add_target_horizon_2_skips_correct_rows():
    closes = [100.0, 102.0, 101.0, 105.0, 110.0]
    df = _df_with_close(closes)
    out = add_target(df, target_type="price", horizon=2)
    assert list(out["target"]) == [101.0, 105.0, 110.0]
    assert len(out) == 3


def test_add_target_never_uses_past_for_future_row():
    """Sanity check: the target for the LAST valid row must come from a row
    strictly after it in time, never from itself or the past."""
    closes = [100.0, 105.0, 90.0, 120.0]
    df = _df_with_close(closes)
    out = add_target(df, target_type="price", horizon=1)
    for i in range(len(out)):
        assert out.loc[i, "target"] == closes[i + 1]


def test_drop_warmup_rows_removes_nan_rows_only():
    df = pd.DataFrame({"a": [np.nan, 1, 2], "b": [1, np.nan, 3], "c": [1, 2, 3]})
    out = drop_warmup_rows(df)
    assert len(out) == 1
    assert out.iloc[0]["c"] == 3
