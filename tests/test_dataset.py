"""Tests for src.timeseries.dataset: chronological split, scaler fit-on-train
-only discipline, and sequence construction (including leakage checks)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.timeseries.dataset import (
    apply_scaler,
    build_sequences,
    chronological_split,
    fit_scaler,
    get_carry_over,
)


def _sample_dataset(n=100):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="1h"),
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.normal(10, 2, n),
            "target": rng.normal(0, 0.01, n),
        }
    )


def test_chronological_split_no_shuffling_and_correct_sizes():
    df = _sample_dataset(100)
    split = chronological_split(df, train_split=0.7, validation_split=0.15)
    assert len(split.train_df) == 70
    assert len(split.val_df) == 15
    assert len(split.test_df) == 15
    # Order preserved: train's last timestamp < val's first timestamp < test's first
    assert split.train_df["timestamp"].iloc[-1] < split.val_df["timestamp"].iloc[0]
    assert split.val_df["timestamp"].iloc[-1] < split.test_df["timestamp"].iloc[0]


def test_chronological_split_invalid_ratios_raise():
    df = _sample_dataset(10)
    with pytest.raises(ValueError):
        chronological_split(df, train_split=0.8, validation_split=0.3)  # sums >= 1
    with pytest.raises(ValueError):
        chronological_split(df, train_split=1.5, validation_split=0.1)


def test_scaler_fit_on_train_only_does_not_see_val_or_test():
    df = _sample_dataset(100)
    split = chronological_split(df, 0.7, 0.15)
    feature_cols = ["feature_a", "feature_b"]

    scaler = fit_scaler(split.train_df, feature_cols)

    # The scaler's learned mean/std must match ONLY the training slice.
    expected_mean = split.train_df[feature_cols].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-6)

    # Applying it to val/test must use transform (not re-fit) -- verify the
    # scaler object's fitted parameters are unchanged after apply_scaler.
    mean_before = scaler.mean_.copy()
    _ = apply_scaler(split.val_df, feature_cols, scaler)
    _ = apply_scaler(split.test_df, feature_cols, scaler)
    np.testing.assert_allclose(scaler.mean_, mean_before)


def test_apply_scaler_preserves_non_feature_columns():
    df = _sample_dataset(20)
    feature_cols = ["feature_a", "feature_b"]
    scaler = fit_scaler(df, feature_cols)
    scaled = apply_scaler(df, feature_cols, scaler)
    pd.testing.assert_series_equal(scaled["target"], df["target"])
    pd.testing.assert_series_equal(scaled["timestamp"], df["timestamp"])


def test_build_sequences_shapes():
    df = _sample_dataset(20)
    feature_cols = ["feature_a", "feature_b"]
    X, y = build_sequences(df, feature_cols, "target", sequence_length=5)
    # 20 rows, seq_len=5, no carry_over -> first 4 rows can't form a full
    # sequence, so we expect 16 sequences.
    assert X.shape == (16, 5, 2)
    assert y.shape == (16,)


def test_build_sequences_content_matches_manual_window():
    df = _sample_dataset(10)
    feature_cols = ["feature_a", "feature_b"]
    X, y = build_sequences(df, feature_cols, "target", sequence_length=3)
    # First sequence should be rows 0,1,2 and target should be row 2's target.
    expected_window = df[feature_cols].iloc[0:3].to_numpy()
    np.testing.assert_allclose(X[0], expected_window)
    assert y[0] == df["target"].iloc[2]


def test_carry_over_enables_full_length_val_sequences():
    df = _sample_dataset(20)
    feature_cols = ["feature_a", "feature_b"]
    split_point = 15
    train_df = df.iloc[:split_point].reset_index(drop=True)
    val_df = df.iloc[split_point:].reset_index(drop=True)

    seq_len = 5
    carry = get_carry_over(train_df, seq_len)
    assert len(carry) == seq_len - 1

    X_val, y_val = build_sequences(val_df, feature_cols, "target", seq_len, carry_over=carry)
    # With carry-over, every row of val_df should produce a sequence.
    assert X_val.shape[0] == len(val_df)

    # The carried-over context must come from train_df's tail, not val_df.
    expected_first_window = pd.concat([train_df.tail(seq_len - 1), val_df.head(1)])[feature_cols].to_numpy()
    np.testing.assert_allclose(X_val[0], expected_first_window)


def test_build_sequences_without_enough_context_skips_rows():
    df = _sample_dataset(5)
    feature_cols = ["feature_a", "feature_b"]
    X, y = build_sequences(df, feature_cols, "target", sequence_length=10)
    assert X.shape[0] == 0
