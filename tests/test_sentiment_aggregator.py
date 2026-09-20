"""Tests for src.sentiment.aggregator, focusing on window correctness and
the leakage guarantee (a window ending at t never includes items after t)."""

from __future__ import annotations

import pandas as pd

from src.sentiment.aggregator import SENTIMENT_FEATURE_COLUMNS, aggregate_sentiment_to_windows


def _sentiment_df():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:10:00",
                    "2026-01-01 00:40:00",
                    "2026-01-01 01:10:00",
                    "2026-01-01 02:50:00",
                ]
            ),
            "sentiment_label": ["positive", "negative", "neutral", "positive"],
            "sentiment_score": [0.8, -0.6, 0.0, 0.9],
        }
    )


def test_aggregate_basic_window_counts():
    window_ends = pd.Series(pd.to_datetime(["2026-01-01 01:00:00", "2026-01-01 02:00:00", "2026-01-01 03:00:00"]))
    result = aggregate_sentiment_to_windows(_sentiment_df(), window_ends, window="1h")

    assert list(result.columns) == SENTIMENT_FEATURE_COLUMNS
    # Window ending 01:00 covers (00:00, 01:00] -> items at 00:10 and 00:40
    assert result.loc[0, "news_count"] == 2
    assert result.loc[0, "positive_count"] == 1
    assert result.loc[0, "negative_count"] == 1
    # Window ending 02:00 covers (01:00, 02:00] -> item at 01:10 only
    assert result.loc[1, "news_count"] == 1
    assert result.loc[1, "neutral_count"] == 1
    # Window ending 03:00 covers (02:00, 03:00] -> item at 02:50 only
    assert result.loc[2, "news_count"] == 1
    assert result.loc[2, "positive_count"] == 1


def test_aggregate_never_includes_items_after_window_end():
    """The core leakage check: an item published strictly after the window
    end must never be counted in that window."""
    window_ends = pd.Series(pd.to_datetime(["2026-01-01 00:30:00"]))  # covers (2026-01-01 -0:30..? ) window=1h
    result = aggregate_sentiment_to_windows(_sentiment_df(), window_ends, window="1h")
    # Window (2025-12-31 23:30, 2026-01-01 00:30] -> only the 00:10 item qualifies
    assert result.loc[0, "news_count"] == 1
    assert result.loc[0, "positive_count"] == 1


def test_aggregate_empty_window_zero_strategy():
    window_ends = pd.Series(pd.to_datetime(["2026-01-05 00:00:00"]))  # far from any news item
    result = aggregate_sentiment_to_windows(_sentiment_df(), window_ends, window="1h", empty_window_strategy="zero")
    assert result.loc[0, "news_count"] == 0
    assert result.loc[0, "mean_sentiment"] == 0.0


def test_aggregate_empty_window_nan_strategy():
    window_ends = pd.Series(pd.to_datetime(["2026-01-05 00:00:00"]))
    result = aggregate_sentiment_to_windows(_sentiment_df(), window_ends, window="1h", empty_window_strategy="nan")
    assert result.loc[0, "news_count"] == 0
    assert pd.isna(result.loc[0, "mean_sentiment"])


def test_aggregate_no_sentiment_data_at_all():
    empty_sentiment = pd.DataFrame(columns=["timestamp", "sentiment_label", "sentiment_score"])
    window_ends = pd.Series(pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 01:00:00"]))
    result = aggregate_sentiment_to_windows(empty_sentiment, window_ends, window="1h")
    assert len(result) == 2
    assert (result["news_count"] == 0).all()


def test_weighted_sentiment_weights_by_magnitude():
    window_ends = pd.Series(pd.to_datetime(["2026-01-01 01:00:00"]))
    result = aggregate_sentiment_to_windows(_sentiment_df(), window_ends, window="1h")
    # items: 0.8 (positive), -0.6 (negative); weighted by |score|:
    # (0.8*0.8 + -0.6*0.6) / (0.8+0.6) = (0.64 - 0.36) / 1.4
    expected = (0.8 * 0.8 + (-0.6) * 0.6) / (0.8 + 0.6)
    assert abs(result.loc[0, "weighted_sentiment"] - expected) < 1e-9
