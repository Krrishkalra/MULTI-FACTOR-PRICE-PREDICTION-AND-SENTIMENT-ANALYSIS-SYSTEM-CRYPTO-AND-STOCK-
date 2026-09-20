"""
Aggregate per-item sentiment scores into fixed time windows so they can be
merged with market data (which is itself bucketed into periods, e.g. hourly
bars).

Leakage-prevention note:
    A market bar timestamped ``t`` in this project represents the period
    ending at ``t`` (standard OHLCV convention: the bar covering
    (t - period, t]). To avoid leaking information from the future into the
    features used to predict what happens after ``t``, the sentiment window
    aggregated for bar ``t`` must only include news/social items with
    ``timestamp <= t`` (and, to bound it, ``timestamp > t - window``). This
    module implements exactly that half-open interval and never looks ahead
    of ``t``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SENTIMENT_FEATURE_COLUMNS = [
    "mean_sentiment",
    "weighted_sentiment",
    "positive_count",
    "negative_count",
    "neutral_count",
    "news_count",
    "sentiment_std",
]


def aggregate_sentiment_to_windows(
    sentiment_df: pd.DataFrame,
    window_ends: pd.Series,
    window: str = "1h",
    empty_window_strategy: str = "zero",
) -> pd.DataFrame:
    """Aggregate scored sentiment items into the time windows ending at each
    timestamp in ``window_ends``.

    Args:
        sentiment_df: DataFrame with columns ``timestamp``, ``sentiment_label``,
            ``sentiment_score`` (output of SentimentScorer.score_dataframe).
        window_ends: Series of timestamps, one per market bar, representing
            the end of each aggregation window (typically the market data's
            ``timestamp`` column). Must be sorted ascending.
        window: pandas offset alias for the aggregation window width, e.g.
            "1h", "30min", "1D". Should typically match the market data's bar
            frequency.
        empty_window_strategy: "zero" fills numeric sentiment features with 0
            (neutral / no signal) for windows with no news; "nan" leaves them
            as NaN.

    Returns:
        DataFrame aligned 1:1 with ``window_ends`` (same index/order),
        containing the columns in ``SENTIMENT_FEATURE_COLUMNS``.
    """
    window_td = pd.Timedelta(window)
    window_ends = pd.to_datetime(window_ends).reset_index(drop=True)

    if sentiment_df.empty:
        logger.warning("No sentiment data provided; all windows will be empty")
        return _empty_result(window_ends, empty_window_strategy)

    sdf = sentiment_df.copy()
    sdf["timestamp"] = pd.to_datetime(sdf["timestamp"])
    sdf = sdf.sort_values("timestamp").reset_index(drop=True)

    ts = sdf["timestamp"].values
    scores = sdf["sentiment_score"].to_numpy(dtype=float)
    labels = sdf["sentiment_label"].to_numpy()

    rows = []
    # searchsorted gives O(log n) window boundaries per bar since both
    # window_ends and ts are sorted -- overall O(n log n), fine for the
    # dataset sizes this project targets locally.
    starts = (window_ends - window_td).values
    ends = window_ends.values

    left_idx = np.searchsorted(ts, starts, side="right")  # timestamp > start
    right_idx = np.searchsorted(ts, ends, side="right")  # timestamp <= end

    for lo, hi in zip(left_idx, right_idx):
        if hi <= lo:
            rows.append(_empty_row(empty_window_strategy))
            continue
        window_scores = scores[lo:hi]
        window_labels = labels[lo:hi]
        n = len(window_scores)
        pos = int((window_labels == "positive").sum())
        neg = int((window_labels == "negative").sum())
        neu = int((window_labels == "neutral").sum())

        mean_sent = float(np.mean(window_scores))
        std_sent = float(np.std(window_scores)) if n > 1 else 0.0
        # Weighted sentiment: weight each item by |score| so strongly
        # positive/negative items count more than near-zero ones, then
        # renormalize by the sum of weights (falls back to mean if all
        # weights are zero, i.e. everything neutral).
        weights = np.abs(window_scores)
        weight_sum = weights.sum()
        weighted_sent = float(np.sum(window_scores * weights) / weight_sum) if weight_sum > 0 else 0.0

        rows.append(
            {
                "mean_sentiment": mean_sent,
                "weighted_sentiment": weighted_sent,
                "positive_count": pos,
                "negative_count": neg,
                "neutral_count": neu,
                "news_count": n,
                "sentiment_std": std_sent,
            }
        )

    result = pd.DataFrame(rows, columns=SENTIMENT_FEATURE_COLUMNS)
    n_empty = int((result["news_count"] == 0).sum())
    logger.info(
        "Aggregated sentiment into %d windows (%d had no news items, strategy=%s)",
        len(result),
        n_empty,
        empty_window_strategy,
    )
    return result


def _empty_row(strategy: str) -> dict:
    if strategy == "nan":
        fill = np.nan
    else:
        fill = 0.0
    return {
        "mean_sentiment": fill,
        "weighted_sentiment": fill,
        "positive_count": 0,
        "negative_count": 0,
        "neutral_count": 0,
        "news_count": 0,
        "sentiment_std": fill,
    }


def _empty_result(window_ends: pd.Series, strategy: str) -> pd.DataFrame:
    rows = [_empty_row(strategy) for _ in range(len(window_ends))]
    return pd.DataFrame(rows, columns=SENTIMENT_FEATURE_COLUMNS)
