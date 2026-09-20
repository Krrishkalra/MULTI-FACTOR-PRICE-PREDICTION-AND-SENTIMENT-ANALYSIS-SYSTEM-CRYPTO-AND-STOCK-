"""
Align market data (with engineered technical features) and aggregated
sentiment features into a single modeling dataset, and construct the
prediction target.

Leakage-prevention notes:
    - Sentiment for the bar at time ``t`` is aggregated over the window
      ``(t - window, t]`` (see sentiment/aggregator.py) -- it never uses
      news published after ``t``.
    - The target for row ``t`` is derived from ``close`` at
      ``t + prediction_horizon``, i.e. it is shifted *backwards* onto the
      feature row at ``t`` using ``.shift(-horizon)``. This is the only place
      in the codebase where a "future" value is intentionally attached to a
      row, and it happens here -- once -- rather than being scattered across
      feature functions, which makes the leakage boundary easy to audit.
    - Rows without a valid target (the last ``horizon`` rows of the series,
      where the future close does not exist) are dropped.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import SentimentConfig
from ..sentiment.aggregator import aggregate_sentiment_to_windows
from ..sentiment.sentiment_model import SentimentScorer
from ..sentiment.preprocessing import preprocess_news_dataframe

logger = logging.getLogger(__name__)


def build_sentiment_features(
    news_df: pd.DataFrame,
    market_timestamps: pd.Series,
    sentiment_config: SentimentConfig,
    scorer: SentimentScorer | None = None,
) -> pd.DataFrame:
    """Run the full sentiment sub-pipeline: preprocess -> score -> aggregate.

    Args:
        news_df: raw news/social DataFrame with ``timestamp`` and ``text``.
        market_timestamps: the market data's timestamp column; each value is
            treated as the end of one aggregation window.
        sentiment_config: controls the model used and aggregation window.
        scorer: optional pre-instantiated ``SentimentScorer`` (useful for
            reusing an already-loaded model across calls, e.g. in tests).

    Returns:
        DataFrame aligned 1:1 with ``market_timestamps``, containing the
        columns defined in ``sentiment.aggregator.SENTIMENT_FEATURE_COLUMNS``.
    """
    if news_df is None or news_df.empty:
        logger.warning("No news data supplied; sentiment features will be empty/neutral for all rows")
        return aggregate_sentiment_to_windows(
            pd.DataFrame(columns=["timestamp", "sentiment_label", "sentiment_score"]),
            market_timestamps,
            window=sentiment_config.aggregation_window,
            empty_window_strategy=sentiment_config.empty_window_strategy,
        )

    cleaned = preprocess_news_dataframe(news_df)
    scorer = scorer or SentimentScorer(sentiment_config)
    scored = scorer.score_dataframe(cleaned)
    return aggregate_sentiment_to_windows(
        scored,
        market_timestamps,
        window=sentiment_config.aggregation_window,
        empty_window_strategy=sentiment_config.empty_window_strategy,
    )


def merge_market_and_sentiment(market_df: pd.DataFrame, sentiment_features: pd.DataFrame) -> pd.DataFrame:
    """Concatenate market (+ technical features) with aggregated sentiment
    features. Both frames must already be row-aligned (same length/order).
    """
    if len(market_df) != len(sentiment_features):
        raise ValueError(
            f"Market ({len(market_df)}) and sentiment ({len(sentiment_features)}) "
            "frames must have the same number of rows to merge by position"
        )
    market_df = market_df.reset_index(drop=True)
    sentiment_features = sentiment_features.reset_index(drop=True)
    return pd.concat([market_df, sentiment_features], axis=1)


def add_target(
    df: pd.DataFrame,
    price_col: str = "close",
    target_type: str = "return",
    horizon: int = 1,
) -> pd.DataFrame:
    """Attach the prediction target to each row and drop rows without one.

    Args:
        df: aligned market+sentiment DataFrame, sorted ascending by time.
        price_col: column to base the target on.
        target_type: "price" -> target is the future close price directly;
            "return" -> target is (future_close - current_close) / current_close.
        horizon: number of periods ahead to predict.

    Returns:
        DataFrame with a ``target`` column added; the trailing ``horizon``
        rows (which have no future value to form a target from) are dropped.

    See README section "Target Variable" for the discussion of price vs
    return as the prediction target -- summary: return is generally
    preferred because it is stationary (roughly constant statistical
    properties over time) which most ML/DL models handle far better than a
    raw, trending price level; price is kept as an option for
    completeness/comparison.
    """
    df = df.copy()
    future_price = df[price_col].shift(-horizon)
    if target_type == "price":
        df["target"] = future_price
    elif target_type == "return":
        df["target"] = (future_price - df[price_col]) / df[price_col]
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    before = len(df)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    logger.info(
        "Added '%s' target (horizon=%d); dropped %d trailing rows with no future value",
        target_type,
        horizon,
        before - len(df),
    )
    return df


def drop_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leading rows that contain NaNs from rolling-window warm-up.

    Must be called AFTER add_target (target NaNs are handled there) and
    ideally right before the sequence/train-val-test split step, so that no
    NaNs reach the model.
    """
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d warm-up rows containing NaN feature values", dropped)
    return df
