"""
Inference pipeline: the single entry point the future FastAPI backend should
call. It accepts raw market data and raw news/social data (in the schemas
defined in ``src.schemas``), internally reproduces the exact preprocessing/
feature-engineering/scaling steps used at training time, and returns a
``PredictionResult``.

The backend never needs to know about sequence windows, scalers, feature
lists, or model internals -- it only needs to provide data matching
``MarketDataPoint`` / ``NewsItem`` and consume ``PredictionResult``.

Usage (CLI, for local testing without a backend):
    python -m src.pipeline.inference_pipeline --model-dir models/price_prediction/default_run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import PRICE_MODEL_DIR, SAMPLE_DATA_DIR
from ..data.alignment import build_sentiment_features, merge_market_and_sentiment
from ..data.cleaner import clean_market_data
from ..data.feature_engineering import engineer_features
from ..data.loader import load_market_data_csv, load_market_data_from_records, load_news_data_from_records
from ..schemas import MarketDataPoint, NewsItem, PredictionResult
from ..sentiment.aggregator import SENTIMENT_FEATURE_COLUMNS
from ..sentiment.sentiment_model import SentimentScorer
from ..timeseries.persistence import load_model_bundle
from ..timeseries.predict import CONFIDENCE_METHOD_DESCRIPTION, compute_confidence, predict_point, predict_with_mc_dropout

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Loads a trained model bundle once and serves repeated predictions.

    Instantiate this once (e.g. as a FastAPI app-level singleton/dependency)
    and call ``.predict(...)`` per request; the model/scaler are not reloaded
    on every call.
    """

    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.model, self.scaler, self.config, self.feature_list, self.metadata = load_model_bundle(
            self.model_dir, device=device
        )
        self.use_sentiment = self.metadata.get("use_sentiment", False)
        self.val_residual_std = float(self.metadata.get("val_residual_std", 0.0))
        # Loaded lazily on first use, then reused so FinBERT is not reloaded per request.
        self.scorer = SentimentScorer(self.config.sentiment) if self.use_sentiment else None

    def predict(
        self,
        market_data: List[MarketDataPoint] | pd.DataFrame,
        sentiment_data: Optional[List[NewsItem]] = None,
        mc_dropout_samples: int = 30,
    ) -> PredictionResult:
        """Produce a prediction from raw market (+ optional sentiment) data.

        Args:
            market_data: at least ``sequence_length`` consecutive OHLCV bars,
                either as a list of ``MarketDataPoint`` or an already-built
                DataFrame with the standard columns.
            sentiment_data: optional list of ``NewsItem``; required if the
                loaded model was trained with sentiment features
                (``self.use_sentiment``). Ignored otherwise.
            mc_dropout_samples: number of stochastic forward passes used to
                estimate the confidence score (see predict.py docstring).

        Returns:
            PredictionResult with a non-fabricated, documented confidence
            estimate.

        Raises:
            ValueError: if fewer than ``sequence_length`` valid rows remain
                after cleaning/feature engineering, or if sentiment data is
                required but not provided.
        """
        if isinstance(market_data, pd.DataFrame):
            market_df = market_data.copy()
        else:
            market_df = load_market_data_from_records(market_data)

        market_df = clean_market_data(market_df)
        market_df = engineer_features(
            market_df,
            ma_windows=self.config.data.ma_windows,
            ema_windows=self.config.data.ema_windows,
            rsi_window=self.config.data.rsi_window,
            macd_fast=self.config.data.macd_fast,
            macd_slow=self.config.data.macd_slow,
            macd_signal=self.config.data.macd_signal,
            volatility_window=self.config.data.volatility_window,
        )

        if self.use_sentiment:
            if sentiment_data is None:
                raise ValueError("This model was trained with sentiment features; sentiment_data is required")
            news_df = load_news_data_from_records(sentiment_data)
            sentiment_features_df = build_sentiment_features(
                news_df, market_df["timestamp"], self.config.sentiment, scorer=self.scorer
            )
            full_df = merge_market_and_sentiment(market_df, sentiment_features_df)
        else:
            full_df = market_df

        full_df = full_df.dropna().reset_index(drop=True)
        seq_len = self.config.timeseries.sequence_length
        if len(full_df) < seq_len:
            raise ValueError(
                f"Need at least {seq_len} valid rows after preprocessing to build a sequence, got {len(full_df)}"
            )

        window_df = full_df.iloc[-seq_len:]
        scaled = window_df.copy()
        scaled[self.feature_list] = self.scaler.transform(window_df[self.feature_list].to_numpy(dtype=float))
        x_sequence = scaled[self.feature_list].to_numpy(dtype=float)

        current_price = float(window_df["close"].iloc[-1])
        point_pred = predict_point(self.model, x_sequence, device=self.device)
        mc_mean, mc_std = predict_with_mc_dropout(self.model, x_sequence, n_samples=mc_dropout_samples, device=self.device)
        confidence = compute_confidence(mc_std, self.val_residual_std)

        if self.config.timeseries.target_type == "return":
            predicted_return = point_pred
            predicted_price = current_price * (1 + predicted_return)
        else:
            predicted_price = point_pred
            predicted_return = (predicted_price - current_price) / current_price

        direction = "UP" if predicted_return >= 0 else "DOWN"

        sentiment_score = None
        if self.use_sentiment and "mean_sentiment" in window_df.columns:
            sentiment_score = float(window_df["mean_sentiment"].iloc[-1])

        return PredictionResult(
            current_price=current_price,
            predicted_price=float(predicted_price),
            predicted_return=float(predicted_return),
            direction=direction,
            confidence=confidence,
            confidence_method=CONFIDENCE_METHOD_DESCRIPTION,
            sentiment_score=sentiment_score,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained model bundle.")
    parser.add_argument("--model-dir", type=Path, default=PRICE_MODEL_DIR / "default_run")
    parser.add_argument("--market-csv", type=Path, default=SAMPLE_DATA_DIR / "market_data.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    pipeline = InferencePipeline(args.model_dir)
    market_df = load_market_data_csv(args.market_csv)
    result = pipeline.predict(market_df.tail(pipeline.config.timeseries.sequence_length + 50))

    logger.info("Prediction result: %s", result)
    print(result)


if __name__ == "__main__":
    main()
