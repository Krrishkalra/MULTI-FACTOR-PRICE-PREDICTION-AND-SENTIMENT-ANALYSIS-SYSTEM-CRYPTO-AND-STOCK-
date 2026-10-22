"""
End-to-end training pipeline: raw CSV/JSON -> cleaned market data -> features
-> sentiment (optional) -> aligned dataset -> chronological split -> scaling
-> sequences -> baselines -> LSTM/Transformer training -> evaluation ->
persistence.

Usage:
    python -m src.pipeline.training_pipeline --use-sentiment --run-name price_plus_sentiment
    python -m src.pipeline.training_pipeline --run-name price_only

Run twice (with and without --use-sentiment) to reproduce the Experiment A
(price-only) vs Experiment B (price+sentiment) comparison described in the
README.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import Config, PRICE_MODEL_DIR, SAMPLE_DATA_DIR, get_default_config, set_global_seed
from ..data.alignment import add_target, build_sentiment_features, drop_warmup_rows, merge_market_and_sentiment
from ..data.cleaner import clean_market_data
from ..data.feature_engineering import engineer_features, get_engineered_feature_names
from ..data.loader import load_market_data_csv, load_news_data_json
from ..evaluation.metrics import evaluate_predictions, save_metrics_report
from ..evaluation.plots import plot_actual_vs_predicted, plot_loss_curves, plot_prediction_errors
from ..sentiment.aggregator import SENTIMENT_FEATURE_COLUMNS
from ..timeseries.baselines import NaiveBaseline, predict_random_forest_baseline, train_random_forest_baseline
from ..timeseries.dataset import apply_scaler, build_sequences, chronological_split, fit_scaler, get_carry_over
from ..timeseries.persistence import save_model_bundle
from ..timeseries.predict import predict_with_mc_dropout
from ..timeseries.train import train_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def build_dataset(
    market_csv: Path,
    news_json: Optional[Path],
    config: Config,
    use_sentiment: bool,
) -> tuple[pd.DataFrame, List[str]]:
    """Build the fully aligned, feature-engineered, target-attached dataset.

    Returns:
        (dataset_df, feature_cols) -- feature_cols excludes timestamp/target/
        raw OHLCV price columns not meant to feed the model directly (open/
        high/low/close/volume ARE included as features here since they carry
        information; only 'timestamp', 'target', and helper columns like
        'is_return_outlier' are excluded).
    """
    market_df = load_market_data_csv(market_csv)
    market_df = clean_market_data(market_df, outlier_z_threshold=config.data.outlier_z_threshold)
    market_df = engineer_features(
        market_df,
        ma_windows=config.data.ma_windows,
        ema_windows=config.data.ema_windows,
        rsi_window=config.data.rsi_window,
        macd_fast=config.data.macd_fast,
        macd_slow=config.data.macd_slow,
        macd_signal=config.data.macd_signal,
        volatility_window=config.data.volatility_window,
    )

    technical_features = ["open", "high", "low", "close", "volume"] + get_engineered_feature_names(
        ma_windows=config.data.ma_windows,
        ema_windows=config.data.ema_windows,
        rsi_window=config.data.rsi_window,
        volatility_window=config.data.volatility_window,
    )

    if use_sentiment:
        if news_json is None:
            raise ValueError("use_sentiment=True requires a news_json path")
        news_df = load_news_data_json(news_json)
        sentiment_features_df = build_sentiment_features(news_df, market_df["timestamp"], config.sentiment)
        dataset = merge_market_and_sentiment(market_df, sentiment_features_df)
        feature_cols = technical_features + SENTIMENT_FEATURE_COLUMNS
    else:
        dataset = market_df.copy()
        feature_cols = technical_features

    dataset = add_target(
        dataset,
        price_col="close",
        target_type=config.timeseries.target_type,
        horizon=config.timeseries.prediction_horizon,
    )
    # Drop helper/non-feature columns before warm-up NaN dropping so they
    # don't affect the dropna() decision.
    keep_cols = ["timestamp"] + feature_cols + ["target"]
    dataset = dataset[keep_cols]
    dataset = drop_warmup_rows(dataset)

    return dataset, feature_cols


def run_training_pipeline(
    market_csv: Path,
    news_json: Optional[Path],
    config: Config,
    use_sentiment: bool,
    run_name: str,
    output_root: Path = PRICE_MODEL_DIR,
    reports_root: Optional[Path] = None,
) -> dict:
    """Run the full pipeline and return a dict of results (metrics for naive,
    random forest, and deep model, plus paths of saved artifacts).
    """
    set_global_seed(config.random_seed)
    reports_root = reports_root or (output_root.parent.parent / "reports" / run_name)

    dataset, feature_cols = build_dataset(market_csv, news_json, config, use_sentiment)
    logger.info("Final dataset: %d rows, %d feature columns", len(dataset), len(feature_cols))

    split = chronological_split(dataset, config.timeseries.train_split, config.timeseries.validation_split)

    scaler = fit_scaler(split.train_df, feature_cols)
    train_scaled = apply_scaler(split.train_df, feature_cols, scaler)
    val_scaled = apply_scaler(split.val_df, feature_cols, scaler)
    test_scaled = apply_scaler(split.test_df, feature_cols, scaler)

    seq_len = config.timeseries.sequence_length
    X_train, y_train = build_sequences(train_scaled, feature_cols, "target", seq_len)

    val_carry = get_carry_over(train_scaled, seq_len)
    X_val, y_val = build_sequences(val_scaled, feature_cols, "target", seq_len, carry_over=val_carry)

    test_carry = get_carry_over(val_scaled, seq_len)
    X_test, y_test = build_sequences(test_scaled, feature_cols, "target", seq_len, carry_over=test_carry)

    logger.info(
        "Sequence shapes -> train: %s, val: %s, test: %s", X_train.shape, X_val.shape, X_test.shape
    )

    results: dict = {"run_name": run_name, "use_sentiment": use_sentiment, "n_features": len(feature_cols)}

    # --- Baseline 1: naive ---
    naive = NaiveBaseline(target_type=config.timeseries.target_type)
    naive_preds = naive.predict(np.zeros(len(y_test)))
    naive_metrics = evaluate_predictions(
        y_test, naive_preds, include_direction_classification=(config.timeseries.target_type == "return")
    )
    results["naive"] = naive_metrics.to_dict()
    logger.info("Naive baseline metrics: %s", naive_metrics.to_dict())

    # --- Baseline 2: RandomForest on tabular (non-sequential) features ---
    # Uses the same target-aligned rows as the sequence data would (i.e. the
    # split.*_df target column), NOT the sequence-shaped arrays.
    rf_model = train_random_forest_baseline(split.train_df.assign(**{c: train_scaled[c] for c in feature_cols}), feature_cols)
    rf_test_df = split.test_df.assign(**{c: test_scaled[c] for c in feature_cols})
    rf_preds = predict_random_forest_baseline(rf_model, rf_test_df, feature_cols)
    rf_metrics = evaluate_predictions(
        split.test_df["target"].to_numpy(),
        rf_preds,
        include_direction_classification=(config.timeseries.target_type == "return"),
    )
    results["random_forest"] = rf_metrics.to_dict()
    logger.info("RandomForest baseline metrics: %s", rf_metrics.to_dict())

    # --- Deep model (LSTM or Transformer per config) ---
    train_result = train_model(
        X_train, y_train, X_val, y_val, config.model, batch_size=config.timeseries.batch_size
    )

    deep_preds = []
    for i in range(len(X_test)):
        from ..timeseries.predict import predict_point

        deep_preds.append(predict_point(train_result.model, X_test[i]))
    deep_preds = np.array(deep_preds)

    deep_metrics = evaluate_predictions(
        y_test, deep_preds, include_direction_classification=(config.timeseries.target_type == "return")
    )
    results[config.model.model_type] = deep_metrics.to_dict()
    logger.info("%s metrics: %s", config.model.model_type, deep_metrics.to_dict())

    # --- Plots ---
    reports_root.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(
        train_result.train_losses,
        train_result.val_losses,
        f"{run_name} - Training/Validation Loss",
        reports_root / "loss_curve.png",
    )
    plot_actual_vs_predicted(
        y_test, deep_preds, f"{run_name} - Actual vs Predicted ({config.timeseries.target_type})",
        reports_root / "actual_vs_predicted.png",
    )
    plot_prediction_errors(y_test, deep_preds, f"{run_name} - Prediction Errors", reports_root / "prediction_errors.png")

    save_metrics_report(
        {"naive": naive_metrics, "random_forest": rf_metrics, config.model.model_type: deep_metrics},
        reports_root / "metrics.json",
    )

    # --- Persistence ---
    save_dir = output_root / run_name
    val_residual_std = float(np.std(np.array(y_val) - np.array(
        [predict_with_mc_dropout(train_result.model, X_val[i], n_samples=1)[0] for i in range(len(X_val))]
    ))) if len(X_val) else 0.0

    save_model_bundle(
        save_dir=save_dir,
        model=train_result.model,
        scaler=scaler,
        config=config,
        feature_list=feature_cols,
        n_features=len(feature_cols),
        metadata={
            "run_name": run_name,
            "use_sentiment": use_sentiment,
            "best_epoch": train_result.best_epoch,
            "best_val_loss": train_result.best_val_loss,
            "val_residual_std": val_residual_std,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "metrics": {
                "naive": naive_metrics.to_dict(),
                "random_forest": rf_metrics.to_dict(),
                config.model.model_type: deep_metrics.to_dict(),
            },
        },
    )

    results["save_dir"] = str(save_dir)
    results["reports_dir"] = str(reports_root)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the price prediction model.")
    parser.add_argument("--market-csv", type=Path, default=SAMPLE_DATA_DIR / "market_data.csv")
    parser.add_argument("--news-json", type=Path, default=SAMPLE_DATA_DIR / "news_data.json")
    parser.add_argument("--use-sentiment", action="store_true")
    parser.add_argument("--run-name", type=str, default="default_run")
    parser.add_argument("--model-type", type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--asset", type=str, default=None, help="Asset label stored in the bundle config, e.g. BTC-USD")
    parser.add_argument("--sentiment-window", type=str, default=None, help="Sentiment aggregation window, e.g. 24h")
    args = parser.parse_args()

    config = get_default_config()
    config.model.model_type = args.model_type
    if args.epochs is not None:
        config.model.epochs = args.epochs
    if args.asset is not None:
        config.data.asset = args.asset
    if args.sentiment_window is not None:
        config.sentiment.aggregation_window = args.sentiment_window

    results = run_training_pipeline(
        market_csv=args.market_csv,
        news_json=args.news_json if args.use_sentiment else None,
        config=config,
        use_sentiment=args.use_sentiment,
        run_name=args.run_name,
    )
    logger.info("Training pipeline complete: %s", results)


if __name__ == "__main__":
    main()
