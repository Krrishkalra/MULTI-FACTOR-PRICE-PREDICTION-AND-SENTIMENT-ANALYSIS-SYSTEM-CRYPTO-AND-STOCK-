"""Integration test for src.pipeline.inference_pipeline.InferencePipeline.

Requires a previously trained (price-only) model bundle under
models/price_prediction/experiment_a_price_only, produced by running:
    python -m src.pipeline.training_pipeline --run-name experiment_a_price_only

If that bundle is not present (e.g. fresh checkout without having run
training yet), this test is skipped rather than failed, since generating it
is a training run, not a unit-test-time operation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import PRICE_MODEL_DIR, SAMPLE_DATA_DIR
from src.data.loader import load_market_data_csv
from src.pipeline.inference_pipeline import InferencePipeline
from src.schemas import PredictionResult

MODEL_DIR = PRICE_MODEL_DIR / "experiment_a_price_only"


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Requires a trained model bundle at models/price_prediction/experiment_a_price_only")
def test_inference_pipeline_returns_well_formed_prediction_result():
    pipeline = InferencePipeline(MODEL_DIR)
    market_df = load_market_data_csv(SAMPLE_DATA_DIR / "market_data.csv")
    window = market_df.tail(pipeline.config.timeseries.sequence_length + 50)

    result = pipeline.predict(window)

    assert isinstance(result, PredictionResult)
    assert isinstance(result.current_price, float)
    assert isinstance(result.predicted_price, float)
    assert isinstance(result.predicted_return, float)
    assert result.direction in ("UP", "DOWN")
    assert 0.0 <= result.confidence <= 1.0
    assert "NOT a calibrated probability" in result.confidence_method
    # Direction must be consistent with the sign of the predicted return.
    if result.predicted_return >= 0:
        assert result.direction == "UP"
    else:
        assert result.direction == "DOWN"


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Requires a trained model bundle at models/price_prediction/experiment_a_price_only")
def test_inference_pipeline_raises_on_insufficient_rows():
    pipeline = InferencePipeline(MODEL_DIR)
    market_df = load_market_data_csv(SAMPLE_DATA_DIR / "market_data.csv")
    too_few = market_df.tail(5)  # fewer than sequence_length after feature warm-up
    with pytest.raises(ValueError):
        pipeline.predict(too_few)
