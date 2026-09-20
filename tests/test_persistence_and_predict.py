"""Tests for model persistence (save/load) and prediction output format.

Uses a tiny LSTM trained for a handful of steps purely so save/load and the
inference contract can be exercised quickly, without asserting anything
about prediction quality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.timeseries.model import build_model
from src.timeseries.persistence import load_model_bundle, save_model_bundle
from src.timeseries.predict import (
    compute_confidence,
    predict_point,
    predict_with_mc_dropout,
)
from sklearn.preprocessing import StandardScaler


def _tiny_config():
    config = Config()
    config.model.model_type = "lstm"
    config.model.hidden_size = 8
    config.model.num_layers = 1
    config.model.epochs = 1
    config.timeseries.sequence_length = 5
    return config


def test_save_and_load_model_bundle_roundtrip(tmp_path):
    config = _tiny_config()
    n_features = 3
    feature_list = ["f1", "f2", "f3"]

    model = build_model("lstm", n_features=n_features, hidden_size=8, num_layers=1)
    scaler = StandardScaler()
    scaler.fit(np.random.default_rng(0).normal(size=(20, n_features)))

    save_dir = tmp_path / "bundle"
    save_model_bundle(
        save_dir,
        model,
        scaler,
        config,
        feature_list,
        n_features,
        metadata={"use_sentiment": False, "val_residual_std": 0.01},
    )

    loaded_model, loaded_scaler, loaded_config, loaded_features, metadata = load_model_bundle(save_dir)

    assert loaded_features == feature_list
    assert loaded_config.model.hidden_size == 8
    assert metadata["use_sentiment"] is False
    np.testing.assert_allclose(loaded_scaler.mean_, scaler.mean_)

    # Loaded model must produce identical output to the original for the same input.
    x = np.random.default_rng(1).normal(size=(config.timeseries.sequence_length, n_features)).astype(np.float32)
    original_pred = predict_point(model, x)
    loaded_pred = predict_point(loaded_model, x)
    assert original_pred == pytest.approx(loaded_pred, abs=1e-6)


def test_predict_point_returns_scalar_float():
    model = build_model("lstm", n_features=2, hidden_size=4, num_layers=1)
    x = np.random.default_rng(0).normal(size=(5, 2)).astype(np.float32)
    pred = predict_point(model, x)
    assert isinstance(pred, float)


def test_predict_with_mc_dropout_returns_mean_and_std():
    model = build_model("lstm", n_features=2, hidden_size=4, num_layers=1, dropout=0.5)
    x = np.random.default_rng(0).normal(size=(5, 2)).astype(np.float32)
    mean, std = predict_with_mc_dropout(model, x, n_samples=10)
    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert std >= 0.0


def test_compute_confidence_bounded_between_0_and_1():
    assert compute_confidence(mc_std=0.0, val_residual_std=1.0) == 1.0
    c = compute_confidence(mc_std=1.0, val_residual_std=1.0)
    assert 0.0 < c < 1.0
    assert compute_confidence(mc_std=1.0, val_residual_std=0.0) == 0.0
