"""Tests for src.evaluation.metrics."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    directional_accuracy,
    evaluate_predictions,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r_squared,
    root_mean_squared_error,
)


def test_mae_and_rmse_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])
    assert mean_absolute_error(y_true, y_pred) == (0 + 0 + 2) / 3
    assert root_mean_squared_error(y_true, y_pred) == np.sqrt((0 + 0 + 4) / 3)


def test_mape_handles_near_zero_true_values():
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.1, 1.1])
    # should not raise / produce inf due to epsilon guard
    result = mean_absolute_percentage_error(y_true, y_pred)
    assert np.isfinite(result)


def test_r_squared_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    assert r_squared(y_true, y_true) == 1.0


def test_directional_accuracy_all_correct():
    y_true = np.array([0.01, -0.02, 0.03])
    y_pred = np.array([0.02, -0.01, 0.05])
    assert directional_accuracy(y_true, y_pred) == 1.0


def test_directional_accuracy_all_wrong():
    y_true = np.array([0.01, -0.02, 0.03])
    y_pred = np.array([-0.02, 0.01, -0.05])
    assert directional_accuracy(y_true, y_pred) == 0.0


def test_evaluate_predictions_with_direction_classification():
    y_true = np.array([0.01, -0.02, 0.03, -0.01])
    y_pred = np.array([0.02, -0.01, -0.01, -0.02])
    metrics = evaluate_predictions(y_true, y_pred, include_direction_classification=True)
    assert metrics.n_samples == 4
    assert metrics.precision_up is not None
    assert metrics.f1_down is not None
    assert 0.0 <= metrics.directional_accuracy <= 1.0


def test_evaluate_predictions_without_direction_classification_leaves_none():
    y_true = np.array([100.0, 101.0])
    y_pred = np.array([99.0, 102.0])
    metrics = evaluate_predictions(y_true, y_pred, include_direction_classification=False)
    assert metrics.precision_up is None
    assert metrics.f1_down is None
