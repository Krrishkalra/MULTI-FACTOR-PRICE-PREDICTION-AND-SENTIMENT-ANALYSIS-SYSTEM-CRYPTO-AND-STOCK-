"""
Plotting utilities for model evaluation: actual vs predicted, training/
validation loss curves, and prediction error distributions.

Matplotlib's default "Agg"-friendly usage (figures are saved to disk, not
shown interactively) so this works in headless/CI environments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot to %s", path)


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_true, label="Actual", linewidth=1.2)
    ax.plot(y_pred, label="Predicted", linewidth=1.0, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time step (test set order)")
    ax.set_ylabel("Value")
    ax.legend()
    _save(fig, path)


def plot_loss_curves(train_losses: List[float], val_losses: List[float], title: str, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train loss")
    ax.plot(val_losses, label="Validation loss")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    _save(fig, path)


def plot_prediction_errors(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: str | Path) -> None:
    errors = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors, bins=40, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Prediction error (predicted - actual)")
    ax.set_ylabel("Count")
    _save(fig, path)
