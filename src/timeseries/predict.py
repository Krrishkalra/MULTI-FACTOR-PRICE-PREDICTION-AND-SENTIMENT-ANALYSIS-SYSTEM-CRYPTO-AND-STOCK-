"""
Inference-time prediction helpers: point predictions plus a documented,
non-fabricated uncertainty/confidence estimate via MC Dropout.

Confidence methodology (IMPORTANT -- read before interpreting output):
    The trained regressor produces a single point estimate; it is not a
    probabilistic model and does not natively output a calibrated
    probability of being correct. To avoid fabricating a confidence number,
    we use Monte Carlo Dropout (Gal & Ghahramani, 2016): at inference time we
    keep dropout ACTIVE (model in train() mode for dropout layers only) and
    run N stochastic forward passes over the same input. The spread (std) of
    these N predictions approximates the model's epistemic uncertainty for
    that specific input.

    We then convert that std into a bounded [0, 1] "confidence" score by
    comparing it to the typical residual spread observed on the held-out
    validation set:

        confidence = 1 / (1 + mc_std / val_residual_std)

    This yields confidence -> 1 when the model's MC-dropout spread for this
    input is much smaller than its typical validation error (i.e. the model
    is relatively certain by its own internal measure), and confidence -> 0
    when the spread is large relative to typical error. This is a heuristic,
    relative uncertainty signal -- it is explicitly NOT a calibrated
    probability of the prediction being correct, and callers must not treat
    it as one. The ``confidence_method`` field in ``PredictionResult``
    documents this explicitly so downstream consumers (e.g. the FastAPI
    backend) can decide how to present it.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import nn

CONFIDENCE_METHOD_DESCRIPTION = (
    "mc_dropout_relative: confidence = 1 / (1 + mc_dropout_std / validation_residual_std). "
    "This is a heuristic relative-uncertainty signal derived from Monte Carlo Dropout, "
    "NOT a calibrated probability of correctness."
)


def predict_point(model: nn.Module, x_sequence: np.ndarray, device: str = "cpu") -> float:
    """Single deterministic forward pass (dropout disabled).

    Args:
        model: a trained LSTMRegressor / TransformerRegressor.
        x_sequence: array of shape (sequence_length, n_features) OR
            (1, sequence_length, n_features).
    """
    model.eval()
    x = _as_batch_tensor(x_sequence, device)
    with torch.no_grad():
        pred = model(x)
    return float(pred.item())


def predict_with_mc_dropout(
    model: nn.Module,
    x_sequence: np.ndarray,
    n_samples: int = 30,
    device: str = "cpu",
) -> Tuple[float, float]:
    """Run N stochastic forward passes with dropout active to estimate
    epistemic uncertainty for a single input sequence.

    Returns:
        (mean_prediction, std_of_predictions)
    """
    _enable_dropout_only(model)
    x = _as_batch_tensor(x_sequence, device)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(float(model(x).item()))
    model.eval()
    return float(np.mean(preds)), float(np.std(preds))


def compute_confidence(mc_std: float, val_residual_std: float) -> float:
    """Convert an MC-dropout std into a bounded, relative confidence score.

    See module docstring for the full methodology and its limitations.
    """
    if val_residual_std <= 0:
        # No meaningful reference scale (e.g. degenerate validation set);
        # cannot produce a meaningful relative confidence.
        return 0.0
    return float(1.0 / (1.0 + (mc_std / val_residual_std)))


def _as_batch_tensor(x_sequence: np.ndarray, device: str) -> torch.Tensor:
    x = np.asarray(x_sequence, dtype=np.float32)
    if x.ndim == 2:
        x = x[np.newaxis, ...]
    return torch.tensor(x, dtype=torch.float32, device=device)


def _enable_dropout_only(model: nn.Module) -> None:
    """Set the model to eval() overall but re-enable train mode on Dropout
    submodules specifically, so BatchNorm-like layers (if any were added in
    the future) don't get affected by MC sampling -- only dropout should.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
