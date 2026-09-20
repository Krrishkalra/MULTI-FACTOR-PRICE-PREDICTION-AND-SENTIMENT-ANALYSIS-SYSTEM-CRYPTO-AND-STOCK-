"""
Training loop for the LSTM/Transformer regressors, with early stopping,
gradient clipping, and best-checkpoint tracking.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import ModelConfig
from .model import build_model

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    model: nn.Module
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    # shuffle=False is used for validation/test; for training we DO shuffle
    # the *sequences* (each sequence is already a self-contained window of
    # past data), which is standard practice and distinct from shuffling raw
    # time steps -- it does not introduce leakage since each sequence's
    # internal order is preserved and no sequence contains data from beyond
    # its own split.
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_config: ModelConfig,
    batch_size: int = 32,
    device: str = "cpu",
) -> TrainResult:
    """Train a model per ``model_config`` with early stopping on validation loss.

    Returns:
        TrainResult containing the best model (weights restored to the best
        validation epoch), plus the full loss history for plotting.
    """
    n_features = X_train.shape[-1]
    model = build_model(
        model_type=model_config.model_type,
        n_features=n_features,
        hidden_size=model_config.hidden_size,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        bidirectional=model_config.bidirectional,
        num_heads=model_config.num_heads,
        ff_dim=model_config.ff_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=model_config.learning_rate, weight_decay=model_config.weight_decay
    )
    loss_fn = nn.MSELoss()

    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0

    train_losses, val_losses = [], []

    for epoch in range(1, model_config.epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        n_train_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            if model_config.grad_clip_norm:
                nn.utils.clip_grad_norm_(model.parameters(), model_config.grad_clip_norm)
            optimizer.step()
            epoch_train_loss += loss.item()
            n_train_batches += 1
        epoch_train_loss /= max(n_train_batches, 1)

        model.eval()
        epoch_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = loss_fn(preds, yb)
                epoch_val_loss += loss.item()
                n_val_batches += 1
        epoch_val_loss /= max(n_val_batches, 1)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % max(1, model_config.epochs // 20) == 0 or epoch == 1:
            logger.info(
                "Epoch %d/%d - train_loss=%.6f val_loss=%.6f (best=%.6f @ %d)",
                epoch,
                model_config.epochs,
                epoch_train_loss,
                epoch_val_loss,
                best_val_loss,
                best_epoch,
            )

        if patience_counter >= model_config.early_stopping_patience:
            logger.info(
                "Early stopping at epoch %d (no val improvement for %d epochs)",
                epoch,
                model_config.early_stopping_patience,
            )
            break

    model.load_state_dict(best_state)
    return TrainResult(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
    )
