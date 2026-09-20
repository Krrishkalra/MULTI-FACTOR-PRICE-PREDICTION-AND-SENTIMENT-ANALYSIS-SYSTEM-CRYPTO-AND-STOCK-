"""
PyTorch model definitions: an LSTM regressor and an (optional/experimental)
Transformer encoder regressor, both predicting a single scalar (future price
or future return) from a sequence of feature vectors.

Both models share the same input/output contract:
    input:  (batch, sequence_length, n_features)
    output: (batch,)  -- a single scalar prediction per sequence

so that training/evaluation code (train.py, predict.py) does not need to
branch on model type beyond construction.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """LSTM-based sequence regressor.

    Architecture: multi-layer LSTM -> dropout -> take the last time step's
    hidden state -> linear projection to a single scalar output.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]  # (batch, hidden_size * num_directions)
        last_step = self.dropout(last_step)
        return self.head(last_step).squeeze(-1)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for the Transformer encoder."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerRegressor(nn.Module):
    """Transformer-encoder-based sequence regressor (experimental).

    Architecture: linear input projection -> positional encoding ->
    TransformerEncoder stack -> mean-pool over time -> linear head.

    This is offered as an alternative to the LSTM, selectable via
    ``ModelConfig.model_type = "transformer"``. It is NOT assumed to
    outperform the LSTM -- the two should be compared empirically on the
    same data (see evaluation module and README experiment results).
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_heads: int = 4,
        ff_dim: int = 128,
        max_len: int = 512,
    ):
        super().__init__()
        # hidden_size doubles as d_model here; must be divisible by num_heads.
        if hidden_size % num_heads != 0:
            hidden_size = num_heads * max(1, hidden_size // num_heads)
        self.input_proj = nn.Linear(n_features, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        encoded = self.encoder(x)  # (batch, seq_len, hidden_size)
        pooled = encoded.mean(dim=1)  # mean-pool over the time dimension
        pooled = self.dropout(pooled)
        return self.head(pooled).squeeze(-1)


def build_model(
    model_type: Literal["lstm", "transformer"],
    n_features: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = False,
    num_heads: int = 4,
    ff_dim: int = 128,
) -> nn.Module:
    """Factory that builds the configured model architecture.

    Kept centralized so the pipeline / persistence code has a single place
    that maps a config to a concrete ``nn.Module`` instance (needed both at
    training time and when reconstructing a model for inference).
    """
    if model_type == "lstm":
        return LSTMRegressor(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
    if model_type == "transformer":
        return TransformerRegressor(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads,
            ff_dim=ff_dim,
        )
    raise ValueError(f"Unknown model_type: {model_type}")
