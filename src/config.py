"""
Central configuration for the ML pipeline.

Every tunable parameter referenced elsewhere in the codebase (sequence length,
model hyperparameters, split ratios, aggregation window, sentiment model name,
etc.) should live here rather than being hardcoded in individual modules.

Configuration can be constructed from defaults, from a JSON file, or from
environment variables (for values such as file paths that vary between
machines). No API keys or secrets are stored here.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# ml/ package root (this file lives at ml/src/config.py)
ML_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("ML_DATA_DIR", ML_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

MODELS_DIR = Path(os.environ.get("ML_MODELS_DIR", ML_ROOT / "models"))
SENTIMENT_MODELS_DIR = MODELS_DIR / "sentiment"
PRICE_MODEL_DIR = MODELS_DIR / "price_prediction"

ModelType = Literal["lstm", "transformer"]
TargetType = Literal["price", "return"]


@dataclass
class DataConfig:
    """Configuration for market-data ingestion and feature engineering."""

    asset: str = "SAMPLE-ASSET"
    timestamp_col: str = "timestamp"
    price_cols: tuple = ("open", "high", "low", "close", "volume")

    # Technical feature parameters (kept deliberately small / explainable).
    ma_windows: tuple = (5, 10, 20)
    ema_windows: tuple = (12, 26)
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volatility_window: int = 20

    # Outlier detection: number of standard deviations for the z-score filter
    # applied to returns. Values outside this range are flagged, not silently
    # dropped (see cleaner.py).
    outlier_z_threshold: float = 6.0


@dataclass
class SentimentConfig:
    """Configuration for the sentiment-scoring and aggregation modules."""

    # Model is swappable: "finbert" (default, finance-tuned) or "generic"
    # (a general-purpose sentiment model). See README for the trade-off
    # discussion between the two.
    model_name: str = "ProsusAI/finbert"
    model_family: Literal["finbert", "generic"] = "finbert"
    generic_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"

    max_length: int = 256
    batch_size: int = 16
    device: str = "cpu"  # "cuda" if available; kept explicit, not auto-guessed

    # Aggregation window, expressed as a pandas offset alias, e.g. "1h", "30min".
    aggregation_window: str = "1h"

    # If no news/social items fall in a window, this determines how the
    # aggregated sentiment features are filled. "zero" fills numeric
    # sentiment features with 0 (neutral) and news_count with 0. "nan" leaves
    # them as NaN so downstream code can decide how to impute/mask.
    empty_window_strategy: Literal["zero", "nan"] = "zero"


@dataclass
class TimeSeriesConfig:
    """Configuration for sequence construction and train/val/test split."""

    sequence_length: int = 60
    prediction_horizon: int = 1
    target_type: TargetType = "return"

    train_split: float = 0.70
    validation_split: float = 0.15
    # test_split is implied: 1 - train_split - validation_split

    batch_size: int = 32
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Configuration for the deep-learning model architecture and training."""

    model_type: ModelType = "lstm"

    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False

    # Transformer-specific (used only when model_type == "transformer")
    num_heads: int = 4
    ff_dim: int = 128

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    early_stopping_patience: int = 10
    grad_clip_norm: float = 1.0


@dataclass
class Config:
    """Top-level configuration bundling all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    timeseries: TimeSeriesConfig = field(default_factory=TimeSeriesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    random_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(
            data=DataConfig(**raw.get("data", {})),
            sentiment=SentimentConfig(**raw.get("sentiment", {})),
            timeseries=TimeSeriesConfig(**raw.get("timeseries", {})),
            model=ModelConfig(**raw.get("model", {})),
            random_seed=raw.get("random_seed", 42),
        )


def set_global_seed(seed: int) -> None:
    """Set random seeds across python/numpy/torch for reproducibility.

    Torch is imported lazily so this module has no hard dependency on it.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_default_config() -> Config:
    """Return a fresh default configuration instance."""
    return Config()
