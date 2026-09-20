"""
Shared data contracts between the ML layer and any external caller (notably
the future FastAPI backend).

These are plain ``dataclasses`` rather than pydantic models to keep the ML
package dependency-light and framework-agnostic. The backend is free to use
pydantic (or anything else) on its side and adapt to/from these shapes -- the
ML layer does not import or depend on FastAPI/pydantic at all.

Design intent (see README section "Backend Integration Contract"):
    - The backend sends raw market data points and raw news/social items.
    - The ML layer is solely responsible for turning those into model-ready
      features. The backend never needs to know about indicators, sequence
      windows, scalers, or model internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class MarketDataPoint:
    """A single OHLCV observation.

    Attributes:
        timestamp: ISO-8601 string or pandas-parseable timestamp.
        open, high, low, close: price fields.
        volume: traded volume in the observation window.
    """

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class NewsItem:
    """A single news / social-media text item with a publication timestamp.

    Attributes:
        timestamp: ISO-8601 string or pandas-parseable timestamp. Must be the
            time the text was published/observed -- never a time in the
            future relative to the market data it will be aligned with.
        text: raw text content (headline, tweet, article body, etc).
        source: origin of the item, e.g. "news" or "twitter". Optional.
    """

    timestamp: str
    text: str
    source: Optional[str] = "news"


@dataclass
class SentimentResult:
    """Output of the sentiment-scoring module for a single text item."""

    timestamp: str
    sentiment_label: Literal["positive", "neutral", "negative"]
    sentiment_score: float  # signed: positive -> +score, negative -> -score, neutral -> 0


@dataclass
class PredictionResult:
    """Output contract of the inference pipeline.

    ``confidence`` is derived from a documented, non-fabricated methodology
    (see predict.py / inference_pipeline.py docstrings) -- it is NOT a
    calibrated probability unless explicitly stated as such.
    """

    current_price: float
    predicted_price: float
    predicted_return: float
    direction: Literal["UP", "DOWN"]
    confidence: float
    confidence_method: str
    sentiment_score: Optional[float] = None
