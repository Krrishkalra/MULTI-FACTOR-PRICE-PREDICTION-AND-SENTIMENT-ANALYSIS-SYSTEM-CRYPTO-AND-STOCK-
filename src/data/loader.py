"""
Market-data loading.

Design intent: the rest of the pipeline should never care whether the OHLCV
data originated from a local CSV file or from the future FastAPI backend. All
loaders in this module funnel into the same output contract: a pandas
DataFrame indexed by a parsed, timezone-naive ``timestamp`` column, sorted
ascending, with columns ``open, high, low, close, volume``.

To plug in the backend later, implement a new loader function (e.g.
``load_market_data_from_api``) that returns the same DataFrame shape and
nothing downstream needs to change.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Union

import pandas as pd

from ..schemas import MarketDataPoint, NewsItem

logger = logging.getLogger(__name__)

REQUIRED_MARKET_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def load_market_data_csv(path: Union[str, Path], timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Load OHLCV market data from a local CSV file.

    Args:
        path: path to a CSV file with at least the columns
            timestamp, open, high, low, close, volume.
        timestamp_col: name of the timestamp column in the source file.

    Returns:
        DataFrame with a parsed ``timestamp`` column (renamed if necessary)
        and the OHLCV columns, sorted ascending by timestamp.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Market data CSV not found: {path}")

    df = pd.read_csv(path)
    if timestamp_col != "timestamp" and timestamp_col in df.columns:
        df = df.rename(columns={timestamp_col: "timestamp"})

    missing = [c for c in REQUIRED_MARKET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Market data is missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    n_bad = df["timestamp"].isna().sum()
    if n_bad:
        logger.warning("Dropping %d rows with unparseable timestamps", n_bad)
        df = df.dropna(subset=["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %d market data rows from %s", len(df), path)
    return df[REQUIRED_MARKET_COLUMNS]


def load_market_data_from_records(records: Iterable[MarketDataPoint]) -> pd.DataFrame:
    """Build the standard market DataFrame from a list of ``MarketDataPoint``.

    This is the entry point the future FastAPI backend should use: convert
    whatever it receives into ``MarketDataPoint`` instances (or plain dicts
    with the same keys) and call this function.
    """
    rows = [r.__dict__ if isinstance(r, MarketDataPoint) else dict(r) for r in records]
    df = pd.DataFrame(rows)
    missing = [c for c in REQUIRED_MARKET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Market data records are missing required fields: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df[REQUIRED_MARKET_COLUMNS]


def load_news_data_json(path: Union[str, Path]) -> pd.DataFrame:
    """Load news/social-media text items from a local JSON or JSONL file.

    Expected shape per item: ``{"timestamp": ..., "text": ..., "source": ...}``.
    Accepts either a JSON array or newline-delimited JSON (JSONL).

    Returns:
        DataFrame with columns ``timestamp`` (parsed), ``text``, ``source``,
        sorted ascending by timestamp.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"News data file not found: {path}")

    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return pd.DataFrame(columns=["timestamp", "text", "source"])

    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            data = [data]
    except json.JSONDecodeError:
        # Fall back to JSONL parsing.
        data = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    df = pd.DataFrame(data)
    if "text" not in df.columns or "timestamp" not in df.columns:
        raise ValueError("News data items must contain 'timestamp' and 'text' fields")
    if "source" not in df.columns:
        df["source"] = "news"
    else:
        df["source"] = df["source"].fillna("news")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    n_bad = df["timestamp"].isna().sum()
    if n_bad:
        logger.warning("Dropping %d news items with unparseable timestamps", n_bad)
        df = df.dropna(subset=["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %d news/social items from %s", len(df), path)
    return df[["timestamp", "text", "source"]]


def load_news_data_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load news/social-media text items from a local CSV file.

    Expected columns: timestamp, text, and optionally source/headline.
    Some public datasets (e.g. financial news headline datasets) use
    'headline' instead of 'text' -- this loader normalizes that.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"News data CSV not found: {path}")

    df = pd.read_csv(path)
    if "text" not in df.columns:
        for alt in ("headline", "title", "content"):
            if alt in df.columns:
                df = df.rename(columns={alt: "text"})
                break
    if "text" not in df.columns or "timestamp" not in df.columns:
        raise ValueError("News CSV must contain 'timestamp' and 'text' (or headline/title) columns")
    if "source" not in df.columns:
        df["source"] = "news"
    else:
        df["source"] = df["source"].fillna("news")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "text", "source"]]


def load_news_data_from_records(records: Iterable[NewsItem]) -> pd.DataFrame:
    """Build the standard news DataFrame from a list of ``NewsItem``.

    This is the entry point the future FastAPI backend should use for
    news/social data, analogous to ``load_market_data_from_records``.
    """
    rows = [r.__dict__ if isinstance(r, NewsItem) else dict(r) for r in records]
    df = pd.DataFrame(rows)
    if "text" not in df.columns or "timestamp" not in df.columns:
        raise ValueError("News records must contain 'timestamp' and 'text' fields")
    if "source" not in df.columns:
        df["source"] = "news"
    else:
        df["source"] = df["source"].fillna("news")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "text", "source"]]
