"""Tests for src.data.loader: CSV/JSON loading and timestamp parsing."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.loader import (
    load_market_data_csv,
    load_market_data_from_records,
    load_news_data_json,
)
from src.schemas import MarketDataPoint


def test_load_market_data_csv(tmp_path):
    csv_path = tmp_path / "market.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-01-01 00:00:00,100,103,99,102,150000\n"
        "2026-01-01 01:00:00,102,105,101,104,180000\n"
    )
    df = load_market_data_csv(csv_path)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    # sorted ascending
    assert df["timestamp"].is_monotonic_increasing


def test_load_market_data_csv_missing_columns_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("timestamp,open,close\n2026-01-01,1,2\n")
    with pytest.raises(ValueError):
        load_market_data_csv(csv_path)


def test_load_market_data_csv_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_market_data_csv(tmp_path / "does_not_exist.csv")


def test_load_market_data_csv_drops_unparseable_timestamps(tmp_path):
    csv_path = tmp_path / "market.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "not_a_date,100,103,99,102,150000\n"
        "2026-01-01 01:00:00,102,105,101,104,180000\n"
    )
    df = load_market_data_csv(csv_path)
    assert len(df) == 1


def test_load_market_data_from_records():
    records = [
        MarketDataPoint(timestamp="2026-01-01T01:00:00", open=1, high=2, low=0.5, close=1.5, volume=100),
        MarketDataPoint(timestamp="2026-01-01T00:00:00", open=1, high=2, low=0.5, close=1.2, volume=90),
    ]
    df = load_market_data_from_records(records)
    assert len(df) == 2
    # sorted ascending despite input order
    assert df["timestamp"].is_monotonic_increasing


def test_load_news_data_json_array(tmp_path):
    news_path = tmp_path / "news.json"
    items = [
        {"timestamp": "2026-01-01T10:00:00", "text": "Great earnings report"},
        {"timestamp": "2026-01-01T09:00:00", "text": "Market awaits decision", "source": "twitter"},
    ]
    news_path.write_text(json.dumps(items))
    df = load_news_data_json(news_path)
    assert len(df) == 2
    assert df["timestamp"].is_monotonic_increasing
    assert set(df["source"]) == {"news", "twitter"}


def test_load_news_data_json_requires_text_and_timestamp(tmp_path):
    news_path = tmp_path / "news.json"
    news_path.write_text(json.dumps([{"foo": "bar"}]))
    with pytest.raises(ValueError):
        load_news_data_json(news_path)
