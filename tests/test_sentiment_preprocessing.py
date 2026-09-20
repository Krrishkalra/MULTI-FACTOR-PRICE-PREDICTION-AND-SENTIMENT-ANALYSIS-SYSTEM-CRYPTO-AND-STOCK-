"""Tests for src.sentiment.preprocessing text cleaning."""

from __future__ import annotations

import pandas as pd

from src.sentiment.preprocessing import clean_text, preprocess_news_dataframe


def test_clean_text_strips_urls():
    text = "Great news! Check https://example.com/article for details"
    cleaned = clean_text(text)
    assert "https://" not in cleaned
    assert "Great news" in cleaned


def test_clean_text_strips_mentions():
    text = "@TraderJoe says the market will rally"
    cleaned = clean_text(text)
    assert "@TraderJoe" not in cleaned
    assert "rally" in cleaned


def test_clean_text_collapses_whitespace():
    text = "Too    much   \n\n whitespace"
    cleaned = clean_text(text)
    assert "  " not in cleaned


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_preprocess_news_dataframe_drops_short_items():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "text": ["ok", "This is a proper length headline about earnings"],
        }
    )
    result = preprocess_news_dataframe(df, min_length=10)
    assert len(result) == 1
    assert "earnings" in result.iloc[0]["text"]


def test_preprocess_news_dataframe_sorts_by_timestamp():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-02", "2026-01-01"]),
            "text": ["Second headline text here", "First headline text here"],
        }
    )
    result = preprocess_news_dataframe(df)
    assert result["timestamp"].is_monotonic_increasing
