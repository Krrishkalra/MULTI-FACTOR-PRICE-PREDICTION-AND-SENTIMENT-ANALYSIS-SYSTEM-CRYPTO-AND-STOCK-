"""Tests for src.sentiment.sentiment_model.

The label->signed-score mapping is tested directly (fast, no model needed).
An integration test that actually loads FinBERT and scores real text is
included but marked so it can be skipped in environments without network
access / the model cached locally.
"""

from __future__ import annotations

import os

import pytest

from src.config import SentimentConfig
from src.sentiment.sentiment_model import SentimentScorer


def test_label_to_signed_score_positive():
    scorer = SentimentScorer(SentimentConfig())
    label, score = scorer._label_to_signed_score("positive", 0.9)
    assert label == "positive"
    assert score == pytest.approx(0.9)


def test_label_to_signed_score_negative():
    scorer = SentimentScorer(SentimentConfig())
    label, score = scorer._label_to_signed_score("negative", 0.7)
    assert label == "negative"
    assert score == pytest.approx(-0.7)


def test_label_to_signed_score_neutral():
    scorer = SentimentScorer(SentimentConfig())
    label, score = scorer._label_to_signed_score("neutral", 0.99)
    assert label == "neutral"
    assert score == 0.0


def test_score_batch_empty_returns_empty_list():
    scorer = SentimentScorer(SentimentConfig())
    assert scorer.score_batch([]) == []


@pytest.mark.skipif(
    os.environ.get("SKIP_MODEL_DOWNLOAD_TESTS") == "1",
    reason="Requires downloading/loading the FinBERT model; set SKIP_MODEL_DOWNLOAD_TESTS=1 to skip",
)
def test_finbert_scores_clearly_positive_and_negative_text():
    scorer = SentimentScorer(SentimentConfig())
    results = scorer.score_batch(
        [
            "Company reports record profits and raises full-year guidance",
            "Company files for bankruptcy after massive fraud investigation",
        ]
    )
    assert results[0]["sentiment_label"] == "positive"
    assert results[0]["sentiment_score"] > 0
    assert results[1]["sentiment_label"] == "negative"
    assert results[1]["sentiment_score"] < 0
