"""
Text preprocessing for news/social-media items before sentiment scoring.

Kept intentionally light: transformer-based sentiment models are trained on
fairly natural text, so aggressive cleaning (stopword removal, stemming)
would actually hurt performance. We only strip structural noise that adds no
sentiment signal (URLs, excess whitespace, HTML entities) and enforce basic
validity checks.
"""

from __future__ import annotations

import html
import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize a single text item for sentiment scoring.

    Steps: HTML-unescape, strip URLs, strip @mentions (common in social
    posts and carry no sentiment content), collapse whitespace, strip.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def preprocess_news_dataframe(df: pd.DataFrame, text_col: str = "text", min_length: int = 3) -> pd.DataFrame:
    """Clean the ``text_col`` of a news/social DataFrame and drop empties.

    Args:
        df: DataFrame with a text column and a ``timestamp`` column.
        text_col: name of the column containing raw text.
        min_length: minimum character length (after cleaning) to keep a row.
            Extremely short items (e.g. a single emoji) carry little
            sentiment signal and can destabilize scoring.

    Returns:
        A new DataFrame with the text column cleaned in place and invalid
        rows removed, sorted by timestamp.
    """
    df = df.copy()
    df[text_col] = df[text_col].apply(clean_text)
    before = len(df)
    df = df[df[text_col].str.len() >= min_length].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d news/social items shorter than %d chars after cleaning", dropped, min_length)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df
