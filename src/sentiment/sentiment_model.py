"""
Sentiment scoring using a pretrained Hugging Face transformer model.

Model choice (FinBERT vs generic) -- see README "Sentiment Model" section for
the full discussion. Summary of the trade-off:

    Generic model (e.g. distilbert-base-uncased-finetuned-sst-2-english):
        + Fast, small, widely used, good on everyday English.
        - Not tuned for financial vocabulary. Words like "liability",
          "charge", "burn", or "beat estimates" carry different connotations
          in financial text than in everyday speech, so a generic model
          frequently mislabels routine financial language as negative.

    FinBERT (ProsusAI/finbert):
        + Fine-tuned on financial text (10-Ks, earnings call transcripts,
          analyst reports), correctly handles finance-specific vocabulary.
        + 3-class output (positive/negative/neutral) which maps directly to
          the sentiment_label contract this module needs.
        - Not flawless: independent evaluations show it can over-predict
          "positive" on headlines that are just neutral corporate
          announcements without real price-moving information. It should be
          treated as a strong domain-specific baseline, not a ground truth.

Default: FinBERT. The model is swappable via ``SentimentConfig.model_family``
("finbert" or "generic") so this decision is not hardcoded into the pipeline.

This module returns a *signed* sentiment score in [-1, 1]: positive labels
carry +confidence, negative labels carry -confidence, neutral carries 0. This
gives the downstream ML pipeline a single numeric feature per text item
instead of forcing it to consume a categorical label directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from ..config import SentimentConfig
from ..schemas import SentimentResult

logger = logging.getLogger(__name__)

# transformers is imported lazily inside the class so that modules which do
# not need sentiment scoring (e.g. pure market-data unit tests) do not pay
# the import cost / require the dependency to be installed.


@dataclass
class _LoadedModel:
    tokenizer: object
    model: object
    id2label: dict


class SentimentScorer:
    """Wraps a Hugging Face sequence-classification model for sentiment scoring.

    The model is loaded once (lazily, on first use) and reused for all
    subsequent calls. Batches of text are scored together for efficiency.
    """

    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()
        self._loaded: Optional[_LoadedModel] = None

    @property
    def model_name(self) -> str:
        if self.config.model_family == "finbert":
            return self.config.model_name
        return self.config.generic_model_name

    def _ensure_loaded(self) -> _LoadedModel:
        if self._loaded is not None:
            return self._loaded

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        name = self.model_name
        logger.info("Loading sentiment model '%s' (family=%s)", name, self.config.model_family)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name)
        model.to(self.config.device)
        model.eval()

        id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
        self._loaded = _LoadedModel(tokenizer=tokenizer, model=model, id2label=id2label)
        return self._loaded

    def _label_to_signed_score(self, label: str, prob: float) -> tuple:
        """Map a raw (label, probability) pair to (normalized_label, signed_score)."""
        label = label.lower()
        if "pos" in label:
            return "positive", float(prob)
        if "neg" in label:
            return "negative", float(-prob)
        return "neutral", 0.0

    def score_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Score a batch of raw text strings (no timestamps attached).

        Returns:
            List of dicts with keys ``sentiment_label`` and ``sentiment_score``,
            one per input text, in order.
        """
        if not texts:
            return []

        import torch

        loaded = self._ensure_loaded()
        results = []
        batch_size = self.config.batch_size

        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            inputs = loaded.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.config.device)

            with torch.no_grad():
                logits = loaded.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)

            top_probs, top_idx = probs.max(dim=-1)
            for prob, idx in zip(top_probs.tolist(), top_idx.tolist()):
                raw_label = loaded.id2label[idx]
                label, score = self._label_to_signed_score(raw_label, prob)
                results.append({"sentiment_label": label, "sentiment_score": score})

        return results

    def score_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """Score every row of a news/social DataFrame.

        Args:
            df: DataFrame containing at least a ``timestamp`` and ``text_col``
                column (already cleaned via sentiment.preprocessing).
            text_col: name of the text column to score.

        Returns:
            A new DataFrame with ``sentiment_label`` and ``sentiment_score``
            columns appended, preserving the original row order.
        """
        if df.empty:
            out = df.copy()
            out["sentiment_label"] = pd.Series(dtype="object")
            out["sentiment_score"] = pd.Series(dtype="float64")
            return out

        texts = df[text_col].astype(str).tolist()
        scored = self.score_batch(texts)
        out = df.copy().reset_index(drop=True)
        out["sentiment_label"] = [s["sentiment_label"] for s in scored]
        out["sentiment_score"] = [s["sentiment_score"] for s in scored]
        return out
