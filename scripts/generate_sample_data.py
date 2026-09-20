"""
Generate synthetic local sample datasets so the ML pipeline can be developed
and tested end-to-end without waiting on the FastAPI backend or on external
dataset downloads.

Produces:
    ml/data/sample/market_data.csv   - hourly OHLCV data for a synthetic asset
    ml/data/sample/news_data.json    - news/social items with timestamps and
                                        text whose sentiment is intentionally
                                        correlated (with noise) to the
                                        next-period return, so that the
                                        price-only vs price+sentiment
                                        experiment has a realistic chance of
                                        showing a measurable difference.

This is NOT meant to resemble real market dynamics -- it exists purely to
exercise the pipeline. Real experiments should replace this with the
Kaggle/HuggingFace datasets referenced in the README.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 42
N_PERIODS = 2000  # hourly bars => ~83 days
START = pd.Timestamp("2026-01-01 00:00:00")

POSITIVE_HEADLINES = [
    "Company reports stronger than expected quarterly earnings",
    "Analysts upgrade price target citing robust demand",
    "Partnership announcement boosts investor confidence",
    "Network upgrade successfully completed, adoption rises",
    "Institutional inflows hit new high this quarter",
    "Regulatory clarity welcomed by market participants",
]
NEGATIVE_HEADLINES = [
    "Company misses revenue expectations amid weak demand",
    "Regulators announce fresh scrutiny over trading practices",
    "Major exchange reports security incident, funds frozen",
    "Analysts downgrade outlook citing macro headwinds",
    "Sell-off intensifies as risk appetite fades",
    "Legal probe launched into reporting practices",
]
NEUTRAL_HEADLINES = [
    "Daily trading volume remains in line with recent averages",
    "Market awaits central bank decision later this week",
    "Analysts maintain hold rating ahead of earnings",
    "Weekly summary: prices trade in a tight range",
    "Company to host investor call next Tuesday",
]


def generate_market_data(rng: np.random.Generator) -> pd.DataFrame:
    """Generate a synthetic hourly OHLCV series with a mild mean-reverting
    drift plus injected 'sentiment events' that nudge subsequent returns.
    """
    timestamps = pd.date_range(START, periods=N_PERIODS, freq="1h")

    price = 100.0
    closes = []
    event_returns = np.zeros(N_PERIODS)  # filled in alongside news generation

    # Pre-decide event periods and their directional bias; used both for
    # price generation and for generating correlated news text below.
    event_periods = sorted(rng.choice(N_PERIODS, size=N_PERIODS // 8, replace=False))
    event_bias = {}
    for idx in event_periods:
        bias = rng.choice([1, -1])
        event_bias[idx] = bias
        # Small persistent nudge over the following few periods.
        for k in range(3):
            if idx + k < N_PERIODS:
                event_returns[idx + k] += bias * 0.004 * (0.6 ** k)

    for i in range(N_PERIODS):
        noise = rng.normal(0, 0.003)
        ret = noise + event_returns[i]
        price = max(price * (1 + ret), 0.01)
        closes.append(price)

    closes = np.array(closes)
    opens = np.roll(closes, 1)
    opens[0] = closes[0] * (1 + rng.normal(0, 0.001))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.002, N_PERIODS)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.002, N_PERIODS)))
    volumes = rng.integers(50_000, 250_000, size=N_PERIODS).astype(float)
    # Volume spikes around events.
    for idx in event_periods:
        volumes[idx] *= rng.uniform(1.5, 2.5)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": np.round(opens, 4),
            "high": np.round(highs, 4),
            "low": np.round(lows, 4),
            "close": np.round(closes, 4),
            "volume": np.round(volumes, 0),
        }
    )
    return df, event_periods, event_bias, timestamps


def generate_news_data(rng: np.random.Generator, event_periods, event_bias, timestamps) -> list[dict]:
    """Generate news items concentrated around event periods, with sentiment
    text direction matching ``event_bias`` (with some label noise), plus a
    sprinkling of unrelated neutral chatter throughout the series.
    """
    items = []

    for idx in event_periods:
        ts = timestamps[idx] - pd.Timedelta(minutes=int(rng.integers(1, 45)))
        bias = event_bias[idx]
        # 85% of the time the headline matches the true bias; 15% noise.
        if rng.random() < 0.85:
            pool = POSITIVE_HEADLINES if bias > 0 else NEGATIVE_HEADLINES
        else:
            pool = NEUTRAL_HEADLINES
        text = rng.choice(pool)
        items.append({"timestamp": ts.isoformat(), "text": str(text), "source": "news"})

        # Occasionally a follow-up social post shortly after.
        if rng.random() < 0.5:
            ts2 = timestamps[idx] + pd.Timedelta(minutes=int(rng.integers(1, 30)))
            pool2 = POSITIVE_HEADLINES if bias > 0 else NEGATIVE_HEADLINES
            items.append({"timestamp": ts2.isoformat(), "text": str(rng.choice(pool2)), "source": "twitter"})

    # Background neutral chatter, sparsely distributed.
    n_background = N_PERIODS // 20
    background_idxs = rng.choice(N_PERIODS, size=n_background, replace=False)
    for idx in background_idxs:
        ts = timestamps[idx] + pd.Timedelta(minutes=int(rng.integers(0, 59)))
        items.append({"timestamp": ts.isoformat(), "text": str(rng.choice(NEUTRAL_HEADLINES)), "source": "news"})

    items.sort(key=lambda x: x["timestamp"])
    return items


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    random.seed(RNG_SEED)

    market_df, event_periods, event_bias, timestamps = generate_market_data(rng)
    news_items = generate_news_data(rng, event_periods, event_bias, timestamps)

    market_path = SAMPLE_DIR / "market_data.csv"
    news_path = SAMPLE_DIR / "news_data.json"

    market_df.to_csv(market_path, index=False)
    with open(news_path, "w", encoding="utf-8") as f:
        json.dump(news_items, f, indent=2)

    print(f"Wrote {len(market_df)} market rows to {market_path}")
    print(f"Wrote {len(news_items)} news items to {news_path}")


if __name__ == "__main__":
    main()
