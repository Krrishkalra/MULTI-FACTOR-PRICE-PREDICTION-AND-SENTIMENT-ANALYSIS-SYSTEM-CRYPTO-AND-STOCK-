"""
Scrape a real dataset for one asset into ``data/raw/``:

    - Hourly OHLCV bars from Yahoo Finance's public chart endpoint (max ~730 days at 1h).
    - Timestamped headlines from Google News RSS, queried week by week so the
      history covers the same span as the price data.

Usage:
    python scripts/scrape_data.py --symbol BTC-USD --query bitcoin
    python scripts/scrape_data.py --symbol AAPL --query "Apple stock"

Writes ``data/raw/<symbol>_market.csv`` and ``data/raw/<symbol>_news.json`` in the
formats documented in README section 4. All timestamps are naive UTC.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def fetch_market(symbol: str, days: int) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?interval=1h&range={days}d"
    result = json.loads(_get(url))["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(result["timestamp"], unit="s"),
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
        }
    )
    # Yahoo returns null rows for halted/illiquid bars; the last bar is still forming.
    return df.dropna().iloc[:-1].reset_index(drop=True)


def fetch_binance_market(pair: str, days: int) -> pd.DataFrame:
    """Hourly klines from Binance's public API. Used for crypto because Yahoo's hourly
    crypto bars report zero volume for roughly half the hours."""
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - days * 86_400_000
    rows: list = []
    while start_ms < end_ms:
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={pair}&interval=1h&startTime={start_ms}&limit=1000"
        batch = json.loads(_get(url))
        if not batch:
            break
        rows += batch
        start_ms = batch[-1][0] + 3_600_000
        time.sleep(0.2)
    df = pd.DataFrame([r[:6] for r in rows], columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    # The last kline is still forming.
    return df.drop_duplicates("timestamp").iloc[:-1].reset_index(drop=True)


def _headline_timestamp(pub: datetime) -> datetime:
    pub = pub.astimezone(timezone.utc).replace(tzinfo=None)
    # Google uses 08:00 GMT as a placeholder for date-only items; make those available from
    # the next midnight so a headline can never be aligned to a bar before it was published.
    if (pub.hour, pub.minute, pub.second) == (8, 0, 0):
        return pub.replace(hour=0) + timedelta(days=1)
    return pub


def fetch_news(query: str, start: datetime, end: datetime, sleep: float = 1.0) -> list[dict]:
    items: dict[str, dict] = {}
    week_start = start
    while week_start < end:
        week_end = week_start + timedelta(days=7)
        q = f"{query} after:{week_start:%Y-%m-%d} before:{week_end:%Y-%m-%d}"
        url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
            {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}
        )
        try:
            root = ET.fromstring(_get(url))
        except Exception as exc:  # one failed week shouldn't lose the whole scrape
            print(f"  skip {week_start:%Y-%m-%d}: {exc}", file=sys.stderr)
            root = None
        for item in root.iter("item") if root is not None else []:
            title = (item.findtext("title") or "").strip()
            publisher = item.findtext("source") or ""
            if publisher and title.endswith(f" - {publisher}"):
                title = title[: -len(publisher) - 3]
            if not title or title in items:
                continue
            ts = _headline_timestamp(parsedate_to_datetime(item.findtext("pubDate")))
            items[title] = {"timestamp": ts.isoformat(), "text": title, "source": "google_news"}
        print(f"  {week_start:%Y-%m-%d}: {len(items)} headlines so far")
        week_start = week_end
        time.sleep(sleep)
    return sorted(items.values(), key=lambda x: x["timestamp"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape real OHLCV + news data for one asset.")
    parser.add_argument("--symbol", required=True, help="Yahoo Finance symbol, e.g. BTC-USD, AAPL")
    parser.add_argument("--query", required=True, help="Google News search query, e.g. bitcoin")
    parser.add_argument("--days", type=int, default=729)
    parser.add_argument("--binance-pair", help="Take bars from Binance instead of Yahoo, e.g. BTCUSDT (recommended for crypto)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {args.symbol} hourly bars...")
    market = fetch_binance_market(args.binance_pair, args.days) if args.binance_pair else fetch_market(args.symbol, args.days)
    market_path = RAW_DIR / f"{args.symbol}_market.csv"
    market.to_csv(market_path, index=False)
    print(f"Wrote {len(market)} bars to {market_path}")

    print(f"Fetching news for '{args.query}'...")
    news = fetch_news(args.query, market["timestamp"].min().to_pydatetime(), market["timestamp"].max().to_pydatetime())
    news_path = RAW_DIR / f"{args.symbol}_news.json"
    news_path.write_text(json.dumps(news, indent=2), encoding="utf-8")
    print(f"Wrote {len(news)} headlines to {news_path}")


if __name__ == "__main__":
    main()
