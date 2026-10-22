"""
FastAPI backend wrapping the ML package.

Run from the repo root:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    GET  /api/models                 trained runs with their asset, flags and test metrics
    GET  /api/market/{symbol}        recent OHLCV bars from data/raw/<symbol>_market.csv
    GET  /api/news/{symbol}          recent headlines from data/raw/<symbol>_news.json
    POST /api/predict                prediction for a run, from stored data or caller-supplied bars/news
    GET  /reports/<run>/<file>.png   training plots
"""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import lru_cache
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import ML_ROOT, PRICE_MODEL_DIR, RAW_DATA_DIR
from src.data.loader import load_market_data_csv, load_market_data_from_records, load_news_data_json
from src.pipeline.inference_pipeline import InferencePipeline

REPORTS_DIR = ML_ROOT / "reports"

app = FastAPI(title="Multi-Factor Price Prediction API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])
REPORTS_DIR.mkdir(exist_ok=True)
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")


class Bar(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class News(BaseModel):
    timestamp: str
    text: str
    source: Optional[str] = "news"


class PredictRequest(BaseModel):
    run: str
    # Optional: omit both to predict from the latest scraped data for the run's asset.
    market_data: Optional[List[Bar]] = None
    news: Optional[List[News]] = None


def _run_dirs():
    return sorted(p for p in PRICE_MODEL_DIR.glob("*") if (p / "model.pt").exists()) if PRICE_MODEL_DIR.exists() else []


@lru_cache(maxsize=None)
def _pipeline(run: str) -> InferencePipeline:
    # Loaded once per run and kept in memory (model, scaler, FinBERT).
    return InferencePipeline(PRICE_MODEL_DIR / run)


def _safe_symbol(symbol: str) -> str:
    if not all(c.isalnum() or c in "-_.^=" for c in symbol):
        raise HTTPException(400, "invalid symbol")
    return symbol


def _market_df(symbol: str) -> pd.DataFrame:
    path = RAW_DATA_DIR / f"{_safe_symbol(symbol)}_market.csv"
    if not path.exists():
        raise HTTPException(404, f"No market data for {symbol}; run scripts/scrape_data.py")
    return load_market_data_csv(path)


def _news_df(symbol: str) -> pd.DataFrame:
    path = RAW_DATA_DIR / f"{_safe_symbol(symbol)}_news.json"
    return load_news_data_json(path) if path.exists() else pd.DataFrame(columns=["timestamp", "text", "source"])


@app.get("/api/models")
def list_models():
    out = []
    for d in _run_dirs():
        config = json.loads((d / "config.json").read_text(encoding="utf-8"))
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        out.append(
            {
                "run": d.name,
                "asset": config["data"]["asset"],
                "model_type": config["model"]["model_type"],
                "use_sentiment": meta.get("use_sentiment", False),
                "sentiment_window": config["sentiment"]["aggregation_window"],
                "sequence_length": config["timeseries"]["sequence_length"],
                "n_train": meta.get("n_train"),
                "n_test": meta.get("n_test"),
                "metrics": meta.get("metrics", {}),
                "plots": [f"/reports/{d.name}/{p.name}" for p in sorted((REPORTS_DIR / d.name).glob("*.png"))],
            }
        )
    return out


@app.get("/api/market/{symbol}")
def market(symbol: str, limit: int = 500):
    df = _market_df(symbol).tail(limit)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df[["timestamp", "open", "high", "low", "close", "volume"]].to_dict(orient="records")


@app.get("/api/news/{symbol}")
def news(symbol: str, limit: int = 30):
    df = _news_df(symbol).sort_values("timestamp").tail(limit).iloc[::-1]
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df[["timestamp", "text", "source"]].to_dict(orient="records")


@app.post("/api/predict")
def predict(req: PredictRequest):
    if req.run not in {d.name for d in _run_dirs()}:
        raise HTTPException(404, f"Unknown run '{req.run}'")
    pipe = _pipeline(req.run)

    if req.market_data is not None:
        market_df = load_market_data_from_records([b.model_dump() for b in req.market_data])
        news_items = [n.model_dump() for n in req.news or []]
    else:
        # Enough history for indicator warm-up (MACD/EMA-26) on top of the sequence window.
        market_df = _market_df(pipe.config.data.asset).tail(pipe.config.timeseries.sequence_length + 100)
        news_df = _news_df(pipe.config.data.asset)
        start = market_df["timestamp"].iloc[0] - pd.Timedelta(pipe.config.sentiment.aggregation_window)
        news_df = news_df[pd.to_datetime(news_df["timestamp"]) > start]
        news_items = news_df.assign(timestamp=news_df["timestamp"].astype(str)).to_dict(orient="records")

    try:
        result = pipe.predict(market_data=market_df, sentiment_data=news_items if pipe.use_sentiment else None)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    last_ts = pd.to_datetime(market_df["timestamp"]).iloc[-1]
    return {"run": req.run, "asset": pipe.config.data.asset, "as_of": last_ts.isoformat(), **asdict(result)}
