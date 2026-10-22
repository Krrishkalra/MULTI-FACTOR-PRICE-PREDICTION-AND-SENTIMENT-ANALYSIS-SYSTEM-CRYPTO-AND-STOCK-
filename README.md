# Multi-Factor Price Prediction and Sentiment Analysis System — ML Component

This is the standalone Machine Learning / data-science component of the Multi-Factor Price
Prediction and Sentiment Analysis System. It combines historical OHLCV market data with
news/social-media sentiment to predict short-horizon price movement, using a modular
pipeline that a separate FastAPI backend can call without knowing any internal details.

The system runs entirely on local CSV/JSON sample data today, so ML development is fully
decoupled from backend development.

## 1. Objective

Given historical price data and (optionally) news/social text, the pipeline:

1. Cleans and preprocesses market data
2. Scores sentiment with a pretrained NLP model (FinBERT by default)
3. Aggregates sentiment into time windows aligned with market bars
4. Engineers a small, explainable set of technical features
5. Builds sequences for an LSTM/Transformer model
6. Trains and compares against two baselines (naive, Random Forest)
7. Evaluates with regression + directional metrics
8. Serves predictions through a clean inference interface with a documented
   (non-fabricated) confidence estimate

## 2. Architecture

```
ml/
├── data/
│   ├── raw/            # place real downloaded datasets here
│   ├── processed/      # intermediate cached datasets (gitignored in practice)
│   └── sample/         # synthetic sample data checked into the repo
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sentiment_exploration.ipynb
│   └── 03_model_experiments.ipynb
├── src/
│   ├── config.py               # all tunables live here
│   ├── schemas.py               # backend integration contract (dataclasses)
│   ├── data/
│   │   ├── loader.py             # CSV/JSON/records -> standard DataFrame
│   │   ├── cleaner.py            # dedup, invalid-value, missing-value, outliers
│   │   ├── feature_engineering.py # returns, MA, EMA, RSI, MACD, volatility
│   │   └── alignment.py          # merge market+sentiment, target creation
│   ├── sentiment/
│   │   ├── sentiment_model.py    # FinBERT / generic HF sentiment scorer
│   │   ├── preprocessing.py      # text cleaning
│   │   └── aggregator.py         # per-window sentiment aggregation
│   ├── timeseries/
│   │   ├── dataset.py            # chronological split, scaling, sequences
│   │   ├── model.py              # LSTMRegressor, TransformerRegressor
│   │   ├── train.py              # training loop, early stopping
│   │   ├── predict.py            # point + MC-dropout uncertainty prediction
│   │   ├── baselines.py          # naive + RandomForest baselines
│   │   └── persistence.py        # save/load model bundle
│   ├── evaluation/
│   │   ├── metrics.py            # MAE/RMSE/MAPE/R2/direction/F1
│   │   └── plots.py              # loss curves, actual-vs-predicted, errors
│   └── pipeline/
│       ├── training_pipeline.py  # end-to-end training entry point
│       └── inference_pipeline.py # end-to-end inference entry point
├── models/
│   ├── sentiment/                 # (HF sentiment models are cached by transformers, not stored here)
│   └── price_prediction/
│       ├── experiment_a_price_only/
│       └── experiment_b_price_sentiment/
├── reports/
│   ├── experiment_a_price_only/   # metrics.json + plots
│   └── experiment_b_price_sentiment/
├── scripts/
│   └── generate_sample_data.py    # synthetic OHLCV + news generator
├── tests/
├── requirements.txt
└── main.py
```

## 3. Installation

```powershell
cd ml
pip install -r requirements.txt
```

Generate local sample data (already included in the repo, but regenerable):

```powershell
python scripts/generate_sample_data.py
```

This writes `data/sample/market_data.csv` (2000 synthetic hourly OHLCV bars) and
`data/sample/news_data.json` (news/social items whose sentiment is intentionally
correlated with future returns, purely so the pipeline has something meaningful to learn
from before real datasets are wired in — see section 12).

## 4. Dataset Format

**Market data (CSV or records):**

```
timestamp,open,high,low,close,volume
2026-01-01 09:15:00,100,103,99,102,150000
2026-01-01 09:30:00,102,105,101,104,180000
```

**News/social data (JSON array or JSONL):**

```json
{"timestamp": "2026-01-01T10:30:00", "text": "Company reports stronger than expected quarterly earnings", "source": "news"}
```

Both formats mirror the schemas the future FastAPI backend is expected to provide (see
section 11).

## 5. Training

```powershell
# Experiment A: price/technical features only
python -m src.pipeline.training_pipeline --run-name experiment_a_price_only

# Experiment B: price/technical features + sentiment features
python -m src.pipeline.training_pipeline --run-name experiment_b_price_sentiment --use-sentiment
```

Useful flags: `--model-type {lstm,transformer}`, `--epochs N`, `--market-csv PATH`,
`--news-json PATH`. All other hyperparameters (sequence length, hidden size, learning
rate, split ratios, aggregation window, ...) come from `src/config.py` — edit
`get_default_config()` or construct a custom `Config` and call
`run_training_pipeline(...)` directly for full control.

Each run produces:
- `models/price_prediction/<run_name>/` — model.pt, scaler.pkl, config.json,
  feature_list.json, metadata.json
- `reports/<run_name>/` — metrics.json, loss_curve.png, actual_vs_predicted.png,
  prediction_errors.png

## 6. Inference

```powershell
python -m src.pipeline.inference_pipeline --model-dir models/price_prediction/experiment_a_price_only
```

Programmatically:

```python
from src.pipeline.inference_pipeline import InferencePipeline

pipeline = InferencePipeline("models/price_prediction/experiment_a_price_only")
result = pipeline.predict(market_data=market_df)  # or a list[MarketDataPoint]
print(result)
# PredictionResult(current_price=..., predicted_price=..., predicted_return=...,
#                   direction='UP'|'DOWN', confidence=..., confidence_method=..., sentiment_score=...)
```

If the loaded model was trained with `--use-sentiment`, pass `sentiment_data=` (a list of
`NewsItem` or a DataFrame) as well; the pipeline raises `ValueError` if it's required but
missing.

## 7. Model Architecture

- **LSTM** (`LSTMRegressor`): multi-layer `nn.LSTM` → dropout → last time-step hidden
  state → linear projection to a scalar. Configurable hidden size, layers, dropout,
  bidirectionality.
- **Transformer** (`TransformerRegressor`, experimental/optional): linear input
  projection → sinusoidal positional encoding → `nn.TransformerEncoder` stack →
  mean-pool over time → linear head. Selected via `model_type = "transformer"`. It is
  **not** assumed to beat the LSTM — compare both empirically on your data.
- Both share the same `(batch, seq_len, n_features) -> (batch,)` contract so the training
  loop, evaluation, and persistence code are architecture-agnostic.

Training uses early stopping on validation loss, gradient clipping, Adam with weight
decay, and restores the best-validation-epoch weights before saving.

## 8. Sentiment Model: FinBERT vs. Generic

Two options are wired up via `SentimentConfig.model_family`:

| | Generic (`distilbert-base-uncased-finetuned-sst-2-english`) | FinBERT (`ProsusAI/finbert`, default) |
|---|---|---|
| Strengths | Fast, small, strong on everyday English | Fine-tuned on financial text (10-Ks, earnings calls, analyst reports); understands finance-specific vocabulary |
| Weaknesses | Misreads finance vocabulary — words like "liability", "charge", or "burn" read as negative in everyday English but are often routine in financial reporting ([source](https://tommijohnsen.substack.com/p/the-market-is-efficient-but-read)) | Not flawless either: independent evaluation found FinBERT over-predicts "positive" on headlines that are just neutral corporate announcements without real price-moving content — in one review, only 45 of 268 headlines FinBERT labeled positive were genuine positive catalysts ([source](https://tommijohnsen.substack.com/i/187663786/what-the-best-model-still-gets-wrong)) |
| Research context | — | Used as a standard financial-NLP baseline in academic work, e.g. FinBERT + LSTM for market trend forecasting ([arXiv:2306.02136](https://arxiv.org/html/2306.02136v2)) |

**Decision**: FinBERT is the default because it handles domain vocabulary correctly, which
matters more than raw accuracy on generic sentiment benchmarks for this use case. It should
still be treated as a strong domain-specific heuristic, not ground truth — the model is
fully swappable via config (`SentimentConfig(model_family="generic")`) if you want to
compare or replace it with something newer (e.g. an LLM-based scorer).

Output contract: each item gets a label (`positive`/`neutral`/`negative`) and a **signed**
numeric score in `[-1, 1]` (positive → `+confidence`, negative → `-confidence`, neutral →
`0`), so it feeds directly into the ML pipeline as a numeric feature rather than a
category.

## 9. Feature Engineering

Kept deliberately small and explainable (see `src/data/feature_engineering.py`):

- `return`, `log_return`
- Simple moving averages: `ma_5`, `ma_10`, `ma_20`
- Exponential moving averages: `ema_12`, `ema_26`
- Rolling volatility: `volatility_20` (std of returns)
- RSI (Wilder's smoothing): `rsi_14`
- MACD: `macd`, `macd_signal`, `macd_hist`
- `volume_change`

Sentiment aggregation features (when enabled): `mean_sentiment`, `weighted_sentiment`,
`positive_count`, `negative_count`, `neutral_count`, `news_count`, `sentiment_std`.

## 10. Data Leakage Prevention

This was treated as a first-class requirement throughout, not an afterthought:

- **Rolling/EMA features** (`feature_engineering.py`) use `pandas.rolling`/`.ewm`, which
  are backward-looking by construction (`center=False` always) — a value at row `t` never
  uses rows after `t`. Verified explicitly in `tests/test_feature_engineering.py` by
  perturbing future rows and asserting past feature values don't change.
- **Sentiment aggregation** (`sentiment/aggregator.py`) for the window ending at market
  bar `t` only includes news/social items with `timestamp` in `(t - window, t]` — never
  items published after `t`. Verified in `tests/test_sentiment_aggregator.py`.
- **Target creation** (`data/alignment.add_target`) is the *only* place a future value is
  intentionally attached to a row, via `.shift(-horizon)`, isolating the one sanctioned
  "look into the future" operation so it's easy to audit. Rows without a valid future
  value (the trailing `horizon` rows) are dropped.
- **No shuffling before the train/val/test split.** `timeseries/dataset.chronological_split`
  splits purely by position: earliest rows → train, then val, then test.
- **Scalers are fit on the training split only.** `dataset.fit_scaler` is only ever called
  on `train_df`; validation/test data goes through `.transform()`, never `.fit_transform()`.
  Verified in `tests/test_dataset.py`.
- **Sequence construction across split boundaries**: the first `sequence_length - 1` rows
  of the validation/test splits can't form a full window from within-split data alone.
  Rather than silently reaching across the boundary or dropping those rows, we explicitly
  carry over the trailing raw (already-scaled-with-the-training-scaler) rows of the
  *previous* split as left-context (`dataset.get_carry_over`). This reuses feature values
  whose scaling came from train, and never reuses or leaks target/label information across
  the boundary.
- **Missing-value handling** only supports forward-fill or drop — back-fill is intentionally
  not offered, since it would pull a future value into a past gap.
- **Confidence estimates are not fabricated.** See section 13.

## 11. Backend Integration Contract

The ML layer has **no dependency on FastAPI or any backend code**. The contract lives in
`src/schemas.py` as plain dataclasses:

- `MarketDataPoint(timestamp, open, high, low, close, volume)`
- `NewsItem(timestamp, text, source)`
- `PredictionResult(current_price, predicted_price, predicted_return, direction, confidence, confidence_method, sentiment_score)`

The backend constructs `MarketDataPoint`/`NewsItem` instances (or plain dicts with the same
keys — both are accepted) from whatever storage/API it wraps, and calls
`InferencePipeline.predict(market_data=..., sentiment_data=...)`. It receives a
`PredictionResult` back and never needs to know about scalers, sequence windows, or model
internals. When the backend is ready, wire `InferencePipeline` in as an app-level singleton
so the model/scaler load once, not per-request.

## 12. Datasets

The task referenced six external datasets. Availability was checked and the plan for each
is:

| # | Dataset | Status / Plan |
|---|---|---|
| 1 | [Sentiment Analysis for Financial News (Kaggle)](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) | Public, labeled financial headlines (positive/neutral/negative) — good for evaluating/fine-tuning the sentiment scorer against a labeled ground truth. Requires a Kaggle account/API token to download; not fetched automatically by this pipeline. |
| 2 | [FNSPID (Hugging Face)](https://huggingface.co/datasets/Zihan1004/FNSPID) | Large-scale financial news + stock price dataset, useful as a real drop-in for `data/raw/` once downloaded — matches this pipeline's need for timestamped news + price series. |
| 3 | [Daily Financial News for 6000 Stocks (Kaggle)](https://www.kaggle.com/datasets/miguelaenlle/daily-financial-news-for-6000-stocks) | Public headline+timestamp dataset; loadable via `data.loader.load_news_data_csv` (it auto-normalizes `headline`/`title` columns to `text`). |
| 4 | [Twitter Financial News Sentiment (Kaggle)](https://www.kaggle.com/datasets/borhanitrash/twitter-financial-news-sentiment-dataset) | Labeled tweets; same normalization path as #3, `source="twitter"`. |
| 5 | [Stock Market Sentiment Dataset (Kaggle)](https://www.kaggle.com/datasets/yash612/stockmarket-sentiment-dataset) | Social-media sentiment-labeled dataset; same loader path. |
| 6 | [ProsusAI/FinBERT (Hugging Face)](https://huggingface.co/ProsusAI/finbert) | **Used directly** as the default sentiment model (`sentiment_model.py`), downloaded automatically via `transformers` on first use and cached locally (~440MB). Verified working end-to-end in this repo. |

All six links were reachable at the time of writing; none required a substitute. Kaggle
datasets (#1, #3, #4, #5) require a Kaggle account to download programmatically (`kaggle`
CLI + API token) — this wasn't done in this session since it would mean downloading
third-party data into the workspace without your explicit go-ahead. To use them: download
the CSV, drop it in `data/raw/`, and point `--news-json`/`--market-csv` (or a small loader
adapter, since #1/#4/#5 use different column names) at it. The loaders in
`data/loader.py` already handle the common `headline`/`title`/`content` column variants.

**Why synthetic sample data was used for the experiments in this repo**: downloading and
wiring in each Kaggle dataset (auth, licensing acceptance, schema normalization per
dataset) is a meaningful chunk of work in its own right and wasn't triggered
automatically. Instead, `scripts/generate_sample_data.py` produces a synthetic hourly
OHLCV series with sentiment intentionally correlated to a subset of future returns, so
Experiments A/B below demonstrate the pipeline mechanics correctly. Swap in a real dataset
via `--market-csv`/`--news-json` to get results that reflect actual market behavior.

## 13. Confidence Methodology (Read Before Trusting It)

The model produces one point estimate; it is not inherently probabilistic. Rather than
fabricate a confidence number, `timeseries/predict.py` uses **Monte Carlo Dropout** (Gal &
Ghahramani, 2016): dropout is kept active at inference time and N stochastic forward
passes are run over the same input. The spread (std) of these N predictions approximates
epistemic uncertainty for that specific input, then:

```
confidence = 1 / (1 + mc_dropout_std / validation_residual_std)
```

This gives confidence → 1 when the model's uncertainty on this specific input is small
relative to its typical validation error, and → 0 when it's large. **This is a heuristic,
relative-uncertainty signal, not a calibrated probability of correctness.** The
`confidence_method` field on every `PredictionResult` states this explicitly so downstream
consumers don't misinterpret it.

## 14. Evaluation and Experiment Results

Metrics computed (`evaluation/metrics.py`): MAE, RMSE, MAPE, R², directional accuracy, and
(for return targets) precision/recall/F1 for the UP/DOWN direction.

**Experiment A (price-only) vs. Experiment B (price + sentiment)**, same LSTM
architecture, same synthetic sample dataset, target = next-period return:

| Metric | A: price-only (LSTM) | B: price + sentiment (LSTM) | Delta |
|---|---|---|---|
| MAE | 0.002551 | 0.002517 | −0.000033 |
| RMSE | 0.003294 | 0.003258 | −0.000036 |
| R² | −0.0117 | **+0.0103** | +0.0220 |
| Directional accuracy | 49.7% | **51.4%** | +1.7 pts |
| F1 (UP) | 0.168 | **0.510** | +0.343 |
| F1 (DOWN) | 0.639 | 0.517 | −0.123 |

Full JSON reports (including the naive and Random Forest baselines) are in
`reports/experiment_a_price_only/metrics.json` and
`reports/experiment_b_price_sentiment/metrics.json`, with loss-curve, actual-vs-predicted,
and prediction-error plots alongside them. `notebooks/03_model_experiments.ipynb`
reproduces this comparison.

**On this synthetic dataset, adding sentiment features measurably improved the model** —
R² turned positive, directional accuracy improved, and F1 on the UP class improved
substantially (at some cost to F1 on the DOWN class, indicating the model shifted its
prediction bias rather than uniformly improving). This result should be read for what it
is: confirmation the *pipeline mechanics* correctly propagate a sentiment signal that was
deliberately injected into the synthetic data-generating process, not evidence that
sentiment predicts real markets. Re-run the same comparison (`--run-name` A/B as documented
in section 5) against a real dataset from section 12 before drawing any real conclusion.

Neither baseline (naive persistence, Random Forest) beat the LSTM in Experiment B, but
both are close to it on Experiment A — a reminder that a deep model is not automatically
better and should be compared against baselines every time, on every new dataset.

## 15. Testing

```powershell
python -m pytest tests/ -v
```

66 tests covering: CSV/JSON/record loading and timestamp parsing, cleaning (dedup,
invalid-OHLC detection, missing-value handling, outlier flagging), feature engineering
with explicit no-lookahead assertions, chronological split + train-only scaler fitting +
sequence construction (including carry-over context), target/alignment correctness,
sentiment aggregation with an explicit "never includes items after window end" check, text
preprocessing, sentiment label→score mapping (plus a live FinBERT integration test),
evaluation metrics, model persistence round-tripping, MC-dropout confidence, and the
end-to-end inference pipeline's output contract.

`test_sentiment_model.py`'s live FinBERT test and `test_inference_pipeline.py` are
skip-guarded (they need the model downloaded / a trained bundle present respectively) so
the suite still runs meaningfully in constrained environments.

## 16. Limitations

- **Do not treat any output of this system as investment advice or a guarantee of future
  performance.** Financial markets are not fully predictable from historical price and
  sentiment data alone.
- The experiment results in section 14 are demonstrated on synthetic sample data designed
  to exercise the pipeline, not real market data. Real-world performance will differ, and
  may show no benefit — or even a detriment — from sentiment features.
- FinBERT is a general finance-sentiment model; it is not fine-tuned to the specific asset,
  exchange, or market regime the backend eventually targets, and can be over-confident on
  routine corporate announcements that carry no real price-moving information.
- Confidence values are a relative epistemic-uncertainty heuristic (MC Dropout), not
  calibrated probabilities. Don't expose them as "% chance of being correct" without
  further calibration work.
- The Transformer model is experimental and unvalidated at scale — it hasn't been proven
  to outperform the LSTM on any real dataset in this repo.
- Outlier detection (`cleaner.flag_return_outliers`) uses a simple global mean/std z-score
  and is informational only (it flags, never auto-drops) — it's a coarse tool, not a robust
  statistical outlier test, and can be under-sensitive on small samples where a single
  outlier inflates its own reference standard deviation.
- No live data ingestion, streaming, or real-time retraining is implemented — this is a
  local, batch-oriented development pipeline pending backend integration.

## 17. Real Data, Backend and Dashboard

**Scrape a real dataset** (hourly OHLCV from Yahoo Finance, or Binance for crypto, and
headlines from Google News RSS queried week by week). Written to `data/raw/` (gitignored),
so re-run this on a fresh clone. Use Binance for crypto: Yahoo's hourly crypto bars report
zero volume for about half the hours.

```powershell
python scripts/scrape_data.py --symbol BTC-USD --query bitcoin --binance-pair BTCUSDT
python scripts/scrape_data.py --symbol AAPL --query "Apple stock"
```

Google News only gives day-level times for some items (shown as 08:00 GMT); those are moved
to the next midnight so a headline is never aligned with a bar before it was published.
Each weekly query returns at most 100 headlines.

**Train** one price-only and one price+sentiment model per asset. Headlines are sparse
relative to hourly bars, so the sentiment window is widened to 24h:

```powershell
python -m src.pipeline.training_pipeline --asset BTC-USD --run-name BTC-USD_price_only --market-csv data/raw/BTC-USD_market.csv
python -m src.pipeline.training_pipeline --asset BTC-USD --run-name BTC-USD_price_sentiment --market-csv data/raw/BTC-USD_market.csv --news-json data/raw/BTC-USD_news.json --use-sentiment --sentiment-window 24h
```

**Run the backend** (FastAPI, from the repo root) and **the dashboard** (React + Vite):

```powershell
uvicorn backend.main:app --port 8000
cd frontend; npm install; npm run dev
```

Open http://localhost:5173. The dashboard lists every trained run under
`models/price_prediction/`, charts the asset's recent bars, shows the latest headlines,
predicts the next bar on demand, and compares the deep model against the naive and
Random Forest baselines on the held-out test set.

API: `GET /api/models`, `GET /api/market/{symbol}`, `GET /api/news/{symbol}`,
`POST /api/predict` with `{"run": "<run name>"}` (optionally `market_data` and `news` lists
in the `MarketDataPoint` / `NewsItem` shapes to predict on your own data instead of the
scraped files).
