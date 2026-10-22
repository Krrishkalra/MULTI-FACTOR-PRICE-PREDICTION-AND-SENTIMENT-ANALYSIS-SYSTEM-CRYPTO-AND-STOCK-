# Handover: Price Prediction + Sentiment System

Status as of 2026-10-22. Read this first, then README section 17 for the exact commands.

## What this project is

Predicts the next hourly return (and so UP/DOWN) for a crypto or stock asset from:
- technical indicators computed from hourly OHLCV bars, and
- optionally, FinBERT sentiment scores of news headlines, averaged over a 24h window.

It has three parts:

| Part | Where | What it does |
|---|---|---|
| ML pipeline | `src/` | Cleaning, features, FinBERT sentiment, LSTM/Transformer training, baselines, inference. Was already here and is unchanged apart from the fixes listed below. |
| Backend | `backend/main.py` | FastAPI app that loads trained models once and serves prices, news, metrics, plots and predictions. |
| Frontend | `frontend/` | React + Vite dashboard: asset/model picker, price chart, prediction card, test metrics vs baselines, training plots, latest headlines. |

Plus a scraper, `scripts/scrape_data.py`, that builds the real dataset.

## What was done

1. **Real dataset scraper** (`scripts/scrape_data.py`)
   - Prices: Yahoo Finance hourly bars for stocks, Binance public API for crypto. Yahoo's
     hourly crypto bars have zero volume for about half the hours, so don't use Yahoo for crypto.
   - News: Google News RSS, one query per week across the whole price history (capped at
     100 headlines per week by Google).
   - Headlines Google only dates to the day are moved to the next midnight, so a headline is
     never matched to a price bar before it was published.
   - Scraped so far: BTC-USD (17,495 bars, 8,400 headlines, Oct 2024 to Oct 2026) and
     AAPL (5,073 bars, 5,578 headlines, Nov 2023 to Oct 2026).
2. **Backend and frontend** as described above. The frontend builds without errors.
3. **Fixes to the existing ML code**
   - `volume_change` was infinite after any zero-volume bar, which crashed training on real
     data. It is now NaN there and those rows get dropped. A test covers it.
   - Inference now keeps FinBERT loaded between requests instead of reloading it every time.
   - The training command takes `--asset` and `--sentiment-window` options.
4. **Tests**: 64 pass, 2 skipped by design (they need FinBERT downloaded or a trained model).

## First result on real data

BTC-USD, price-only, held-out test set:

| Model | MAE | R² | Direction accuracy |
|---|---|---|---|
| Naive (predict no change) | 0.0023 | -0.0009 | n/a |
| Random Forest | 0.0042 | -1.37 | 51.1% |
| LSTM | 0.0023 | -0.0004 | 49.9% |

The LSTM is no better than a coin flip at calling the next hour's direction. That is normal
for hourly returns on real market data, and much worse than the README's synthetic-data
results, which were built to have a learnable signal. Don't present these models as
predictive yet.

## Important: data and models are not in git

`data/raw/`, `models/` and `reports/` are gitignored, so a fresh clone has no dataset and
no trained models. To rebuild them, run these commands from the repo root (all commands
are also in README section 17):

```powershell
python -m venv .venv
.venv\Scripts\pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements.txt fastapi uvicorn
.venv\Scripts\python scripts\scrape_data.py --symbol BTC-USD --query bitcoin --binance-pair BTCUSDT
.venv\Scripts\python scripts\scrape_data.py --symbol AAPL --query "Apple stock"
```

Then run the four training commands from README section 17: price-only and price +
sentiment, for each asset. The scrape takes about 5 minutes per asset. Training takes
roughly 10 minutes per price-only run on CPU, and the sentiment runs take longer because
they first score every headline with FinBERT (a 440 MB download the first time).

If `pip install` fails with dropped connections or "no matching distribution", the network
is flaky. Install the packages one at a time so finished ones stay installed.

## What's left

1. **Finish training the remaining 3 models.** BTC-USD price-only is trained. BTC-USD price +
   sentiment, AAPL price-only and AAPL price + sentiment were in progress at handover.
   Re-run whichever is missing from `models/price_prediction/`.
2. **Check the whole app end to end.** Start the backend and frontend, then for each model
   click "Predict next hour" and confirm you get a prediction, metrics and plots. The backend
   has not yet been run against trained models.
3. **Decide whether sentiment helps on real data.** Compare `*_price_only` against
   `*_price_sentiment` in the dashboard's metrics table. On the synthetic data it helped; on
   real data it may not.
4. **Improve the models, if that's the goal.** Ideas: predict daily instead of hourly (less
   noise, and it matches the day-level news timing better), predict direction directly as a
   classification task, and tune the sentiment window.
5. **Keep data fresh.** The scraper is run by hand. A scheduled re-scrape plus retrain would
   be needed for live use.
6. **Deployment.** Everything runs locally only. CORS allows only the Vite dev server
   (`http://localhost:5173`), and there's no authentication.

## Things to keep in mind

- The confidence number is a relative uncertainty score from Monte Carlo dropout, **not** a
  probability of being right. The UI says so; keep it that way.
- Nothing here is investment advice.
- Google News and Yahoo Finance are unofficial endpoints and can change or rate-limit
  without notice. Binance's public data API is more stable.
