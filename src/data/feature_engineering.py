"""
Technical / time-series feature engineering for OHLCV market data.

Leakage-prevention policy (see also README "Data Leakage Prevention"):
    - Every feature computed here (returns, moving averages, EMA, RSI, MACD,
      rolling volatility, volume change) uses ``pandas.Series.rolling`` /
      ``.ewm`` / ``.shift`` in a strictly backward-looking way: the value at
      row ``t`` only uses data from rows ``<= t``. Pandas rolling/ewm windows
      are backward-looking by construction (they never reach into future
      rows), so as long as we never call them with ``center=True`` (we never
      do) there is no leakage introduced at this stage.
    - The target variable is deliberately NOT created in this module. It is
      created later in the dataset-assembly step, specifically shifted so
      that a feature row at time t is paired with a target that depends on
      time t + horizon (see timeseries/dataset.py). This keeps "feature
      engineering" and "label creation" separated, which makes leakage bugs
      much easier to spot in review.
    - Feature scaling/normalization is deliberately NOT done in this module.
      Scalers must be fit on the training split only (see
      timeseries/dataset.py), so scaling happens after the chronological
      train/val/test split, not here.

The feature set is intentionally small and explainable rather than an
exhaustive indicator dump, per project requirements.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Add simple and log returns computed from the current and prior close."""
    df = df.copy()
    df["return"] = df[price_col].pct_change()
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    return df


def add_moving_averages(df: pd.DataFrame, price_col: str = "close", windows=(5, 10, 20)) -> pd.DataFrame:
    """Add simple moving averages for each window in ``windows``."""
    df = df.copy()
    for w in windows:
        df[f"ma_{w}"] = df[price_col].rolling(window=w, min_periods=w).mean()
    return df


def add_exponential_moving_averages(df: pd.DataFrame, price_col: str = "close", windows=(12, 26)) -> pd.DataFrame:
    """Add exponential moving averages for each span in ``windows``."""
    df = df.copy()
    for w in windows:
        df[f"ema_{w}"] = df[price_col].ewm(span=w, adjust=False, min_periods=w).mean()
    return df


def add_rolling_volatility(df: pd.DataFrame, return_col: str = "return", window: int = 20) -> pd.DataFrame:
    """Add rolling standard deviation of returns as a volatility proxy."""
    df = df.copy()
    if return_col not in df.columns:
        df = add_returns(df)
    df[f"volatility_{window}"] = df[return_col].rolling(window=window, min_periods=window).std()
    return df


def add_rsi(df: pd.DataFrame, price_col: str = "close", window: int = 14) -> pd.DataFrame:
    """Add the Relative Strength Index (RSI) using Wilder's smoothing.

    RSI = 100 - 100 / (1 + RS), RS = avg_gain / avg_loss over ``window``
    periods, computed with an exponential (Wilder) moving average so that
    only past observations contribute to each value.
    """
    df = df.copy()
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # Where avg_loss is 0 and avg_gain > 0, RSI should be 100 (no losses at all).
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    # Where both are 0 (no movement), RSI is conventionally 50 (neutral).
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    df[f"rsi_{window}"] = rsi
    return df


def add_macd(df: pd.DataFrame, price_col: str = "close", fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add MACD line, signal line, and histogram."""
    df = df.copy()
    ema_fast = df[price_col].ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = macd_line - signal_line
    return df


def add_volume_change(df: pd.DataFrame, volume_col: str = "volume") -> pd.DataFrame:
    """Add the percentage change in traded volume vs the prior period."""
    df = df.copy()
    # A change from zero volume is undefined, not infinite; NaN rows are dropped with the warm-up rows.
    df["volume_change"] = df[volume_col].pct_change().replace([np.inf, -np.inf], np.nan)
    return df


def engineer_features(
    df: pd.DataFrame,
    price_col: str = "close",
    ma_windows=(5, 10, 20),
    ema_windows=(12, 26),
    rsi_window: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    volatility_window: int = 20,
) -> pd.DataFrame:
    """Run the full, explainable feature-engineering sequence.

    Args mirror ``DataConfig`` fields so this function can be called directly
    with ``**asdict(config.data)``-style unpacking from the pipeline code.

    Returns:
        A new DataFrame with all engineered columns appended. Rows at the
        start of the series will contain NaNs wherever a rolling window has
        not yet accumulated enough observations -- these should be dropped
        downstream (after the train/val/test split boundary is known) rather
        than here, to keep this function a pure, side-effect-free transform.
    """
    out = df.copy()
    out = add_returns(out, price_col=price_col)
    out = add_moving_averages(out, price_col=price_col, windows=ma_windows)
    out = add_exponential_moving_averages(out, price_col=price_col, windows=ema_windows)
    out = add_rolling_volatility(out, return_col="return", window=volatility_window)
    out = add_rsi(out, price_col=price_col, window=rsi_window)
    out = add_macd(out, price_col=price_col, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    out = add_volume_change(out)

    n_total = len(out)
    n_warmup = int(out.isna().any(axis=1).sum())
    logger.info(
        "Engineered features for %d rows (%d rows contain warm-up NaNs from rolling windows)",
        n_total,
        n_warmup,
    )
    return out


def get_engineered_feature_names(
    ma_windows=(5, 10, 20),
    ema_windows=(12, 26),
    rsi_window: int = 14,
    volatility_window: int = 20,
) -> list[str]:
    """Return the list of column names produced by ``engineer_features``.

    Kept in sync manually with ``engineer_features`` -- used by the pipeline
    to build the model's feature list without re-deriving it from a DataFrame.
    """
    names = ["return", "log_return"]
    names += [f"ma_{w}" for w in ma_windows]
    names += [f"ema_{w}" for w in ema_windows]
    names += [f"volatility_{volatility_window}"]
    names += [f"rsi_{rsi_window}"]
    names += ["macd", "macd_signal", "macd_hist"]
    names += ["volume_change"]
    return names
