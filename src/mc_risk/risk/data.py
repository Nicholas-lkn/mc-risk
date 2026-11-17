"""Market data loading, CAD conversion and a synthetic data generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PORTFOLIO = ["RY.TO", "TD.TO", "SHOP.TO", "ENB.TO", "AAPL", "MSFT", "JPM"]
BENCHMARKS = ["XIU.TO", "SPY"]
US_TICKERS = ["AAPL", "MSFT", "JPM", "SPY"]
FX = "CAD=X"  # Yahoo symbol for USDCAD: Canadian dollars per US dollar
ALL_TICKERS = PORTFOLIO + BENCHMARKS + [FX]

# Portfolio weights by CAD market value. They sum to 1.
DEFAULT_WEIGHTS = {
    "RY.TO": 0.15,
    "TD.TO": 0.15,
    "SHOP.TO": 0.10,
    "ENB.TO": 0.10,
    "AAPL": 0.20,
    "MSFT": 0.20,
    "JPM": 0.10,
}


def load_prices(path: str | Path) -> pd.DataFrame:
    """Read a CSV of daily adjusted closes (first column dates, one column per ticker)."""
    prices = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    prices.index.name = "Date"
    missing = [c for c in ALL_TICKERS if c not in prices.columns or prices[c].notna().sum() == 0]
    if missing:
        raise ValueError(f"price file {path} has no data for: {missing}")
    return prices[ALL_TICKERS].astype(float)


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Align the Toronto and New York calendars.

    Days on which no equity traded (the FX series quotes on some of them) are
    dropped. On a day when only one exchange is closed, the closed market's
    prices are carried forward, so those names show a zero return that day and
    the full move on the next. Leading rows with missing data are dropped.
    """
    equities = [c for c in prices.columns if c != FX]
    traded = prices[equities].notna().any(axis=1)
    return prices.loc[traded].ffill(limit=5).dropna()


def to_cad(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert US-listed tickers to CAD using USDCAD. The FX column is kept."""
    out = prices.copy()
    for ticker in US_TICKERS:
        if ticker in out.columns:
            out[ticker] = out[ticker] * out[FX]
    return out


def cad_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns in CAD for every ticker (and the USDCAD return itself)."""
    return to_cad(clean_prices(prices)).pct_change().dropna()


def synthetic_prices(seed: int = 0, start: str = "2020-01-02", n_days: int = 1750) -> pd.DataFrame:
    """Generate prices shaped like the real download, for tests and offline runs.

    One market factor with GARCH(1,1) volatility and Student-t shocks drives all
    equities through fixed betas; each name adds its own t-distributed noise.
    USDCAD moves against the market (the US dollar strengthens in sell-offs).
    A stress window in February-March 2020 scales market volatility up and adds
    a negative drift. The numbers are not real and must not be read as such.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    stress = (dates >= "2020-02-20") & (dates <= "2020-03-23")

    market = np.empty(n_days)
    var = 0.01**2
    omega, alpha, beta = 2e-6, 0.09, 0.89
    shocks = rng.standard_t(5, size=n_days) / np.sqrt(5 / 3)
    for i in range(n_days):
        scale = 3.0 if stress[i] else 1.0
        market[i] = (3e-4 - (0.012 if stress[i] else 0.0)) + scale * np.sqrt(var) * shocks[i]
        var = omega + alpha * (market[i] - 3e-4) ** 2 / scale**2 + beta * var

    betas = {
        "RY.TO": 0.8,
        "TD.TO": 0.85,
        "SHOP.TO": 1.8,
        "ENB.TO": 0.7,
        "AAPL": 1.2,
        "MSFT": 1.15,
        "JPM": 1.1,
        "XIU.TO": 0.85,
        "SPY": 1.0,
    }
    idio = {
        "RY.TO": 0.007,
        "TD.TO": 0.008,
        "SHOP.TO": 0.03,
        "ENB.TO": 0.008,
        "AAPL": 0.011,
        "MSFT": 0.010,
        "JPM": 0.010,
        "XIU.TO": 0.003,
        "SPY": 0.002,
    }
    start_price = {
        "RY.TO": 80.0,
        "TD.TO": 55.0,
        "SHOP.TO": 55.0,
        "ENB.TO": 33.0,
        "AAPL": 72.0,
        "MSFT": 153.0,
        "JPM": 120.0,
        "XIU.TO": 21.0,
        "SPY": 295.0,
    }
    data = {}
    for ticker, b in betas.items():
        noise = idio[ticker] * rng.standard_t(5, size=n_days) / np.sqrt(5 / 3)
        data[ticker] = start_price[ticker] * np.exp(np.cumsum(b * market + noise - 1e-4))
    fx_ret = -0.15 * market + 0.004 * rng.standard_normal(n_days)
    data[FX] = 1.30 * np.exp(np.cumsum(fx_ret - fx_ret.mean()))
    frame = pd.DataFrame(data, index=dates)[ALL_TICKERS]
    frame.index.name = "Date"
    return frame
