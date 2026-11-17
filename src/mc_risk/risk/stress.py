"""Stress scenarios for the CAD-based portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mc_risk.risk.data import US_TICKERS


def historical_replay(
    returns: pd.DataFrame, weights: dict[str, float], start: str, end: str
) -> dict:
    """Replay a historical window with today's weights, rebalanced daily.

    Returns the compounded portfolio return over the window, the worst single
    day and the maximum drawdown inside the window.
    """
    window = returns.loc[start:end, list(weights)]
    if window.empty:
        raise ValueError(f"no data between {start} and {end}")
    daily = window.to_numpy() @ np.array(list(weights.values()))
    wealth = np.cumprod(1.0 + daily)
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return {
        "days": int(daily.size),
        "total_return": float(wealth[-1] - 1.0),
        "worst_day": float(daily.min()),
        "worst_day_date": window.index[int(daily.argmin())].date().isoformat(),
        "max_drawdown": float((wealth / peak - 1.0).min()),
    }


def fx_shock(weights: dict[str, float], cad_change: float) -> float:
    """Instant portfolio return if the Canadian dollar moves by ``cad_change`` against USD.

    ``cad_change = -0.10`` means the CAD loses 10% of its USD value, so USDCAD
    rises by 1 / 0.9 - 1 = 11.1% and US holdings are worth more in CAD. Local
    prices are held fixed; only the translation effect is measured.
    """
    usdcad_return = 1.0 / (1.0 + cad_change) - 1.0
    us_weight = sum(w for t, w in weights.items() if t in US_TICKERS)
    return float(us_weight * usdcad_return)


def factor_shock(weights: dict[str, float], shocks: dict[str, float]) -> float:
    """Instant portfolio return for given per-ticker returns (missing tickers: zero)."""
    return float(sum(w * shocks.get(t, 0.0) for t, w in weights.items()))
