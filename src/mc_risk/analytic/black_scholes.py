"""Black-Scholes-Merton closed forms for European options and their Greeks."""

from __future__ import annotations

import math

from scipy.stats import norm


def _d1_d2(s0: float, k: float, r: float, sigma: float, t: float, q: float) -> tuple[float, float]:
    vol = sigma * math.sqrt(t)
    d1 = (math.log(s0 / k) + (r - q + 0.5 * sigma**2) * t) / vol
    return d1, d1 - vol


def bs_price(
    s0: float, k: float, r: float, sigma: float, t: float, option: str = "call", q: float = 0.0
) -> float:
    """European option price with continuous dividend yield ``q``."""
    d1, d2 = _d1_d2(s0, k, r, sigma, t, q)
    df_q, df_r = math.exp(-q * t), math.exp(-r * t)
    if option == "call":
        return s0 * df_q * norm.cdf(d1) - k * df_r * norm.cdf(d2)
    if option == "put":
        return k * df_r * norm.cdf(-d2) - s0 * df_q * norm.cdf(-d1)
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def bs_delta(
    s0: float, k: float, r: float, sigma: float, t: float, option: str = "call", q: float = 0.0
) -> float:
    """dPrice/dS0."""
    d1, _ = _d1_d2(s0, k, r, sigma, t, q)
    df_q = math.exp(-q * t)
    return df_q * norm.cdf(d1) if option == "call" else df_q * (norm.cdf(d1) - 1.0)


def bs_vega(s0: float, k: float, r: float, sigma: float, t: float, q: float = 0.0) -> float:
    """dPrice/dSigma (same for calls and puts), per unit of volatility."""
    d1, _ = _d1_d2(s0, k, r, sigma, t, q)
    return s0 * math.exp(-q * t) * norm.pdf(d1) * math.sqrt(t)
