"""Closed forms for geometric-average Asian options under GBM."""

from __future__ import annotations

import math

from scipy.stats import norm

from mc_risk.analytic.black_scholes import bs_price


def geometric_asian_discrete(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_steps: int,
    option: str = "call",
    q: float = 0.0,
) -> float:
    """Geometric Asian with ``n_steps`` equally spaced fixings at T/n, ..., T.

    The log of the geometric average is normal with

        mean = ln S0 + (r - q - sigma^2/2) T (n+1) / (2n)
        var  = sigma^2 T (n+1)(2n+1) / (6 n^2)

    so the price is a Black-type formula. This is exact for the simulated grid,
    which is what a control variate needs.
    """
    n = n_steps
    mean = math.log(s0) + (r - q - 0.5 * sigma**2) * t * (n + 1) / (2 * n)
    var = sigma**2 * t * (n + 1) * (2 * n + 1) / (6 * n**2)
    std = math.sqrt(var)
    forward = math.exp(mean + 0.5 * var)
    d1 = (mean - math.log(k) + var) / std
    d2 = d1 - std
    df = math.exp(-r * t)
    if option == "call":
        return df * (forward * norm.cdf(d1) - k * norm.cdf(d2))
    if option == "put":
        return df * (k * norm.cdf(-d2) - forward * norm.cdf(-d1))
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def geometric_asian_kemna_vorst(
    s0: float, k: float, r: float, sigma: float, t: float, option: str = "call", q: float = 0.0
) -> float:
    """Kemna-Vorst (1990) price for a continuously monitored geometric Asian.

    Equivalent to Black-Scholes with volatility sigma / sqrt(3) and dividend
    yield (r + q + sigma^2 / 6) / 2. It is the n -> infinity limit of
    ``geometric_asian_discrete``.
    """
    return bs_price(s0, k, r, sigma / math.sqrt(3.0), t, option, q=0.5 * (r + q + sigma**2 / 6.0))
