"""Closed form for a continuously monitored down-and-out call under GBM."""

from __future__ import annotations

import math

from scipy.stats import norm

from mc_risk.analytic.black_scholes import bs_price

BGK_BETA = 0.5826  # -zeta(1/2) / sqrt(2 pi), Broadie-Glasserman-Kou (1997)


def down_and_out_call(
    s0: float, k: float, barrier: float, r: float, sigma: float, t: float, q: float = 0.0
) -> float:
    """Down-and-out call, continuous monitoring, barrier at or below the strike.

    Price = vanilla call - down-and-in call (Reiner and Rubinstein, 1991).
    """
    if barrier > k:
        raise NotImplementedError("closed form implemented for barrier <= strike only")
    if s0 <= barrier:
        return 0.0
    vol = sigma * math.sqrt(t)
    lam = (r - q + 0.5 * sigma**2) / sigma**2
    y = math.log(barrier**2 / (s0 * k)) / vol + lam * vol
    ratio = barrier / s0
    down_and_in = s0 * math.exp(-q * t) * ratio ** (2 * lam) * norm.cdf(y) - k * math.exp(
        -r * t
    ) * ratio ** (2 * lam - 2) * norm.cdf(y - vol)
    return bs_price(s0, k, r, sigma, t, "call", q) - down_and_in


def bgk_adjusted_barrier(barrier: float, sigma: float, t: float, n_monitor: int) -> float:
    """Shift a down barrier so the continuous formula prices discrete monitoring.

    A barrier checked at ``n_monitor`` equally spaced dates is approximately a
    continuous barrier at H * exp(-beta * sigma * sqrt(T / m)): the discretely
    watched path can dip below H between dates without being knocked out.
    """
    return barrier * math.exp(-BGK_BETA * sigma * math.sqrt(t / n_monitor))
