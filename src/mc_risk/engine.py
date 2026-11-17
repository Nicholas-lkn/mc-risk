"""Monte Carlo pricing engine for payoffs on GBM paths.

``mc_price`` prices any payoff written as a function of a path array.
``price_european`` is a thin wrapper for vanilla options.
"""

from __future__ import annotations

import math
import time

import numpy as np

from mc_risk.models.gbm import brownian_increments, gbm_paths_from_brownian
from mc_risk.payoffs import Payoff, european
from mc_risk.result import PriceResult

METHODS = ("plain",)


def mc_price(
    payoff: Payoff,
    s0: float,
    r: float,
    sigma: float,
    t: float,
    *,
    q: float = 0.0,
    n_steps: int = 1,
    n_paths: int = 100_000,
    method: str = "plain",
    seed: int = 0,
) -> PriceResult:
    """Price a payoff under GBM by Monte Carlo.

    Args:
        payoff: callable mapping paths (n_paths, n_steps + 1) to undiscounted payoffs.
        s0, r, sigma, t, q: spot, risk-free rate, volatility, maturity, dividend yield.
        n_steps: number of monitoring dates (1 is enough for terminal payoffs).
        n_paths: number of simulated paths.
        method: sampling scheme, one of ``METHODS``.
        seed: seed for the random generator. Same seed, same result.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    disc = math.exp(-r * t)

    z = rng.standard_normal((n_paths, n_steps))
    paths = gbm_paths_from_brownian(s0, r, q, sigma, t, brownian_increments(z, t))
    y = disc * payoff(paths)
    return PriceResult.from_samples(y, n_paths, time.perf_counter() - start, method)


def price_european(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    option: str = "call",
    *,
    q: float = 0.0,
    n_paths: int = 100_000,
    method: str = "plain",
    seed: int = 0,
) -> PriceResult:
    """European call or put under GBM."""
    return mc_price(
        european(k, option), s0, r, sigma, t, q=q, n_paths=n_paths, method=method, seed=seed
    )
