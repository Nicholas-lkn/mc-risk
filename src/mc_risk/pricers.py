"""European pricers under the Heston and Merton models."""

from __future__ import annotations

import math
import time

import numpy as np

from mc_risk.models.heston import heston_paths
from mc_risk.models.merton import merton_paths
from mc_risk.result import PriceResult


def _vanilla(s_t: np.ndarray, k: float, option: str) -> np.ndarray:
    if option not in ("call", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")
    sign = 1.0 if option == "call" else -1.0
    return np.maximum(sign * (s_t - k), 0.0)


def price_european_heston(
    s0: float,
    k: float,
    v0: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    t: float,
    option: str = "call",
    *,
    q: float = 0.0,
    n_steps: int = 100,
    n_paths: int = 100_000,
    seed: int = 0,
) -> PriceResult:
    """European option under Heston by full-truncation Euler Monte Carlo.

    The estimate carries a discretisation bias that shrinks with ``n_steps``;
    the confidence interval covers sampling error only.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    batch = max(1, min(n_paths, 250_000))
    samples = []
    for first in range(0, n_paths, batch):
        size = min(batch, n_paths - first)
        s_t, _ = heston_paths(
            s0, v0, r, kappa, theta, xi, rho, t, n_steps, size, q=q, seed=rng, full_path=False
        )
        samples.append(math.exp(-r * t) * _vanilla(s_t, k, option))
    y = np.concatenate(samples)
    return PriceResult.from_samples(y, n_paths, time.perf_counter() - start, "heston-ft-euler")


def price_european_merton(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    lam: float,
    mu_j: float,
    delta: float,
    t: float,
    option: str = "call",
    *,
    q: float = 0.0,
    n_paths: int = 100_000,
    seed: int = 0,
) -> PriceResult:
    """European option under Merton jump diffusion (exact terminal simulation)."""
    start = time.perf_counter()
    s_t = merton_paths(s0, r, sigma, lam, mu_j, delta, t, 1, n_paths, q=q, seed=seed)[:, -1]
    y = math.exp(-r * t) * _vanilla(s_t, k, option)
    return PriceResult.from_samples(y, n_paths, time.perf_counter() - start, "merton-exact")
