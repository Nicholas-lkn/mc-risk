"""Monte Carlo Greeks for terminal payoffs under GBM.

Three estimators of delta (dV/dS0) and vega (dV/dsigma):

- ``pathwise``: differentiate the payoff along each path. Needs a payoff that
  is continuous in S_T, so it is not available for digitals.
- ``lr``: likelihood ratio (score function). Differentiate the density instead
  of the payoff. Works for any payoff, at the price of a higher variance.
- ``fd``: central finite difference with common random numbers.
"""

from __future__ import annotations

import math
import time

import numpy as np

from mc_risk.result import PriceResult

GREEKS = ("delta", "vega")
METHODS = ("pathwise", "lr", "fd")
PAYOFFS = ("call", "put", "digital")


def _terminal(s0: float, r: float, q: float, sigma: float, t: float, z: np.ndarray) -> np.ndarray:
    return s0 * np.exp((r - q - 0.5 * sigma**2) * t + sigma * math.sqrt(t) * z)


def _payoff(s_t: np.ndarray, k: float, payoff: str) -> np.ndarray:
    if payoff == "call":
        return np.maximum(s_t - k, 0.0)
    if payoff == "put":
        return np.maximum(k - s_t, 0.0)
    return (s_t > k).astype(float)


def mc_greek(
    greek: str,
    method: str,
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    payoff: str = "call",
    *,
    q: float = 0.0,
    n_paths: int = 100_000,
    seed: int = 0,
    bump: float | None = None,
    common_random_numbers: bool = True,
) -> PriceResult:
    """Estimate delta or vega of a European payoff under GBM.

    Args:
        greek: "delta" or "vega".
        method: "pathwise", "lr" or "fd".
        payoff: "call", "put" or "digital" (cash-or-nothing call).
        bump: finite-difference half-width h. Defaults to 0.5% of spot for delta
            and 0.005 (half a volatility point) for vega.
        common_random_numbers: for "fd", reuse the same normals in both bumped
            valuations. Turning this off is only useful to show why it matters.

    Returns:
        PriceResult whose ``price`` field holds the Greek estimate.
    """
    if greek not in GREEKS or method not in METHODS or payoff not in PAYOFFS:
        raise ValueError(f"unsupported combination: {greek=}, {method=}, {payoff=}")
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    disc = math.exp(-r * t)
    sqrt_t = math.sqrt(t)

    if method == "pathwise":
        if payoff == "digital":
            raise ValueError(
                "pathwise estimator is not valid for a digital payoff: the payoff is "
                "discontinuous, its pathwise derivative is zero almost surely"
            )
        s_t = _terminal(s0, r, q, sigma, t, z)
        sign = 1.0 if payoff == "call" else -1.0
        itm = sign * (s_t - k) > 0
        if greek == "delta":
            ds = s_t / s0
        else:  # dS_T/dsigma = S_T (sqrt(T) Z - sigma T)
            ds = s_t * (sqrt_t * z - sigma * t)
        samples = disc * sign * itm * ds
    elif method == "lr":
        s_t = _terminal(s0, r, q, sigma, t, z)
        if greek == "delta":
            score = z / (s0 * sigma * sqrt_t)
        else:
            score = (z * z - 1.0) / sigma - z * sqrt_t
        samples = disc * _payoff(s_t, k, payoff) * score
    else:
        if bump is None:
            bump = 0.005 * s0 if greek == "delta" else 0.005
        z_down = z if common_random_numbers else rng.standard_normal(n_paths)
        if greek == "delta":
            up = _terminal(s0 + bump, r, q, sigma, t, z)
            down = _terminal(s0 - bump, r, q, sigma, t, z_down)
        else:
            up = _terminal(s0, r, q, sigma + bump, t, z)
            down = _terminal(s0, r, q, sigma - bump, t, z_down)
        samples = disc * (_payoff(up, k, payoff) - _payoff(down, k, payoff)) / (2.0 * bump)

    runtime = time.perf_counter() - start
    return PriceResult.from_samples(samples, n_paths, runtime, f"{greek}-{method}")


__all__ = ["GREEKS", "METHODS", "PAYOFFS", "mc_greek"]
