"""Monte Carlo pricing engine for payoffs on GBM paths.

``mc_price`` prices any payoff written as a function of a path array, with a
choice of sampling scheme (plain, antithetic, randomised Sobol) and an optional
control variate. ``price_european`` is a thin wrapper for vanilla options.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, qmc

from mc_risk.models.gbm import brownian_bridge, brownian_increments, gbm_paths_from_brownian
from mc_risk.payoffs import Payoff, european
from mc_risk.result import PriceResult

METHODS = ("plain", "antithetic", "sobol")
MAX_NORMALS_PER_BATCH = 4_000_000


@dataclass(frozen=True)
class ControlVariate:
    """A payoff whose price is known exactly.

    Attributes:
        payoff: maps paths to undiscounted control payoffs.
        price: exact discounted expectation of ``payoff``.
    """

    payoff: Payoff
    price: float


def underlying_control(s0: float, q: float, t: float) -> ControlVariate:
    """The terminal stock price: its discounted expectation is S0 * exp(-q T)."""
    return ControlVariate(lambda paths: paths[:, -1], s0 * math.exp(-q * t))


def apply_control(y: np.ndarray, x: np.ndarray, x_mean: float) -> tuple[np.ndarray, float]:
    """Return control-variate adjusted samples y - beta (x - E[x]) and beta.

    beta = Cov(y, x) / Var(x) is estimated from the same samples. This adds a
    bias of order 1/N, which is negligible next to the 1/sqrt(N) sampling error.
    """
    var_x = x.var(ddof=1)
    beta = 0.0 if var_x == 0 else float(np.cov(y, x, ddof=1)[0, 1] / var_x)
    return y - beta * (x - x_mean), beta


def sobol_normals(n_points_log2: int, dim: int, seed: int) -> np.ndarray:
    """Scrambled Sobol points mapped to standard normals, shape (2**m, dim)."""
    try:
        sampler = qmc.Sobol(d=dim, scramble=True, rng=seed)
    except TypeError:  # SciPy < 1.15 uses ``seed``
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    u = sampler.random_base2(n_points_log2)
    return norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))


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
    control: ControlVariate | None = None,
    seed: int = 0,
    n_scrambles: int = 64,
    bridge: bool = True,
) -> PriceResult:
    """Price a payoff under GBM by Monte Carlo.

    Args:
        payoff: callable mapping paths (n_paths, n_steps + 1) to undiscounted payoffs.
        s0, r, sigma, t, q: spot, risk-free rate, volatility, maturity, dividend yield.
        n_steps: number of monitoring dates (1 is enough for terminal payoffs).
        n_paths: total number of path evaluations (the budget). Antithetic uses
            n_paths / 2 pairs. Sobol rounds down to n_scrambles * 2**m points.
        method: "plain", "antithetic" or "sobol" (randomised quasi-Monte Carlo).
        control: optional control variate, combined with any method.
        seed: seed for the random generator. Same seed, same result.
        n_scrambles: independent Sobol scrambles used to estimate the RQMC error.
        bridge: build Sobol paths by Brownian bridge (ignored by other methods).

    Returns:
        PriceResult. For "sobol" the standard error is the standard deviation of
        the scramble means over sqrt(n_scrambles), and the CI uses a t quantile.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    disc = math.exp(-r * t)
    label = method + ("+cv" if control else "")

    def evaluate(w: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        paths = gbm_paths_from_brownian(s0, r, q, sigma, t, w)
        x = disc * control.payoff(paths) if control else None
        return disc * payoff(paths), x

    if method == "sobol":
        m = max(int(math.log2(max(n_paths // n_scrambles, 2))), 1)
        build = brownian_bridge if bridge else brownian_increments
        means = np.empty(n_scrambles)
        for i, sub_seed in enumerate(rng.integers(0, 2**31 - 1, size=n_scrambles)):
            y, x = evaluate(build(sobol_normals(m, n_steps, int(sub_seed)), t))
            if control:
                y, _ = apply_control(y, x, control.price)
            means[i] = y.mean()
        runtime = time.perf_counter() - start
        return PriceResult.from_samples(means, n_scrambles * 2**m, runtime, label, student_t=True)

    # Simulate in batches so path-dependent payoffs with many steps fit in memory.
    antithetic = method == "antithetic"
    n_draws = n_paths // 2 if antithetic else n_paths
    batch = max(1, min(n_draws, MAX_NORMALS_PER_BATCH // n_steps))
    ys, xs = [], []
    for first in range(0, n_draws, batch):
        size = min(batch, n_draws - first)
        w = brownian_increments(rng.standard_normal((size, n_steps)), t)
        y, x = evaluate(w)
        if antithetic:  # one sample per antithetic pair
            y_neg, x_neg = evaluate(-w)
            y = 0.5 * (y + y_neg)
            x = 0.5 * (x + x_neg) if control else None
        ys.append(y)
        xs.append(x)
    y = np.concatenate(ys)
    n_paths = 2 * n_draws if antithetic else n_draws
    if control:
        y, _ = apply_control(y, np.concatenate(xs), control.price)
    return PriceResult.from_samples(y, n_paths, time.perf_counter() - start, label)


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
    n_scrambles: int = 64,
) -> PriceResult:
    """European call or put under GBM.

    ``method`` is "plain", "antithetic", "control" (underlying as control
    variate) or "sobol".
    """
    control = None
    if method == "control":
        method, control = "plain", underlying_control(s0, q, t)
    return mc_price(
        european(k, option),
        s0,
        r,
        sigma,
        t,
        q=q,
        n_paths=n_paths,
        method=method,
        control=control,
        seed=seed,
        n_scrambles=n_scrambles,
    )


def variance_reduction_factor(baseline: PriceResult, other: PriceResult) -> float:
    """Variance of the baseline over variance of ``other`` at equal path budget."""
    base = baseline.std_error**2 * baseline.n_paths
    return base / (other.std_error**2 * other.n_paths)
