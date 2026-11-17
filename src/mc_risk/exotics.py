"""Exotic option pricers under GBM: Asian, barrier and American (Longstaff-Schwartz)."""

from __future__ import annotations

import math
import time

import numpy as np

from mc_risk.analytic.asian import geometric_asian_discrete
from mc_risk.engine import ControlVariate, mc_price
from mc_risk.models.gbm import gbm_paths
from mc_risk.payoffs import asian_arithmetic, asian_geometric, down_and_out
from mc_risk.result import PriceResult


def price_asian(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    option: str = "call",
    *,
    q: float = 0.0,
    n_steps: int = 50,
    n_paths: int = 100_000,
    average: str = "arithmetic",
    method: str = "plain",
    control_variate: bool = True,
    seed: int = 0,
) -> PriceResult:
    """Asian option on the average of ``n_steps`` equally spaced fixings.

    For the arithmetic average the geometric Asian, whose price is known in
    closed form on the same fixing dates, is used as a control variate unless
    ``control_variate=False``.
    """
    if average == "geometric":
        payoff, control = asian_geometric(k, option), None
    elif average == "arithmetic":
        payoff, control = asian_arithmetic(k, option), None
        if control_variate:
            exact = geometric_asian_discrete(s0, k, r, sigma, t, n_steps, option, q)
            control = ControlVariate(asian_geometric(k, option), exact)
    else:
        raise ValueError(f"average must be 'arithmetic' or 'geometric', got {average!r}")
    return mc_price(
        payoff,
        s0,
        r,
        sigma,
        t,
        q=q,
        n_steps=n_steps,
        n_paths=n_paths,
        method=method,
        control=control,
        seed=seed,
    )


def price_down_and_out_call(
    s0: float,
    k: float,
    barrier: float,
    r: float,
    sigma: float,
    t: float,
    *,
    q: float = 0.0,
    n_monitor: int = 50,
    n_paths: int = 100_000,
    method: str = "plain",
    seed: int = 0,
) -> PriceResult:
    """Down-and-out call with the barrier checked at ``n_monitor`` equally spaced dates."""
    return mc_price(
        down_and_out(k, barrier, "call"),
        s0,
        r,
        sigma,
        t,
        q=q,
        n_steps=n_monitor,
        n_paths=n_paths,
        method=method,
        seed=seed,
    )


def price_american_lsm(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    option: str = "put",
    *,
    q: float = 0.0,
    n_steps: int = 50,
    n_paths: int = 100_000,
    degree: int = 3,
    seed: int = 0,
) -> PriceResult:
    """American (strictly: Bermudan with ``n_steps`` dates) option by Longstaff-Schwartz.

    Working backwards, the continuation value at each exercise date is
    estimated by least-squares regression of the discounted future cash flow on
    a polynomial in S / K, using in-the-money paths only. A path is exercised
    when the immediate payoff exceeds the fitted continuation value.

    The standard error treats the per-path discounted cash flows as independent.
    That ignores the dependence introduced by the shared regression, so it is a
    slight understatement. The same paths are used to fit and to value.
    """
    start = time.perf_counter()
    sign = 1.0 if option == "call" else -1.0
    paths = gbm_paths(s0, r, sigma, t, n_steps, n_paths, q=q, seed=seed)
    disc = math.exp(-r * t / n_steps)

    cash = np.maximum(sign * (paths[:, -1] - k), 0.0)
    for i in range(n_steps - 1, 0, -1):
        cash *= disc
        spot = paths[:, i]
        exercise = np.maximum(sign * (spot - k), 0.0)
        itm = np.flatnonzero(exercise > 0)
        if itm.size <= degree + 1:
            continue
        basis = np.vander(spot[itm] / k, degree + 1)
        coef, *_ = np.linalg.lstsq(basis, cash[itm], rcond=None)
        stop = exercise[itm] > basis @ coef
        cash[itm[stop]] = exercise[itm[stop]]
    cash *= disc

    res = PriceResult.from_samples(cash, n_paths, time.perf_counter() - start, "lsm")
    intrinsic = max(sign * (s0 - k), 0.0)
    if intrinsic > res.price:  # exercising today is better than any continuation
        return PriceResult(intrinsic, 0.0, (intrinsic, intrinsic), n_paths, res.runtime, "lsm")
    return res
