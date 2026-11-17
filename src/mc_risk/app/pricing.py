"""Dispatch used by the Streamlit app: one entry point for every model and payoff.

Kept free of Streamlit imports so it can be tested without the UI.
"""

from __future__ import annotations

import pandas as pd

from mc_risk.analytic import (
    bgk_adjusted_barrier,
    bs_price,
    crr_price,
    down_and_out_call,
    heston_price,
    merton_price,
)
from mc_risk.engine import price_european
from mc_risk.exotics import price_american_lsm, price_asian, price_down_and_out_call
from mc_risk.pricers import price_european_heston, price_european_merton
from mc_risk.result import PriceResult

PAYOFFS = {
    "GBM": [
        "European call",
        "European put",
        "Asian arithmetic call",
        "Down-and-out call",
        "American put",
    ],
    "Heston": ["European call", "European put"],
    "Merton": ["European call", "European put"],
}


def methods_for(model: str, payoff: str) -> list[str]:
    """Sampling methods available for a model and payoff."""
    if model != "GBM":
        return ["plain"]
    if payoff.startswith("European"):
        return ["plain", "antithetic", "control", "sobol"]
    if payoff.startswith("Asian"):
        return ["control", "plain", "antithetic", "sobol"]
    if payoff.startswith("Down"):
        return ["plain", "antithetic", "sobol"]
    return ["lsm"]


def price(
    model: str, payoff: str, method: str, p: dict, n_paths: int, seed: int = 0
) -> tuple[PriceResult, float | None, str]:
    """Price one instrument.

    Args:
        p: parameter dict. Always s0, k, r, q, t; plus sigma (GBM, Merton),
            n_steps and barrier where relevant, v0/kappa/theta/xi/rho (Heston),
            lam/mu_j/delta (Merton).

    Returns:
        (result, reference price or None, name of the reference).
    """
    s0, k, r, q, t = p["s0"], p["k"], p["r"], p["q"], p["t"]
    option = "put" if payoff.endswith("put") else "call"
    if model == "Heston":
        h = {x: p[x] for x in ("v0", "kappa", "theta", "xi", "rho")}
        res = price_european_heston(
            s0,
            k,
            r=r,
            t=t,
            option=option,
            q=q,
            n_steps=p["n_steps"],
            n_paths=n_paths,
            seed=seed,
            **h,
        )
        return res, heston_price(s0, k, r=r, t=t, option=option, q=q, **h), "Heston semi-analytic"
    if model == "Merton":
        m = {x: p[x] for x in ("sigma", "lam", "mu_j", "delta")}
        res = price_european_merton(
            s0, k, r, t=t, option=option, q=q, n_paths=n_paths, seed=seed, **m
        )
        return res, merton_price(s0, k, r, t=t, option=option, q=q, **m), "Merton series"

    sigma = p["sigma"]
    if payoff.startswith("European"):
        res = price_european(
            s0, k, r, sigma, t, option, q=q, n_paths=n_paths, method=method, seed=seed
        )
        return res, bs_price(s0, k, r, sigma, t, option, q), "Black-Scholes"
    if payoff.startswith("Asian"):
        use_cv = method == "control"
        res = price_asian(
            s0,
            k,
            r,
            sigma,
            t,
            q=q,
            n_steps=p["n_steps"],
            n_paths=n_paths,
            method="plain" if use_cv else method,
            control_variate=use_cv or method == "sobol",
            seed=seed,
        )
        return res, None, "no closed form"
    if payoff.startswith("Down"):
        res = price_down_and_out_call(
            s0,
            k,
            p["barrier"],
            r,
            sigma,
            t,
            q=q,
            n_monitor=p["n_steps"],
            n_paths=n_paths,
            method=method,
            seed=seed,
        )
        ref = None
        if p["barrier"] <= k and p["barrier"] < s0:
            shifted = bgk_adjusted_barrier(p["barrier"], sigma, t, p["n_steps"])
            ref = down_and_out_call(s0, k, shifted, r, sigma, t, q)
        return res, ref, "closed form with BGK correction"
    res = price_american_lsm(
        s0, k, r, sigma, t, "put", q=q, n_steps=p["n_steps"], n_paths=n_paths, seed=seed
    )
    tree_steps = p["n_steps"] * max(1, 1000 // p["n_steps"])
    ref = crr_price(
        s0, k, r, sigma, t, n_steps=tree_steps, q=q, exercise_every=tree_steps // p["n_steps"]
    )
    return res, ref, f"CRR tree, {tree_steps} steps, same exercise dates"


def convergence(
    model: str, payoff: str, method: str, p: dict, max_paths: int, seed: int = 0, n_points: int = 7
) -> pd.DataFrame:
    """Independent estimates at increasing path counts (each halves the previous)."""
    rows = []
    for i in range(n_points - 1, -1, -1):
        n = max(max_paths // 2**i, 256)
        res, _, _ = price(model, payoff, method, p, n, seed + i)
        rows.append((res.n_paths, res.price, res.ci99[0], res.ci99[1], res.std_error))
    return pd.DataFrame(rows, columns=["n_paths", "price", "ci_low", "ci_high", "std_error"])
