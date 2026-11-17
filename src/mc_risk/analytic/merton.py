"""Merton (1976) jump-diffusion price as a Poisson-weighted series of Black-Scholes prices."""

from __future__ import annotations

import math

from mc_risk.analytic.black_scholes import bs_price


def merton_price(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    lam: float,
    mu_j: float,
    delta: float,
    t: float,
    option: str = "call",
    q: float = 0.0,
    n_terms: int = 60,
) -> float:
    """European option under Merton jump diffusion.

    Conditional on n jumps the terminal price is lognormal, so

        V = sum_n  e^{-lam' T} (lam' T)^n / n!  *  BS(S0, K, r_n, sigma_n, T)

    with lam' = lam (1 + k), sigma_n^2 = sigma^2 + n delta^2 / T,
    r_n = r - lam k + n ln(1 + k) / T and k = exp(mu_j + delta^2 / 2) - 1.
    """
    k_bar = math.exp(mu_j + 0.5 * delta**2) - 1.0
    lam_p = lam * (1.0 + k_bar)
    total = 0.0
    weight = math.exp(-lam_p * t)  # Poisson weight for n = 0
    for n in range(n_terms):
        sigma_n = math.sqrt(sigma**2 + n * delta**2 / t)
        r_n = r - lam * k_bar + n * math.log(1.0 + k_bar) / t
        total += weight * bs_price(s0, k, r_n, sigma_n, t, option, q)
        weight *= lam_p * t / (n + 1)
    return total
