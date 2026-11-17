"""One-day Value-at-Risk and Conditional VaR for a linear portfolio.

Conventions: returns are simple daily returns, weights are fractions of
portfolio value, and losses are positive numbers (loss = -portfolio return).
VaR and CVaR are therefore positive fractions of portfolio value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats


@dataclass(frozen=True)
class VarResult:
    """VaR and CVaR at confidence ``alpha``, as fractions of portfolio value."""

    method: str
    alpha: float
    var: float
    cvar: float
    n_scenarios: int
    dof: float | None = None


def var_cvar(losses: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    """Empirical VaR (alpha-quantile of losses) and CVaR (mean loss at or beyond VaR)."""
    losses = np.asarray(losses, dtype=float)
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    return var, float(tail.mean())


def historical_var(returns: np.ndarray, weights: np.ndarray, alpha: float = 0.99) -> VarResult:
    """Historical simulation: apply today's weights to every past day's returns."""
    losses = -(np.asarray(returns) @ np.asarray(weights))
    var, cvar = var_cvar(losses, alpha)
    return VarResult("historical", alpha, var, cvar, losses.size)


def gaussian_mc_var(
    returns: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.99,
    n_sims: int = 100_000,
    seed: int | np.random.Generator = 0,
) -> VarResult:
    """Monte Carlo VaR with multivariate normal returns (sample mean and covariance)."""
    returns = np.asarray(returns)
    rng = np.random.default_rng(seed)
    mean = returns.mean(axis=0)
    chol = np.linalg.cholesky(np.cov(returns, rowvar=False))
    sims = mean + rng.standard_normal((n_sims, returns.shape[1])) @ chol.T
    var, cvar = var_cvar(-(sims @ np.asarray(weights)), alpha)
    return VarResult("gaussian-mc", alpha, var, cvar, n_sims)


def fit_student_t_dof(returns: np.ndarray, bounds: tuple[float, float] = (2.1, 100.0)) -> float:
    """Maximum-likelihood degrees of freedom of a multivariate Student-t.

    The mean and covariance are fixed at their sample values (so the scale
    matrix is cov * (nu - 2) / nu) and the likelihood is maximised over nu only.
    """
    x = np.asarray(returns)
    n, d = x.shape
    cov = np.cov(x, rowvar=False)
    centred = x - x.mean(axis=0)
    maha = np.einsum("ij,ij->i", centred @ np.linalg.inv(cov), centred)
    _, logdet = np.linalg.slogdet(cov)

    def neg_loglik(nu: float) -> float:
        shrink = (nu - 2.0) / nu
        ll = (
            n
            * (
                special.gammaln((nu + d) / 2)
                - special.gammaln(nu / 2)
                - 0.5 * d * np.log(nu * np.pi)
                - 0.5 * (logdet + d * np.log(shrink))
            )
            - 0.5 * (nu + d) * np.log1p(maha / (shrink * nu)).sum()
        )
        return -ll

    res = optimize.minimize_scalar(neg_loglik, bounds=bounds, method="bounded")
    return float(res.x)


def student_t_mc_var(
    returns: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.99,
    n_sims: int = 100_000,
    seed: int | np.random.Generator = 0,
    dof: float | None = None,
) -> VarResult:
    """Monte Carlo VaR with Cholesky-correlated multivariate Student-t returns.

    X = mean + sqrt(nu / W) * L Z with W ~ chi-square(nu), Z standard normal and
    L the Cholesky factor of cov * (nu - 2) / nu, so the simulated covariance
    matches the sample covariance while the tails are heavier.
    """
    returns = np.asarray(returns)
    rng = np.random.default_rng(seed)
    nu = fit_student_t_dof(returns) if dof is None else float(dof)
    mean = returns.mean(axis=0)
    chol = np.linalg.cholesky(np.cov(returns, rowvar=False) * (nu - 2.0) / nu)
    z = rng.standard_normal((n_sims, returns.shape[1])) @ chol.T
    mix = np.sqrt(nu / rng.chisquare(nu, size=n_sims))
    sims = mean + z * mix[:, None]
    var, cvar = var_cvar(-(sims @ np.asarray(weights)), alpha)
    return VarResult("student-t-mc", alpha, var, cvar, n_sims, dof=nu)


def gaussian_var_closed_form(
    returns: np.ndarray, weights: np.ndarray, alpha: float = 0.99
) -> tuple[float, float]:
    """Exact normal VaR and CVaR for a linear portfolio (reference for the MC)."""
    returns, w = np.asarray(returns), np.asarray(weights)
    mu = float(returns.mean(axis=0) @ w)
    sigma = float(np.sqrt(w @ np.cov(returns, rowvar=False) @ w))
    z = stats.norm.ppf(alpha)
    return -mu + sigma * z, -mu + sigma * stats.norm.pdf(z) / (1 - alpha)


def student_t_var_closed_form(
    returns: np.ndarray, weights: np.ndarray, dof: float, alpha: float = 0.99
) -> tuple[float, float]:
    """Exact Student-t VaR and CVaR for a linear portfolio (reference for the MC)."""
    returns, w = np.asarray(returns), np.asarray(weights)
    mu = float(returns.mean(axis=0) @ w)
    scale = float(np.sqrt(w @ np.cov(returns, rowvar=False) @ w * (dof - 2.0) / dof))
    tq = stats.t.ppf(alpha, dof)
    es = stats.t.pdf(tq, dof) / (1 - alpha) * (dof + tq**2) / (dof - 1)
    return -mu + scale * tq, -mu + scale * es
