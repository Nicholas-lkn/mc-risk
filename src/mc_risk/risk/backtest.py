"""Rolling VaR backtest with Kupiec and Christoffersen tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

from mc_risk.risk.var import (
    fit_student_t_dof,
    gaussian_mc_var,
    historical_var,
    student_t_mc_var,
)

METHODS = ("historical", "gaussian-mc", "student-t-mc")


def rolling_backtest(
    returns: pd.DataFrame,
    weights: np.ndarray,
    method: str,
    alpha: float = 0.99,
    window: int = 500,
    n_sims: int = 20_000,
    seed: int = 0,
    refit_dof_every: int = 21,
) -> pd.DataFrame:
    """One-day-ahead VaR forecasts against realised losses.

    For each day t the model is fitted on the ``window`` days before t (never
    including t) and its VaR is compared with the loss realised on day t.

    Returns:
        DataFrame indexed by date with columns loss, var, cvar and exceed.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    x = returns.to_numpy()
    w = np.asarray(weights)
    rng = np.random.default_rng(seed)
    rows = []
    dof = None
    for i, t in enumerate(range(window, len(x))):
        hist = x[t - window : t]
        if method == "historical":
            res = historical_var(hist, w, alpha)
        elif method == "gaussian-mc":
            res = gaussian_mc_var(hist, w, alpha, n_sims, rng)
        else:
            if i % refit_dof_every == 0:
                dof = fit_student_t_dof(hist)
            res = student_t_mc_var(hist, w, alpha, n_sims, rng, dof=dof)
        loss = -float(x[t] @ w)
        rows.append((loss, res.var, res.cvar, loss > res.var))
    out = pd.DataFrame(
        rows, columns=["loss", "var", "cvar", "exceed"], index=returns.index[window:]
    )
    return out


def kupiec_pof(exceed: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    """Kupiec proportion-of-failures test (unconditional coverage).

    H0: the exceedance probability is 1 - alpha. Returns (LR statistic, p-value);
    LR is chi-square with 1 degree of freedom under H0.
    """
    exceed = np.asarray(exceed, dtype=bool)
    n, x = exceed.size, int(exceed.sum())
    p = 1.0 - alpha
    pi = x / n
    log_h0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    log_h1 = xlogy(n - x, 1 - pi) + xlogy(x, pi)
    lr = max(-2.0 * (log_h0 - log_h1), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_independence(exceed: np.ndarray) -> tuple[float, float]:
    """Christoffersen test that exceedances do not cluster.

    H0: the probability of an exceedance today does not depend on whether there
    was one yesterday (first-order Markov alternative). LR is chi-square(1).
    """
    e = np.asarray(exceed, dtype=int)
    prev, curr = e[:-1], e[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    log_h0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_h1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = max(-2.0 * (log_h0 - log_h1), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def conditional_coverage(exceed: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    """Christoffersen conditional coverage: Kupiec + independence, chi-square(2)."""
    lr = kupiec_pof(exceed, alpha)[0] + christoffersen_independence(exceed)[0]
    return float(lr), float(stats.chi2.sf(lr, 2))


def summarise_backtest(bt: pd.DataFrame, alpha: float = 0.99) -> dict:
    """Exceedance counts and test p-values for one backtest."""
    exceed = bt["exceed"].to_numpy()
    n, x = exceed.size, int(exceed.sum())
    worst = bt.loc[bt["exceed"], "loss"] / bt.loc[bt["exceed"], "var"]
    return {
        "days": n,
        "exceedances": x,
        "expected": n * (1 - alpha),
        "rate": x / n,
        "kupiec_p": kupiec_pof(exceed, alpha)[1],
        "independence_p": christoffersen_independence(exceed)[1],
        "cond_coverage_p": conditional_coverage(exceed, alpha)[1],
        "mean_exceed_ratio": float(worst.mean()) if x else float("nan"),
    }
