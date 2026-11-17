"""Heston stochastic volatility model, full-truncation Euler scheme.

dS = (r - q) S dt + sqrt(v) S dW_s
dv = kappa (theta - v) dt + xi sqrt(v) dW_v,   d<W_s, W_v> = rho dt
"""

from __future__ import annotations

import math

import numpy as np


def heston_paths(
    s0: float,
    v0: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    t: float,
    n_steps: int,
    n_paths: int,
    q: float = 0.0,
    seed: int | np.random.Generator = 0,
    full_path: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Heston paths with the full-truncation Euler scheme.

    The variance can go negative under Euler. Full truncation (Lord, Koekkoek
    and van Dijk, 2010) lets the stored variance go negative but uses
    v+ = max(v, 0) wherever it enters a drift or a diffusion coefficient:

        v_{i+1} = v_i + kappa (theta - v_i+) dt + xi sqrt(v_i+ dt) Z_v
        ln S_{i+1} = ln S_i + (r - q - v_i+ / 2) dt + sqrt(v_i+ dt) Z_s

    Args:
        full_path: if False only the terminal values are kept, which saves memory.

    Returns:
        (S, v). Shapes (n_paths, n_steps + 1) if ``full_path`` else (n_paths,).
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    sqrt_dt = math.sqrt(dt)
    rho_bar = math.sqrt(1.0 - rho * rho)
    log_s = np.full(n_paths, math.log(s0))
    v = np.full(n_paths, float(v0))
    if full_path:
        s_out = np.empty((n_paths, n_steps + 1))
        v_out = np.empty((n_paths, n_steps + 1))
        s_out[:, 0], v_out[:, 0] = s0, v0
    for i in range(n_steps):
        z_v = rng.standard_normal(n_paths)
        z_s = rho * z_v + rho_bar * rng.standard_normal(n_paths)
        v_pos = np.maximum(v, 0.0)
        vol = np.sqrt(v_pos) * sqrt_dt
        log_s += (r - q - 0.5 * v_pos) * dt + vol * z_s
        v += kappa * (theta - v_pos) * dt + xi * vol * z_v
        if full_path:
            s_out[:, i + 1] = np.exp(log_s)
            v_out[:, i + 1] = np.maximum(v, 0.0)
    if full_path:
        return s_out, v_out
    return np.exp(log_s), np.maximum(v, 0.0)
