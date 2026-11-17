"""Merton (1976) jump diffusion.

    dS / S = (r - q - lam * k) dt + sigma dW + (J - 1) dN

N is a Poisson process with intensity ``lam`` and ln J ~ N(mu_j, delta^2), so the
mean relative jump is k = exp(mu_j + delta^2 / 2) - 1. The drift is compensated
by ``lam * k`` so that the discounted price is a martingale.
"""

from __future__ import annotations

import math

import numpy as np


def merton_paths(
    s0: float,
    r: float,
    sigma: float,
    lam: float,
    mu_j: float,
    delta: float,
    t: float,
    n_steps: int,
    n_paths: int,
    q: float = 0.0,
    seed: int | np.random.Generator = 0,
) -> np.ndarray:
    """Simulate jump-diffusion paths. Exact in distribution at the grid points.

    Over a step the number of jumps is Poisson(lam dt); given n jumps their
    summed log size is N(n mu_j, n delta^2).

    Returns:
        Array of shape (n_paths, n_steps + 1); column 0 is ``s0``.
    """
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    k = math.exp(mu_j + 0.5 * delta**2) - 1.0
    drift = (r - q - lam * k - 0.5 * sigma**2) * dt
    z = rng.standard_normal((n_paths, n_steps))
    n_jumps = rng.poisson(lam * dt, size=(n_paths, n_steps))
    jumps = mu_j * n_jumps + delta * np.sqrt(n_jumps) * rng.standard_normal((n_paths, n_steps))
    log_s = np.cumsum(drift + sigma * math.sqrt(dt) * z + jumps, axis=1)
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(log_s)
    return paths
