"""Geometric Brownian motion path generation (exact in distribution)."""

from __future__ import annotations

import numpy as np


def brownian_increments(z: np.ndarray, t: float) -> np.ndarray:
    """Brownian motion on an equally spaced grid from normals, built step by step.

    Args:
        z: standard normals, shape (n_paths, n_steps).
        t: horizon in years.

    Returns:
        W at times t/n, 2t/n, ..., t with shape (n_paths, n_steps).
    """
    n_steps = z.shape[1]
    return np.cumsum(z * np.sqrt(t / n_steps), axis=1)


def gbm_paths_from_brownian(
    s0: float, r: float, q: float, sigma: float, t: float, w: np.ndarray
) -> np.ndarray:
    """Map Brownian paths to GBM paths. Returns shape (n_paths, n_steps + 1) with S_0 first."""
    n_paths, n_steps = w.shape
    times = np.linspace(t / n_steps, t, n_steps)
    log_s = (r - q - 0.5 * sigma**2) * times + sigma * w
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(log_s)
    return paths


def gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    q: float = 0.0,
    seed: int | np.random.Generator = 0,
) -> np.ndarray:
    """Simulate GBM paths under the risk-neutral measure.

    dS = (r - q) S dt + sigma S dW. The log-Euler scheme used here is exact,
    so there is no discretisation bias at the grid points.

    Returns:
        Array of shape (n_paths, n_steps + 1); column 0 is ``s0``.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps))
    return gbm_paths_from_brownian(s0, r, q, sigma, t, brownian_increments(z, t))


def brownian_bridge(z: np.ndarray, t: float) -> np.ndarray:
    """Brownian motion on an equally spaced grid built by Brownian bridge.

    The first normal fixes W(T), the second the midpoint given the ends, and so
    on by bisection. The paths have the same law as ``brownian_increments``, but
    most of the variance is carried by the first few columns of ``z``. That
    matters for Sobol points, whose leading dimensions are the best distributed.

    Args:
        z: standard normals, shape (n_paths, n_steps). Any n_steps >= 1 works.
        t: horizon in years.

    Returns:
        W at times t/n, ..., t with shape (n_paths, n_steps).
    """
    n_paths, n = z.shape
    grid = np.linspace(0.0, t, n + 1)
    w = np.zeros((n_paths, n + 1))
    w[:, n] = np.sqrt(t) * z[:, 0]
    used = 1
    queue = [(0, n)]
    while queue:
        left, right = queue.pop(0)
        if right - left < 2:
            continue
        mid = (left + right) // 2
        tl, tm, tr = grid[left], grid[mid], grid[right]
        mean = ((tr - tm) * w[:, left] + (tm - tl) * w[:, right]) / (tr - tl)
        std = np.sqrt((tm - tl) * (tr - tm) / (tr - tl))
        w[:, mid] = mean + std * z[:, used]
        used += 1
        queue += [(left, mid), (mid, right)]
    return w[:, 1:]
