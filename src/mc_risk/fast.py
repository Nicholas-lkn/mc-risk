"""Numba-accelerated kernels. Optional: install with ``uv sync --extra fast``.

Three kernels, each with the same model as its NumPy counterpart:

- ``gbm_paths_numba``: full GBM paths from pre-drawn normals, parallel over paths.
  Same normals in, same paths out as ``mc_risk.models.gbm_paths``.
- ``asian_payoffs_numba``: arithmetic Asian payoffs with the random numbers drawn
  inside the loop, so no (n_paths, n_steps) array is ever allocated.
- ``heston_terminal_numba``: full-truncation Euler terminal prices, same idea.

The in-loop kernels run in parallel over fixed chunks of paths. Each chunk
reseeds its thread's generator from (seed, chunk index), so the result does not
depend on the number of threads. They use Numba's own generator (Mersenne
Twister), so the numbers differ from the NumPy pricers for the same seed while
agreeing in distribution.
"""

from __future__ import annotations

import math
import time

import numpy as np

try:
    from numba import njit, prange
except ImportError as exc:  # pragma: no cover
    raise ImportError("mc_risk.fast needs numba: install with `uv sync --extra fast`") from exc

from mc_risk.result import PriceResult

CHUNK = 1024  # paths per independently seeded chunk


@njit(parallel=True, cache=True)
def _gbm_kernel(s0, drift, vol, z, out):  # pragma: no cover - compiled
    n_paths, n_steps = z.shape
    for i in prange(n_paths):
        log_s = math.log(s0)
        out[i, 0] = s0
        for j in range(n_steps):
            log_s += drift + vol * z[i, j]
            out[i, j + 1] = math.exp(log_s)


@njit(parallel=True, cache=True)
def _asian_kernel(s0, drift, vol, k, sign, n_steps, n_paths, seed):  # pragma: no cover
    out = np.empty(n_paths)
    log_s0 = math.log(s0)
    n_chunks = (n_paths + CHUNK - 1) // CHUNK
    for c in prange(n_chunks):
        np.random.seed((seed * 1_000_003 + c) % 4_294_967_296)
        for i in range(c * CHUNK, min((c + 1) * CHUNK, n_paths)):
            log_s = log_s0
            total = 0.0
            for _ in range(n_steps):
                log_s += drift + vol * np.random.standard_normal()
                total += math.exp(log_s)
            out[i] = max(sign * (total / n_steps - k), 0.0)
    return out


@njit(parallel=True, cache=True)
def _heston_kernel(
    s0, v0, r, q, kappa, theta, xi, rho, t, n_steps, n_paths, seed
):  # pragma: no cover
    out = np.empty(n_paths)
    dt = t / n_steps
    sqrt_dt = math.sqrt(dt)
    rho_bar = math.sqrt(1.0 - rho * rho)
    log_s0 = math.log(s0)
    n_chunks = (n_paths + CHUNK - 1) // CHUNK
    for c in prange(n_chunks):
        np.random.seed((seed * 1_000_003 + c) % 4_294_967_296)
        for i in range(c * CHUNK, min((c + 1) * CHUNK, n_paths)):
            log_s = log_s0
            v = v0
            for _ in range(n_steps):
                z_v = np.random.standard_normal()
                z_s = rho * z_v + rho_bar * np.random.standard_normal()
                v_pos = max(v, 0.0)
                vol = math.sqrt(v_pos) * sqrt_dt
                log_s += (r - q - 0.5 * v_pos) * dt + vol * z_s
                v += kappa * (theta - v_pos) * dt + xi * vol * z_v
            out[i] = math.exp(log_s)
    return out


def gbm_paths_numba(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    q: float = 0.0,
    seed: int | np.random.Generator = 0,
) -> np.ndarray:
    """Drop-in replacement for ``mc_risk.models.gbm_paths`` (identical output)."""
    z = np.random.default_rng(seed).standard_normal((n_paths, n_steps))
    out = np.empty((n_paths, n_steps + 1))
    dt = t / n_steps
    _gbm_kernel(s0, (r - q - 0.5 * sigma**2) * dt, sigma * math.sqrt(dt), z, out)
    return out


def price_asian_numba(
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
    seed: int = 0,
) -> PriceResult:
    """Plain Monte Carlo arithmetic Asian option with an allocation-free kernel."""
    start = time.perf_counter()
    dt = t / n_steps
    sign = 1.0 if option == "call" else -1.0
    payoffs = _asian_kernel(
        s0, (r - q - 0.5 * sigma**2) * dt, sigma * math.sqrt(dt), k, sign, n_steps, n_paths, seed
    )
    y = math.exp(-r * t) * payoffs
    return PriceResult.from_samples(y, n_paths, time.perf_counter() - start, "plain-numba")


def heston_terminal_numba(
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
    seed: int = 0,
) -> np.ndarray:
    """Terminal Heston prices by full-truncation Euler."""
    return _heston_kernel(s0, v0, r, q, kappa, theta, xi, rho, t, n_steps, n_paths, seed)
