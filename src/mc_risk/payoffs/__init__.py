"""Payoff functions. Each factory returns a callable mapping paths to payoffs.

Paths have shape (n_paths, n_steps + 1) with the spot in column 0, so the
monitoring dates of a path-dependent payoff are columns 1..n_steps.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Payoff = Callable[[np.ndarray], np.ndarray]


def _sign(option: str) -> float:
    if option not in ("call", "put"):
        raise ValueError(f"option must be 'call' or 'put', got {option!r}")
    return 1.0 if option == "call" else -1.0


def european(k: float, option: str = "call") -> Payoff:
    """Vanilla payoff on the terminal price."""
    sign = _sign(option)
    return lambda paths: np.maximum(sign * (paths[:, -1] - k), 0.0)


def digital(k: float, option: str = "call") -> Payoff:
    """Cash-or-nothing payoff of 1 on the terminal price."""
    sign = _sign(option)
    return lambda paths: (sign * (paths[:, -1] - k) > 0).astype(float)


def asian_arithmetic(k: float, option: str = "call") -> Payoff:
    """Payoff on the arithmetic average of the monitoring dates (columns 1..n)."""
    sign = _sign(option)
    return lambda paths: np.maximum(sign * (paths[:, 1:].mean(axis=1) - k), 0.0)


def asian_geometric(k: float, option: str = "call") -> Payoff:
    """Payoff on the geometric average of the monitoring dates (columns 1..n)."""
    sign = _sign(option)
    return lambda paths: np.maximum(sign * (np.exp(np.log(paths[:, 1:]).mean(axis=1)) - k), 0.0)


def down_and_out(k: float, barrier: float, option: str = "call") -> Payoff:
    """Vanilla payoff that is lost if any monitoring date is at or below the barrier."""
    sign = _sign(option)

    def payoff(paths: np.ndarray) -> np.ndarray:
        alive = paths[:, 1:].min(axis=1) > barrier
        return np.maximum(sign * (paths[:, -1] - k), 0.0) * alive

    return payoff


__all__ = [
    "Payoff",
    "asian_arithmetic",
    "asian_geometric",
    "digital",
    "down_and_out",
    "european",
]
