"""Cox-Ross-Rubinstein binomial tree, used as the reference for early exercise."""

from __future__ import annotations

import math

import numpy as np


def crr_price(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_steps: int = 2000,
    option: str = "put",
    american: bool = True,
    q: float = 0.0,
    exercise_every: int = 1,
) -> float:
    """CRR tree price of a European, American or Bermudan vanilla option.

    Args:
        n_steps: number of tree steps.
        american: allow early exercise.
        exercise_every: with ``american=True``, exercise is allowed only at tree
            steps that are multiples of this number. 1 gives an American option;
            n_steps / m gives a Bermudan option with m equally spaced dates,
            which is the right comparison for Longstaff-Schwartz with m dates.
    """
    sign = 1.0 if option == "call" else -1.0
    dt = t / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    p = (math.exp((r - q) * dt) - 1.0 / u) / (u - 1.0 / u)
    disc = math.exp(-r * dt)
    j = np.arange(n_steps + 1)  # number of down moves
    values = np.maximum(sign * (s0 * u ** (n_steps - 2.0 * j) - k), 0.0)
    for i in range(n_steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1.0 - p) * values[1:])
        if american and i % exercise_every == 0:
            spot = s0 * u ** (i - 2.0 * j[: i + 1])
            values = np.maximum(values, sign * (spot - k))
    return float(values[0])
