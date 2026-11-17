import math

import numpy as np
import pytest

pytest.importorskip("numba")

from mc_risk import price_asian  # noqa: E402
from mc_risk.analytic import heston_price  # noqa: E402
from mc_risk.fast import gbm_paths_numba, heston_terminal_numba, price_asian_numba  # noqa: E402
from mc_risk.models import gbm_paths  # noqa: E402


def test_numba_gbm_paths_identical_to_numpy():
    args = (100.0, 0.03, 0.2, 1.0, 12, 5000)
    assert np.allclose(gbm_paths_numba(*args, q=0.01, seed=3), gbm_paths(*args, q=0.01, seed=3))


def test_numba_asian_agrees_with_control_variate_price():
    args = {"s0": 100.0, "k": 100.0, "r": 0.03, "sigma": 0.2, "t": 1.0, "q": 0.01}
    fast = price_asian_numba(**args, n_steps=50, n_paths=200_000, seed=1)
    reference = price_asian(**args, n_steps=50, n_paths=200_000, seed=1)  # SE about 5e-4
    assert fast.contains(reference.price)
    assert price_asian_numba(**args, n_paths=1000, seed=7).price == pytest.approx(
        price_asian_numba(**args, n_paths=1000, seed=7).price
    )


def test_numba_heston_inside_99ci_of_semi_analytic():
    h = {"s0": 100.0, "v0": 0.04, "r": 0.03, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7}
    s_t = heston_terminal_numba(**h, t=1.0, n_steps=100, n_paths=200_000, seed=2)
    y = math.exp(-0.03) * np.maximum(s_t - 100.0, 0.0)
    se = y.std(ddof=1) / math.sqrt(y.size)
    assert abs(y.mean() - heston_price(k=100.0, t=1.0, **h)) < 2.576 * se
