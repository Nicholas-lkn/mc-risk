import dataclasses
import math

import numpy as np
import pytest

from mc_risk import price_european
from mc_risk.analytic import bs_price
from mc_risk.models import gbm_paths


def test_gbm_shape_and_start():
    paths = gbm_paths(100.0, 0.03, 0.2, 1.0, n_steps=12, n_paths=1000, seed=1)
    assert paths.shape == (1000, 13)
    assert np.all(paths[:, 0] == 100.0)
    assert np.all(paths > 0)


def test_gbm_terminal_moments():
    s0, r, q, sigma, t = 100.0, 0.03, 0.01, 0.25, 2.0
    st = gbm_paths(s0, r, sigma, t, n_steps=4, n_paths=400_000, q=q, seed=2)[:, -1]
    se = st.std(ddof=1) / math.sqrt(st.size)
    assert abs(st.mean() - s0 * math.exp((r - q) * t)) < 2.576 * se
    log_var = np.log(st / s0).var(ddof=1)
    assert log_var == pytest.approx(sigma**2 * t, rel=0.01)


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("k", [80.0, 100.0, 120.0])
def test_european_inside_99ci_of_black_scholes(mkt, option, k):
    args = {**mkt, "k": k}
    res = price_european(**args, option=option, n_paths=200_000, seed=7)
    assert res.contains(bs_price(**args, option=option))


def test_black_scholes_put_call_parity(mkt):
    c, p = bs_price(**mkt, option="call"), bs_price(**mkt, option="put")
    fwd = mkt["s0"] * math.exp(-mkt["q"] * mkt["t"]) - mkt["k"] * math.exp(-mkt["r"] * mkt["t"])
    assert c - p == pytest.approx(fwd, abs=1e-12)


def test_monte_carlo_put_call_parity(mkt):
    c = price_european(**mkt, option="call", n_paths=200_000, seed=3)
    p = price_european(**mkt, option="put", n_paths=200_000, seed=3)
    fwd = mkt["s0"] * math.exp(-mkt["q"] * mkt["t"]) - mkt["k"] * math.exp(-mkt["r"] * mkt["t"])
    # Same seed, so C - P is the MC estimate of the discounted forward payoff.
    tol = 2.576 * math.hypot(c.std_error, p.std_error)
    assert abs((c.price - p.price) - fwd) < tol


def test_result_is_frozen_and_deterministic(mkt):
    a = price_european(**mkt, n_paths=10_000, seed=11)
    b = price_european(**mkt, n_paths=10_000, seed=11)
    assert a.price == b.price and a.std_error == b.std_error
    assert a.ci99[0] < a.price < a.ci99[1]
    assert a.n_paths == 10_000 and a.runtime > 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.price = 0.0


def test_standard_error_scales_as_inverse_sqrt_n(mkt):
    small = price_european(**mkt, n_paths=10_000, seed=5)
    large = price_european(**mkt, n_paths=160_000, seed=5)
    assert small.std_error / large.std_error == pytest.approx(4.0, rel=0.1)
