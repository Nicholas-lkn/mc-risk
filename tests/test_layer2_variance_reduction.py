import numpy as np
import pytest

from mc_risk import price_european, variance_reduction_factor
from mc_risk.analytic import bs_price
from mc_risk.engine import mc_price
from mc_risk.models import brownian_bridge, brownian_increments
from mc_risk.payoffs import european

N = 2**17


@pytest.mark.parametrize("method", ["antithetic", "control", "sobol"])
@pytest.mark.parametrize("option", ["call", "put"])
def test_methods_inside_99ci_of_black_scholes(mkt, method, option):
    res = price_european(**mkt, option=option, n_paths=N, method=method, seed=21)
    assert res.contains(bs_price(**mkt, option=option))


@pytest.mark.parametrize(
    "method,min_factor", [("antithetic", 1.2), ("control", 2.0), ("sobol", 50)]
)
def test_methods_reduce_variance(mkt, method, min_factor):
    plain = price_european(**mkt, n_paths=N, seed=4)
    other = price_european(**mkt, n_paths=N, method=method, seed=4)
    assert variance_reduction_factor(plain, other) > min_factor


def test_brownian_bridge_has_brownian_covariance():
    rng = np.random.default_rng(0)
    t, n = 2.0, 7  # not a power of two on purpose
    w = brownian_bridge(rng.standard_normal((200_000, n)), t)
    times = np.linspace(t / n, t, n)
    expected = np.minimum.outer(times, times)
    assert np.allclose(np.cov(w.T), expected, atol=0.02)


def test_bridge_and_increments_agree_in_law():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((100_000, 16))
    a, b = brownian_bridge(z, 1.0), brownian_increments(z, 1.0)
    assert a[:, -1].var() == pytest.approx(b[:, -1].var(), rel=0.03)
    assert a[:, 7].var() == pytest.approx(0.5, rel=0.03)


def test_sobol_with_bridge_prices_multistep_payoff(mkt):
    # A terminal payoff priced on a 32-step grid: dimension 32, same answer.
    res = mc_price(
        european(mkt["k"]),
        mkt["s0"],
        mkt["r"],
        mkt["sigma"],
        mkt["t"],
        q=mkt["q"],
        n_steps=32,
        n_paths=2**17,
        method="sobol",
        seed=9,
    )
    assert res.contains(bs_price(**mkt))
    assert res.method == "sobol"


def test_unknown_method_raises(mkt):
    with pytest.raises(ValueError):
        price_european(**mkt, method="magic")
