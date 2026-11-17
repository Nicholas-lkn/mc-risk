import pytest

from mc_risk import (
    price_american_lsm,
    price_asian,
    price_down_and_out_call,
    variance_reduction_factor,
)
from mc_risk.analytic import (
    bgk_adjusted_barrier,
    bs_price,
    crr_price,
    down_and_out_call,
    geometric_asian_discrete,
    geometric_asian_kemna_vorst,
)

ASIAN = {"s0": 100.0, "k": 100.0, "r": 0.03, "sigma": 0.2, "t": 1.0, "q": 0.01}
PUT = {"s0": 36.0, "k": 40.0, "r": 0.06, "sigma": 0.2, "t": 1.0}  # Longstaff-Schwartz Table 1


@pytest.mark.parametrize("option", ["call", "put"])
def test_geometric_asian_inside_99ci_of_closed_form(option):
    res = price_asian(**ASIAN, option=option, average="geometric", n_steps=50, seed=1)
    assert res.contains(geometric_asian_discrete(**ASIAN, n_steps=50, option=option))


def test_discrete_geometric_converges_to_kemna_vorst():
    discrete = geometric_asian_discrete(**ASIAN, n_steps=100_000)
    assert discrete == pytest.approx(geometric_asian_kemna_vorst(**ASIAN), rel=1e-4)


def test_arithmetic_asian_control_variate():
    plain = price_asian(**ASIAN, n_steps=50, control_variate=False, seed=2)
    cv = price_asian(**ASIAN, n_steps=50, seed=2)
    assert variance_reduction_factor(plain, cv) > 100
    # The two estimators agree, and the plain CI contains the far tighter CV price.
    assert plain.contains(cv.price)
    # AM-GM: the arithmetic average call is worth more than the geometric one.
    assert cv.ci99[0] > geometric_asian_discrete(**ASIAN, n_steps=50)
    assert cv.price < bs_price(**ASIAN)  # averaging lowers volatility


@pytest.mark.parametrize("n_monitor", [12, 52])
def test_barrier_inside_99ci_of_bgk_corrected_closed_form(n_monitor):
    s0, k, h, r, sigma, t, q = 100.0, 100.0, 90.0, 0.03, 0.2, 1.0, 0.01
    res = price_down_and_out_call(
        s0, k, h, r, sigma, t, q=q, n_monitor=n_monitor, n_paths=200_000, seed=3
    )
    shifted = bgk_adjusted_barrier(h, sigma, t, n_monitor)
    assert res.contains(down_and_out_call(s0, k, shifted, r, sigma, t, q))
    # Without the correction the continuous formula is too low for discrete monitoring.
    assert down_and_out_call(s0, k, h, r, sigma, t, q) < res.ci99[0]


def test_barrier_closed_form_limits():
    args = (100.0, 100.0)
    rest = (0.03, 0.2, 1.0, 0.01)
    assert down_and_out_call(*args, 1e-8, *rest) == pytest.approx(
        bs_price(*args, *rest[:3], q=0.01)
    )
    assert down_and_out_call(*args, 99.999999, *rest) == pytest.approx(0.0, abs=1e-4)
    assert down_and_out_call(*args, 95.0, *rest) < down_and_out_call(*args, 85.0, *rest)


def test_crr_matches_black_scholes_for_european():
    tree = crr_price(**PUT, n_steps=2000, american=False)
    assert tree == pytest.approx(bs_price(**PUT, option="put"), abs=2e-3)


def test_lsm_inside_99ci_of_binomial_tree():
    res = price_american_lsm(**PUT, n_steps=50, n_paths=100_000, seed=4)
    bermudan = crr_price(**PUT, n_steps=2000, exercise_every=40)  # same 50 exercise dates
    assert res.contains(bermudan)
    assert bermudan <= crr_price(**PUT, n_steps=2000)  # fewer rights, lower value


def test_lsm_american_at_least_european():
    res = price_american_lsm(**PUT, n_steps=50, n_paths=100_000, seed=5)
    assert res.ci99[0] > bs_price(**PUT, option="put")


def test_lsm_deep_in_the_money_returns_intrinsic():
    res = price_american_lsm(10.0, 40.0, 0.06, 0.2, 1.0, n_paths=20_000, seed=6)
    assert res.price == pytest.approx(30.0)
