import pytest

from mc_risk.analytic import bs_delta, bs_vega, digital_delta, digital_vega
from mc_risk.greeks import mc_greek

ARGS = {"s0": 100.0, "k": 105.0, "r": 0.03, "sigma": 0.2, "t": 1.0}
Q = 0.01


def _analytic(greek, payoff):
    if payoff == "digital":
        return (digital_delta if greek == "delta" else digital_vega)(**ARGS, q=Q)
    if greek == "delta":
        return bs_delta(**ARGS, option=payoff, q=Q)
    return bs_vega(**ARGS, q=Q)


@pytest.mark.parametrize("greek", ["delta", "vega"])
@pytest.mark.parametrize("method", ["pathwise", "lr", "fd"])
@pytest.mark.parametrize("payoff", ["call", "put"])
def test_vanilla_greeks_inside_99ci_of_black_scholes(greek, method, payoff):
    res = mc_greek(greek, method, **ARGS, payoff=payoff, q=Q, n_paths=200_000, seed=31)
    assert res.contains(_analytic(greek, payoff))


@pytest.mark.parametrize("greek", ["delta", "vega"])
def test_digital_greeks_by_likelihood_ratio(greek):
    res = mc_greek(greek, "lr", **ARGS, payoff="digital", q=Q, n_paths=200_000, seed=32)
    assert res.contains(_analytic(greek, "digital"))


def test_pathwise_rejects_digital():
    with pytest.raises(ValueError, match="discontinuous"):
        mc_greek("delta", "pathwise", **ARGS, payoff="digital")


@pytest.mark.parametrize("greek", ["delta", "vega"])
def test_likelihood_ratio_has_higher_variance_than_pathwise(greek):
    pw = mc_greek(greek, "pathwise", **ARGS, q=Q, seed=33)
    lr = mc_greek(greek, "lr", **ARGS, q=Q, seed=33)
    assert lr.std_error > 2 * pw.std_error


def test_common_random_numbers_cut_finite_difference_variance():
    crn = mc_greek("delta", "fd", **ARGS, q=Q, seed=34)
    indep = mc_greek("delta", "fd", **ARGS, q=Q, seed=34, common_random_numbers=False)
    assert indep.std_error > 10 * crn.std_error


def test_finite_difference_on_digital_gets_noisier_as_bump_shrinks():
    wide = mc_greek("delta", "fd", **ARGS, payoff="digital", q=Q, bump=1.0, seed=35)
    narrow = mc_greek("delta", "fd", **ARGS, payoff="digital", q=Q, bump=0.05, seed=35)
    assert narrow.std_error > 3 * wide.std_error
