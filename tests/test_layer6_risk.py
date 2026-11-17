import numpy as np
import pandas as pd
import pytest

from mc_risk.risk import (
    DEFAULT_WEIGHTS,
    PORTFOLIO,
    cad_returns,
    christoffersen_independence,
    clean_prices,
    conditional_coverage,
    fit_student_t_dof,
    fx_shock,
    gaussian_mc_var,
    gaussian_var_closed_form,
    historical_replay,
    historical_var,
    kupiec_pof,
    rolling_backtest,
    student_t_mc_var,
    student_t_var_closed_form,
    summarise_backtest,
    synthetic_prices,
    to_cad,
    var_cvar,
)

W = np.array([DEFAULT_WEIGHTS[t] for t in PORTFOLIO])


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return cad_returns(synthetic_prices(seed=0))


def test_weights_sum_to_one():
    assert W.sum() == pytest.approx(1.0)


def test_cad_conversion_and_calendar_alignment():
    prices = synthetic_prices(seed=1, n_days=300)
    prices.iloc[10, prices.columns.get_loc("AAPL")] = np.nan  # US holiday
    prices.iloc[20, :-1] = np.nan  # FX-only day: no equity traded
    clean = clean_prices(prices)
    assert len(clean) == 299 and not clean.isna().any().any()
    assert clean["AAPL"].iloc[10] == clean["AAPL"].iloc[9]
    cad = to_cad(clean)
    assert np.allclose(cad["AAPL"], clean["AAPL"] * clean["CAD=X"])
    assert np.allclose(cad["RY.TO"], clean["RY.TO"])


def test_var_cvar_on_known_sample():
    losses = np.arange(1, 101, dtype=float)  # 1..100
    var, cvar = var_cvar(losses, 0.95)
    assert var == pytest.approx(95.05)
    assert cvar == pytest.approx(np.mean([96, 97, 98, 99, 100]))
    assert cvar >= var


def test_gaussian_mc_matches_closed_form(returns):
    x = returns[PORTFOLIO].to_numpy()[-500:]
    mc = gaussian_mc_var(x, W, n_sims=400_000, seed=1)
    var, cvar = gaussian_var_closed_form(x, W)
    assert mc.var == pytest.approx(var, rel=0.02)
    assert mc.cvar == pytest.approx(cvar, rel=0.02)


def test_student_t_mc_matches_closed_form(returns):
    x = returns[PORTFOLIO].to_numpy()[-500:]
    mc = student_t_mc_var(x, W, n_sims=400_000, seed=2, dof=5.0)
    var, cvar = student_t_var_closed_form(x, W, dof=5.0)
    assert mc.var == pytest.approx(var, rel=0.03)
    assert mc.cvar == pytest.approx(cvar, rel=0.04)


def test_student_t_dof_fit_recovers_truth():
    rng = np.random.default_rng(3)
    nu, n, d = 5.0, 4000, 4
    z = rng.standard_normal((n, d)) * np.sqrt(nu / rng.chisquare(nu, size=n))[:, None]
    assert fit_student_t_dof(0.01 * z) == pytest.approx(nu, abs=0.7)
    assert fit_student_t_dof(0.01 * rng.standard_normal((n, d))) > 25


def test_student_t_tail_is_heavier_than_gaussian(returns):
    x = returns[PORTFOLIO].to_numpy()[-500:]
    g = gaussian_mc_var(x, W, seed=4)
    t = student_t_mc_var(x, W, seed=4, dof=4.0)
    assert t.cvar > g.cvar
    assert historical_var(x, W).cvar >= historical_var(x, W).var


def test_kupiec_accepts_correct_rate_and_rejects_wrong_rate():
    exact = np.zeros(1000, dtype=bool)
    exact[::100] = True  # exactly 1%
    lr, p = kupiec_pof(exact, 0.99)
    assert lr == pytest.approx(0.0, abs=1e-9) and p == pytest.approx(1.0)
    too_many = np.zeros(1000, dtype=bool)
    too_many[::20] = True  # 5%
    assert kupiec_pof(too_many, 0.99)[1] < 1e-6
    assert kupiec_pof(np.zeros(1000, dtype=bool), 0.99)[1] < 0.01  # zero exceedances in 1000


def test_christoffersen_detects_clustering():
    spread = np.zeros(1000, dtype=bool)
    spread[::100] = True
    clustered = np.zeros(1000, dtype=bool)
    clustered[500:510] = True
    assert christoffersen_independence(spread)[1] > 0.5
    assert christoffersen_independence(clustered)[1] < 1e-6
    assert conditional_coverage(clustered, 0.99)[1] < 1e-6


def test_rolling_backtest_has_no_lookahead_and_sane_coverage(returns):
    sub = returns[PORTFOLIO].iloc[:900]
    bt = rolling_backtest(sub, W, "historical", window=250)
    assert len(bt) == 650 and bt.index[0] == sub.index[250]
    # The first forecast only uses the first 250 days.
    first = historical_var(sub.to_numpy()[:250], W)
    assert bt["var"].iloc[0] == pytest.approx(first.var)
    summary = summarise_backtest(bt)
    assert summary["exceedances"] == int(bt["exceed"].sum())
    assert 0 < summary["rate"] < 0.05


@pytest.mark.parametrize("method", ["gaussian-mc", "student-t-mc"])
def test_rolling_backtest_monte_carlo_methods_run(returns, method):
    sub = returns[PORTFOLIO].iloc[-400:]
    bt = rolling_backtest(sub, W, method, window=250, n_sims=5000, seed=5)
    assert len(bt) == 150 and (bt["var"] > 0).all() and (bt["cvar"] >= bt["var"]).all()
    again = rolling_backtest(sub, W, method, window=250, n_sims=5000, seed=5)
    pd.testing.assert_frame_equal(bt, again)


def test_stress_scenarios(returns):
    us_weight = DEFAULT_WEIGHTS["AAPL"] + DEFAULT_WEIGHTS["MSFT"] + DEFAULT_WEIGHTS["JPM"]
    assert fx_shock(DEFAULT_WEIGHTS, -0.10) == pytest.approx(us_weight * (1 / 0.9 - 1))
    assert fx_shock(DEFAULT_WEIGHTS, 0.10) < 0  # stronger CAD hurts US holdings
    replay = historical_replay(returns, DEFAULT_WEIGHTS, "2020-02-20", "2020-03-23")
    assert replay["total_return"] < -0.10  # the synthetic data has a crash in this window
    assert replay["max_drawdown"] <= replay["total_return"] + 1e-12
    assert replay["worst_day"] < 0
