"""Streamlit front end for mc-risk.

Run from the repository root:
    uv run --extra app streamlit run src/mc_risk/app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from mc_risk.app.pricing import PAYOFFS, convergence, methods_for, price
from mc_risk.risk import (
    DEFAULT_WEIGHTS,
    PORTFOLIO,
    cad_returns,
    fx_shock,
    gaussian_mc_var,
    historical_replay,
    historical_var,
    load_prices,
    rolling_backtest,
    student_t_mc_var,
    summarise_backtest,
    synthetic_prices,
)

PRICE_FILE = Path("data") / "prices.csv"
BLUE, RED, GREY = "#1f5fa8", "#c0392b", "#6b7280"

st.set_page_config(page_title="mc-risk", layout="wide")
st.title("mc-risk")
st.caption("Monte Carlo option pricing and portfolio risk. Fixed seeds: same inputs, same output.")
pricer_tab, risk_tab = st.tabs(["Pricer", "Risk"])


# ----------------------------------------------------------------------------- pricer
with pricer_tab:
    left, right = st.columns([1, 2])
    with left:
        model = st.selectbox("Model", list(PAYOFFS))
        payoff = st.selectbox("Payoff", PAYOFFS[model])
        method = st.selectbox("Method", methods_for(model, payoff))
        n_paths = st.select_slider(
            "Paths", options=[2**p for p in range(12, 21)], value=2**16, format_func="{:,}".format
        )
        seed = st.number_input("Seed", min_value=0, value=0, step=1)
        c1, c2 = st.columns(2)
        p = {
            "s0": c1.number_input("Spot", value=100.0, min_value=1.0),
            "k": c2.number_input("Strike", value=100.0, min_value=1.0),
            "r": c1.number_input("Rate", value=0.03, format="%.4f"),
            "q": c2.number_input("Dividend yield", value=0.01, format="%.4f"),
            "t": c1.number_input("Maturity (years)", value=1.0, min_value=0.05),
        }
        if model != "Heston":
            p["sigma"] = c2.number_input("Volatility", value=0.20, min_value=0.01, format="%.4f")
        needs_steps = model == "Heston" or not payoff.startswith("European")
        if needs_steps:
            p["n_steps"] = int(st.number_input("Time steps / monitoring dates", 2, 500, 50))
        if payoff.startswith("Down"):
            p["barrier"] = st.number_input("Barrier", value=90.0, min_value=1.0)
        if model == "Heston":
            p["v0"] = c2.number_input("v0", value=0.04, min_value=0.0001, format="%.4f")
            p["kappa"] = c1.number_input("kappa", value=2.0, min_value=0.01)
            p["theta"] = c2.number_input("theta", value=0.04, min_value=0.0001, format="%.4f")
            p["xi"] = c1.number_input("xi (vol of vol)", value=0.3, min_value=0.01)
            p["rho"] = c2.number_input("rho", value=-0.7, min_value=-0.99, max_value=0.99)
        if model == "Merton":
            p["lam"] = c1.number_input("Jump intensity", value=0.5, min_value=0.0)
            p["mu_j"] = c2.number_input("Mean log jump", value=-0.10)
            p["delta"] = c1.number_input("Jump volatility", value=0.15, min_value=0.0)

    with right:
        res, ref, ref_name = price(model, payoff, method, p, int(n_paths), int(seed))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Price", f"{res.price:.4f}")
        m2.metric("Std error", f"{res.std_error:.5f}")
        m3.metric("99% CI", f"[{res.ci99[0]:.4f}, {res.ci99[1]:.4f}]")
        m4.metric("Runtime", f"{res.runtime * 1e3:.0f} ms")
        if ref is None:
            st.info(f"Reference: {ref_name}.")
        else:
            inside = res.contains(ref)
            text = f"Reference ({ref_name}): **{ref:.4f}**. Error {res.price - ref:+.4f}."
            (st.success if inside else st.warning)(
                text + (" Inside the 99% CI." if inside else " Outside the 99% CI.")
            )
        conv = convergence(model, payoff, method, p, int(n_paths), int(seed) + 1)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.fill_between(
            conv["n_paths"], conv["ci_low"], conv["ci_high"], color=BLUE, alpha=0.2, label="99% CI"
        )
        ax.plot(conv["n_paths"], conv["price"], "o-", color=BLUE, label="Estimate")
        if ref is not None:
            ax.axhline(ref, color=RED, lw=1, ls="--", label="Reference")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Paths")
        ax.set_ylabel("Price")
        ax.set_title("Convergence (independent runs)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"Estimator: {res.method}, {res.n_paths:,} paths.")


# ------------------------------------------------------------------------------- risk
@st.cache_data
def _returns() -> tuple[pd.DataFrame, str]:
    if PRICE_FILE.exists():
        return cad_returns(load_prices(PRICE_FILE)), "real"
    return cad_returns(synthetic_prices(seed=0)), "synthetic"


@st.cache_data
def _backtest(method: str, alpha: float, window: int, n_sims: int) -> pd.DataFrame:
    rets, _ = _returns()
    w = np.array([DEFAULT_WEIGHTS[t] for t in PORTFOLIO])
    return rolling_backtest(rets[PORTFOLIO], w, method, alpha, window, n_sims, seed=0)


with risk_tab:
    rets, source = _returns()
    if source == "synthetic":
        st.warning(
            "data/prices.csv not found: showing SYNTHETIC data. Run "
            "`uv run --extra data python scripts/download_data.py` for real prices."
        )
    w = np.array([DEFAULT_WEIGHTS[t] for t in PORTFOLIO])
    c1, c2, c3, c4 = st.columns(4)
    value = c1.number_input("Portfolio value (CAD)", value=1_000_000, step=100_000)
    alpha = c2.selectbox("Confidence", [0.99, 0.975, 0.95])
    window = c3.selectbox("Window (days)", [500, 250, 750])
    bt_method = c4.selectbox("Backtest method", ["historical", "gaussian-mc", "student-t-mc"])
    st.caption(
        "Weights: "
        + ", ".join(f"{t} {DEFAULT_WEIGHTS[t]:.0%}" for t in PORTFOLIO)
        + f". Data to {rets.index[-1].date()}."
    )

    x = rets[PORTFOLIO].to_numpy()[-window:]
    results = [
        historical_var(x, w, alpha),
        gaussian_mc_var(x, w, alpha, seed=0),
        student_t_mc_var(x, w, alpha, seed=0),
    ]
    st.subheader(f"1-day VaR and CVaR at {alpha:.1%}")
    st.dataframe(
        pd.DataFrame(
            {
                "Method": [r.method for r in results],
                "VaR": [f"{r.var:.2%}" for r in results],
                "CVaR": [f"{r.cvar:.2%}" for r in results],
                "VaR (CAD)": [f"{r.var * value:,.0f}" for r in results],
                "CVaR (CAD)": [f"{r.cvar * value:,.0f}" for r in results],
                "Fitted dof": ["", "", f"{results[2].dof:.2f}"],
            }
        ),
        hide_index=True,
    )

    st.subheader("Rolling backtest")
    bt = _backtest(bt_method, alpha, window, 5000)
    s = summarise_backtest(bt, alpha)
    b1, b2, b3, b4 = st.columns(4)
    b1.metric(
        "Exceedances",
        f"{s['exceedances']} / {s['days']}",
        f"expected {s['expected']:.1f}",
        delta_color="off",
    )
    b2.metric("Kupiec p-value", f"{s['kupiec_p']:.3f}")
    b3.metric("Independence p-value", f"{s['independence_p']:.3f}")
    b4.metric("Conditional coverage p-value", f"{s['cond_coverage_p']:.3f}")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(bt.index, -bt["loss"] * 100, width=1.0, color="#c7ccd4", label="Daily P&L")
    ax.plot(bt.index, -bt["var"] * 100, color=BLUE, lw=1.3, label=f"-VaR {alpha:.1%}")
    hits = bt[bt["exceed"]]
    ax.scatter(hits.index, -hits["loss"] * 100, color=RED, s=18, zorder=3, label="Exceedances")
    ax.set_ylabel("Return (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Stress scenarios")
    rows = []
    for label, start, end in (
        ("COVID crash replay (2020-02-20 to 2020-03-23)", "2020-02-20", "2020-03-23"),
        ("2022 sell-off replay (2022-01-03 to 2022-06-16)", "2022-01-03", "2022-06-16"),
    ):
        try:
            rows.append(
                (label, historical_replay(rets, DEFAULT_WEIGHTS, start, end)["total_return"])
            )
        except ValueError:
            pass
    rows.append(("CAD -10% against USD", fx_shock(DEFAULT_WEIGHTS, -0.10)))
    rows.append(("CAD +10% against USD", fx_shock(DEFAULT_WEIGHTS, 0.10)))
    st.dataframe(
        pd.DataFrame(
            {
                "Scenario": [r[0] for r in rows],
                "Portfolio return": [f"{r[1]:+.1%}" for r in rows],
                "P&L (CAD)": [f"{r[1] * value:+,.0f}" for r in rows],
            }
        ),
        hide_index=True,
    )
