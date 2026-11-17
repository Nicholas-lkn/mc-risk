"""Run every validation check, write plots to docs/img/ and tables to docs/validation.md.

Usage: uv run python scripts/validate.py

Every number in docs/validation.md and in the README results block is produced
here. The script exits with a non-zero status if any check fails.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from mc_risk import (
    mc_price,
    price_american_lsm,
    price_asian,
    price_down_and_out_call,
    price_european,
    price_european_heston,
    price_european_merton,
    variance_reduction_factor,
)
from mc_risk.analytic import (
    bgk_adjusted_barrier,
    bs_delta,
    bs_price,
    bs_vega,
    crr_price,
    digital_delta,
    down_and_out_call,
    geometric_asian_discrete,
    geometric_asian_kemna_vorst,
    heston_price,
    merton_price,
)
from mc_risk.greeks import mc_greek
from mc_risk.models import gbm_paths, heston_paths
from mc_risk.payoffs import european
from mc_risk.risk import (
    DEFAULT_WEIGHTS,
    FX,
    PORTFOLIO,
    cad_returns,
    factor_shock,
    fit_student_t_dof,
    fx_shock,
    gaussian_mc_var,
    gaussian_var_closed_form,
    historical_replay,
    historical_var,
    load_prices,
    rolling_backtest,
    student_t_mc_var,
    student_t_var_closed_form,
    summarise_backtest,
    synthetic_prices,
    var_cvar,
)

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
BASE = {"s0": 100.0, "k": 105.0, "r": 0.03, "sigma": 0.2, "t": 1.0, "q": 0.01}
BLUE, ORANGE, GREEN, GREY = "#1f5fa8", "#d9730d", "#2e8b57", "#6b7280"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)


class Report:
    """Collects markdown and pass/fail counts."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.passed = 0
        self.failed = 0
        self.summary: list[list[str]] = []  # rows for the README table

    def add(self, text: str = "") -> None:
        self.lines.append(text)

    def verdict(self, ok: bool) -> str:
        if ok:
            self.passed += 1
            return "PASS"
        self.failed += 1
        return "**FAIL**"

    def table(self, headers: list[str], rows: list[list]) -> None:
        self.add("| " + " | ".join(headers) + " |")
        self.add("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            self.add("| " + " | ".join(str(c) for c in row) + " |")
        self.add()

    def price_row(self, label: str, res, ref: float) -> list:
        """Standard row: estimate against a reference, pass if inside the 99% CI."""
        err = res.price - ref
        z = err / res.std_error if res.std_error > 0 else float("nan")
        return [
            label,
            f"{ref:.5f}",
            f"{res.price:.5f}",
            f"{err:+.5f}",
            f"{res.std_error:.5f}",
            f"[{res.ci99[0]:.5f}, {res.ci99[1]:.5f}]",
            f"{z:+.2f}",
            self.verdict(res.contains(ref)),
        ]


PRICE_HEADERS = ["Case", "Reference", "MC price", "Error", "Std error", "99% CI", "Error / SE", ""]


def savefig(fig, name: str) -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMG / name)
    plt.close(fig)


# --------------------------------------------------------------------------- layer 1
def layer1(rep: Report) -> None:
    rep.add("## 1. GBM paths and European options")
    rep.add()
    rep.add(
        "Checked against: Black-Scholes-Merton closed form. Base case "
        f"S0={BASE['s0']:g}, r={BASE['r']:g}, q={BASE['q']:g}, sigma={BASE['sigma']:g}, "
        f"T={BASE['t']:g}. 400,000 paths, plain Monte Carlo, seed 0."
    )
    rep.add()
    rows = []
    for option in ("call", "put"):
        for k in (80.0, 100.0, 105.0, 120.0):
            args = {**BASE, "k": k}
            res = price_european(**args, option=option, n_paths=400_000, seed=0)
            rows.append(rep.price_row(f"{option} K={k:g}", res, bs_price(**args, option=option)))
    rep.table(PRICE_HEADERS, rows)

    # Convergence: RMSE over independent seeds against N.
    ref = bs_price(**BASE)
    ns = [2**p for p in range(8, 21, 2)]
    n_seeds = 32
    rmse, mean_se = [], []
    for n in ns:
        runs = [price_european(**BASE, n_paths=n, seed=1000 + s) for s in range(n_seeds)]
        rmse.append(math.sqrt(np.mean([(x.price - ref) ** 2 for x in runs])))
        mean_se.append(np.mean([x.std_error for x in runs]))
    slope = np.polyfit(np.log(ns), np.log(rmse), 1)[0]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.loglog(ns, rmse, "o-", color=BLUE, label=f"RMSE over {n_seeds} seeds")
    ax.loglog(ns, mean_se, "s--", color=ORANGE, label="Mean reported std error", alpha=0.9)
    ax.loglog(
        ns,
        rmse[0] * np.sqrt(ns[0] / np.array(ns)),
        ":",
        color=GREY,
        label=r"$1/\sqrt{N}$ reference",
    )
    ax.set_xlabel("Number of paths N")
    ax.set_ylabel("Absolute pricing error")
    ax.set_title(f"European call: Monte Carlo convergence (fitted slope {slope:.3f})")
    ax.legend(frameon=False)
    savefig(fig, "convergence_european.png")

    rep.add("Convergence of the base-case call (K=105) against the number of paths:")
    rep.add()
    rep.add("![Convergence](img/convergence_european.png)")
    rep.add()
    ratio = rmse[-1] / mean_se[-1]
    rep.table(
        ["Check", "Value", "Expected", ""],
        [
            [
                "Log-log slope of RMSE vs N",
                f"{slope:.3f}",
                "-0.5 +/- 0.05",
                rep.verdict(abs(slope + 0.5) < 0.05),
            ],
            [
                f"RMSE / reported std error at N={ns[-1]:,}",
                f"{ratio:.2f}",
                "about 1 (0.7 to 1.4)",
                rep.verdict(0.7 < ratio < 1.4),
            ],
        ],
    )
    rep.add(
        "Limitations: GBM assumes constant volatility and no jumps. The log-Euler scheme is "
        "exact for GBM, so the only error here is sampling error."
    )
    rep.add()
    rep.summary.append(["European call/put (GBM)", "Black-Scholes", rows[2][3], rows[2][4], "8/8"])


# --------------------------------------------------------------------------- layer 2
def layer2(rep: Report) -> None:
    rep.add("## 2. Variance reduction")
    rep.add()
    n = 2**18
    rep.add(
        f"Checked against: Black-Scholes price of the base-case call (K=105). Budget {n:,} path "
        "evaluations per method, seed 0. Variance reduction factor (VRF) is the variance of plain "
        "Monte Carlo over the variance of the method at equal budget. Efficiency also accounts "
        "for runtime: VRF x (plain runtime / method runtime)."
    )
    rep.add()
    ref = bs_price(**BASE)
    labels = {
        "plain": "Plain",
        "antithetic": "Antithetic",
        "control": "Control variate (underlying)",
        "sobol": "Scrambled Sobol, 64 scrambles (RQMC)",
    }
    results = {}
    for method in labels:
        price_european(**BASE, n_paths=n, method=method, seed=0)  # warm-up for timing
        results[method] = price_european(**BASE, n_paths=n, method=method, seed=0)
    plain = results["plain"]
    rows = []
    for method, label in labels.items():
        res = results[method]
        vrf = variance_reduction_factor(plain, res)
        rows.append(
            [
                label,
                f"{res.price:.5f}",
                f"{res.price - ref:+.5f}",
                f"{res.std_error:.6f}",
                f"{vrf:,.1f}",
                f"{res.runtime * 1e3:.1f}",
                f"{vrf * plain.runtime / res.runtime:,.1f}",
                rep.verdict(res.contains(ref)),
            ]
        )
    rep.table(
        ["Method", "Price", "Error", "Std error", "VRF", "Runtime (ms)", "Efficiency", "In 99% CI"],
        rows,
    )
    rep.table(
        ["Check", "Value", "Expected", ""],
        [
            [
                f"VRF of {labels[m]}",
                f"{variance_reduction_factor(plain, results[m]):,.1f}",
                "> 1",
                rep.verdict(variance_reduction_factor(plain, results[m]) > 1),
            ]
            for m in ("antithetic", "control", "sobol")
        ],
    )

    # Brownian bridge: same terminal payoff priced on a 64-step grid (64 dimensions).
    rep.add(
        "Effect of the Brownian bridge. The same call is priced on a 64-step path, so the Sobol "
        "points are 64-dimensional. With the bridge, the first coordinate alone fixes the "
        "terminal value; with incremental construction the payoff depends on all 64."
    )
    rep.add()
    args = {k: v for k, v in BASE.items() if k != "k"}
    common = {"n_steps": 64, "n_paths": 2**16, "seed": 0}
    s0, r, sigma, t, q = args["s0"], args["r"], args["sigma"], args["t"], args["q"]
    pay = european(BASE["k"])
    runs = {
        "Plain MC": mc_price(pay, s0, r, sigma, t, q=q, **common),
        "Sobol, incremental": mc_price(
            pay, s0, r, sigma, t, q=q, method="sobol", bridge=False, **common
        ),
        "Sobol, Brownian bridge": mc_price(pay, s0, r, sigma, t, q=q, method="sobol", **common),
    }
    base = runs["Plain MC"]
    rep.table(
        ["Construction", "Price", "Std error", "VRF vs plain", "In 99% CI"],
        [
            [
                name,
                f"{res.price:.5f}",
                f"{res.std_error:.6f}",
                f"{variance_reduction_factor(base, res):,.1f}",
                rep.verdict(res.contains(ref)),
            ]
            for name, res in runs.items()
        ],
    )
    # Coverage of the RQMC interval: how often does the 99% CI contain the true price?
    rep.add(
        "Coverage of the RQMC confidence interval. The scramble means are not exactly normal, so "
        "the t interval built from a few scrambles can be too narrow. The table shows the share "
        "of 200 independent replications (seeds 0 to 199, 65,536 points each) whose 99% CI "
        "contains the Black-Scholes price. This is why the default is 64 scrambles, not 16."
    )
    rep.add()
    cover_rows = []
    for n_scr in (16, 32, 64):
        hits = [
            price_european(
                **BASE, n_paths=2**16, method="sobol", seed=s, n_scrambles=n_scr
            ).contains(ref)
            for s in range(200)
        ]
        cover = float(np.mean(hits))
        expected = ">= 0.97" if n_scr == 64 else "reported only"
        mark = rep.verdict(cover >= 0.97) if n_scr == 64 else ""
        cover_rows.append([n_scr, f"{cover:.3f}", expected, mark])
    rep.table(["Scrambles", "Empirical coverage of 99% CI", "Expected", ""], cover_rows)
    rep.add(
        "Limitations: the RQMC standard error comes from a limited number of scramble means and "
        "is itself noisy; coverage with few scrambles is below nominal (table above). The control "
        "variate coefficient is estimated from the same sample, which introduces an O(1/N) bias. "
        "Antithetic sampling helps monotone payoffs and can do nothing, or harm, for symmetric "
        "ones such as straddles. Runtimes are single runs on the machine that built the report."
    )
    rep.add()
    best = results["sobol"]
    rep.summary.append(
        [
            "Variance reduction (European call)",
            "Black-Scholes",
            f"{best.price - ref:+.5f}",
            f"{best.std_error:.6f}",
            f"Sobol VRF {variance_reduction_factor(plain, best):,.0f}x",
        ]
    )


# --------------------------------------------------------------------------- layer 3
def layer3(rep: Report) -> None:
    rep.add("## 3. Exotics: Asian, barrier, American")
    rep.add()
    asian = {"s0": 100.0, "k": 100.0, "r": 0.03, "sigma": 0.2, "t": 1.0, "q": 0.01}
    n_fix = 50

    rep.add("### 3.1 Asian options")
    rep.add()
    rep.add(
        f"Checked against: closed form for the discretely monitored geometric Asian ({n_fix} "
        "fixings), which is exact on the simulated grid. The Kemna-Vorst continuous formula is "
        "its limit and is shown for reference. S0=K=100, 200,000 paths."
    )
    rep.add()
    rows = []
    for option in ("call", "put"):
        res = price_asian(
            **asian, option=option, average="geometric", n_steps=n_fix, n_paths=200_000, seed=0
        )
        ref = geometric_asian_discrete(**asian, n_steps=n_fix, option=option)
        rows.append(rep.price_row(f"geometric {option}", res, ref))
    rep.table(PRICE_HEADERS, rows)
    kv = geometric_asian_kemna_vorst(**asian)
    disc_ref = geometric_asian_discrete(**asian, n_steps=n_fix)
    rep.add(
        f"Kemna-Vorst continuous call: {kv:.5f}; discrete call with {n_fix} fixings: "
        f"{disc_ref:.5f}. The gap is the effect of discrete monitoring, not an error."
    )
    rep.add()
    rep.add(
        "The arithmetic Asian has no closed form. It is priced with the geometric Asian as "
        "control variate; the check is that the independent plain estimate agrees."
    )
    rep.add()
    plain = price_asian(**asian, n_steps=n_fix, n_paths=200_000, control_variate=False, seed=0)
    cv = price_asian(**asian, n_steps=n_fix, n_paths=200_000, seed=1)
    sob = price_asian(**asian, n_steps=n_fix, n_paths=2**17, method="sobol", seed=2)
    arows = []
    for name, res in (
        ("Plain", plain),
        ("Geometric control variate", cv),
        ("Sobol + bridge + control variate", sob),
    ):
        arows.append(
            [
                name,
                f"{res.price:.5f}",
                f"{res.std_error:.6f}",
                f"[{res.ci99[0]:.5f}, {res.ci99[1]:.5f}]",
                f"{res.n_paths:,}",
                f"{variance_reduction_factor(plain, res):,.0f}",
                f"{res.runtime * 1e3:.0f}",
            ]
        )
    rep.table(
        ["Arithmetic call", "Price", "Std error", "99% CI", "Paths", "VRF", "Runtime (ms)"], arows
    )
    rep.table(
        ["Check", "Value", "Expected", ""],
        [
            [
                "Plain CI contains the control-variate price",
                f"{cv.price:.5f} in [{plain.ci99[0]:.5f}, {plain.ci99[1]:.5f}]",
                "yes",
                rep.verdict(plain.contains(cv.price)),
            ],
            [
                "Control-variate VRF",
                f"{variance_reduction_factor(plain, cv):,.0f}",
                "> 100",
                rep.verdict(variance_reduction_factor(plain, cv) > 100),
            ],
            [
                "Arithmetic price above geometric price (AM-GM)",
                f"{cv.price:.5f} > {disc_ref:.5f}",
                "yes",
                rep.verdict(cv.ci99[0] > disc_ref),
            ],
        ],
    )
    rep.summary.append(
        [
            "Arithmetic Asian call",
            "Geometric Asian closed form (control)",
            f"{rows[0][3]} (geometric)",
            f"{cv.std_error:.6f}",
            f"CV VRF {variance_reduction_factor(plain, cv):,.0f}x",
        ]
    )

    rep.add("### 3.2 Down-and-out barrier call")
    rep.add()
    s0, k, h, r, sigma, t, q = 100.0, 100.0, 90.0, 0.03, 0.2, 1.0, 0.01
    cont = down_and_out_call(s0, k, h, r, sigma, t, q)
    rep.add(
        "Checked against: Reiner-Rubinstein closed form for continuous monitoring, with the "
        "Broadie-Glasserman-Kou (BGK) barrier shift H exp(-0.5826 sigma sqrt(T/m)) for m "
        f"monitoring dates. S0=K=100, H=90. Continuous-monitoring price: {cont:.5f}. "
        "400,000 paths."
    )
    rep.add()
    brows = []
    for m in (12, 52, 252):
        res = price_down_and_out_call(s0, k, h, r, sigma, t, q=q, n_monitor=m, n_paths=400_000)
        ref = down_and_out_call(s0, k, bgk_adjusted_barrier(h, sigma, t, m), r, sigma, t, q)
        brows.append(
            [
                m,
                f"{res.price:.5f}",
                f"{res.std_error:.5f}",
                f"{res.price - cont:+.5f}",
                f"{ref:.5f}",
                f"{res.price - ref:+.5f}",
                f"{(res.price - ref) / res.std_error:+.2f}",
                rep.verdict(res.contains(ref)),
            ]
        )
    rep.table(
        [
            "Monitoring dates m",
            "MC price",
            "Std error",
            "MC - continuous formula",
            "BGK-corrected formula",
            "MC - BGK",
            "Error / SE",
            "In 99% CI",
        ],
        brows,
    )
    rep.add(
        "The uncorrected continuous formula is biased low by many standard errors even with "
        "daily monitoring; the BGK shift removes the bias. Limitations: the correction is "
        "asymptotic in m and degrades when the barrier is very close to the spot or m is small. "
        "Only the down-and-out call with H <= K has a reference here."
    )
    rep.add()
    rep.summary.append(
        ["Down-and-out call (m=52)", "Closed form + BGK", brows[1][5], brows[1][2], brows[1][7]]
    )

    rep.add("### 3.3 American put (Longstaff-Schwartz)")
    rep.add()
    rep.add(
        "Checked against: CRR binomial tree with 2,000 steps. LSM with 50 exercise dates prices "
        "a Bermudan option, so the like-for-like reference is the tree with exercise allowed at "
        "the same 50 dates. The fully American tree value is shown as well. K=40, r=0.06, T=1 "
        "(cases from Longstaff and Schwartz, 2001, Table 1). 200,000 paths, cubic polynomial, "
        "in-the-money paths only."
    )
    rep.add()
    lrows = []
    worst = None
    for spot, vol in ((36.0, 0.2), (36.0, 0.4), (40.0, 0.2), (44.0, 0.2)):
        args = {"s0": spot, "k": 40.0, "r": 0.06, "sigma": vol, "t": 1.0}
        res = price_american_lsm(**args, n_steps=50, n_paths=200_000, seed=0)
        berm = crr_price(**args, n_steps=2000, exercise_every=40)
        amer = crr_price(**args, n_steps=2000)
        euro = bs_price(**args, option="put")
        lrows.append(
            [
                f"S0={spot:g}, sigma={vol:g}",
                f"{euro:.4f}",
                f"{amer:.4f}",
                f"{berm:.4f}",
                f"{res.price:.4f}",
                f"{res.price - berm:+.4f}",
                f"{res.std_error:.4f}",
                rep.verdict(res.contains(berm)),
                rep.verdict(res.ci99[0] > euro),
            ]
        )
        if worst is None:
            worst = lrows[-1]
    rep.table(
        [
            "Case",
            "European (BS)",
            "CRR American",
            "CRR Bermudan-50",
            "LSM",
            "LSM - Bermudan",
            "Std error",
            "In 99% CI",
            "LSM > European",
        ],
        lrows,
    )
    rep.add(
        "Limitations: LSM has two opposing biases. The fitted exercise rule is suboptimal (low "
        "bias), and the same paths are used to fit and to value (high bias, small at this path "
        "count). The standard error ignores the dependence created by the shared regression. A "
        "separate out-of-sample valuation pass would give a clean lower bound. Polynomials in "
        "S/K are adequate for one underlying; they do not scale to baskets."
    )
    rep.add()
    rep.summary.append(
        ["American put (LSM, S0=36)", "CRR tree, 2,000 steps", worst[5], worst[6], worst[7]]
    )


# --------------------------------------------------------------------------- layer 4
def layer4(rep: Report) -> None:
    rep.add("## 4. Models: Heston and Merton")
    rep.add()
    heston = {
        "s0": 100.0,
        "v0": 0.04,
        "r": 0.03,
        "kappa": 2.0,
        "theta": 0.04,
        "xi": 0.3,
        "rho": -0.7,
        "t": 1.0,
    }
    rep.add("### 4.1 Heston, full-truncation Euler")
    rep.add()
    rep.add(
        "Checked against: semi-analytic price from the characteristic function (Albrecher et "
        "al. formulation, adaptive quadrature). Parameters v0=theta=0.04, kappa=2, xi=0.3, "
        "rho=-0.7, r=0.03, T=1 (Feller condition holds: 2 kappa theta = 0.16 > xi^2 = 0.09). "
        "200 time steps, 400,000 paths."
    )
    rep.add()
    rows = []
    for option in ("call", "put"):
        for k in (80.0, 100.0, 120.0):
            res = price_european_heston(
                k=k, **heston, option=option, n_steps=200, n_paths=400_000, seed=0
            )
            rows.append(
                rep.price_row(f"{option} K={k:g}", res, heston_price(k=k, **heston, option=option))
            )
    rep.table(PRICE_HEADERS, rows)
    flat = bs_price(100.0, 100.0, 0.03, 0.2, 1.0)
    limit = heston_price(k=100.0, **{**heston, "xi": 0.01, "rho": 0.0})
    rep.table(
        ["Check on the reference itself", "Value", "Expected", ""],
        [
            [
                "Heston with xi=0.01, rho=0 vs Black-Scholes (sigma=0.2)",
                f"{limit:.6f} vs {flat:.6f}",
                "difference < 1e-3",
                rep.verdict(abs(limit - flat) < 1e-3),
            ]
        ],
    )

    # Discretisation bias against the number of time steps, in a harder regime.
    hard = {**heston, "kappa": 1.5, "xi": 0.6}
    ref = heston_price(k=100.0, **hard)
    steps = [2, 4, 8, 16, 32, 64, 128]
    runs = [
        price_european_heston(k=100.0, **hard, n_steps=n, n_paths=1_000_000, seed=100 + n)
        for n in steps
    ]
    bias = np.array([x.price - ref for x in runs])
    half = np.array([x.ci99[1] - x.price for x in runs])
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.errorbar(steps, bias, yerr=half, fmt="o-", color=BLUE, capsize=3, label="MC - semi-analytic")
    ax.axhline(0.0, color=GREY, lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Time steps per year")
    ax.set_ylabel("Price error (bars: 99% CI of sampling error)")
    ax.set_title("Heston full-truncation Euler: discretisation bias")
    ax.legend(frameon=False)
    savefig(fig, "heston_bias.png")
    rep.add(
        "Discretisation bias. A harder case where the Feller condition fails (kappa=1.5, "
        f"xi=0.6, so 2 kappa theta = 0.12 < xi^2 = 0.36), ATM call, reference {ref:.5f}, "
        "1,000,000 paths per point so the sampling error is small next to the bias at coarse "
        "grids."
    )
    rep.add()
    rep.add("![Heston bias](img/heston_bias.png)")
    rep.add()
    rep.table(
        ["Time steps", "MC price", "Bias (MC - reference)", "Std error", "Bias / SE"],
        [
            [n, f"{x.price:.5f}", f"{b:+.5f}", f"{x.std_error:.5f}", f"{b / x.std_error:+.1f}"]
            for n, x, b in zip(steps, runs, bias, strict=True)
        ],
    )
    rep.table(
        ["Check", "Value", "Expected", ""],
        [
            [
                "Bias at 128 steps inside 99% CI",
                f"{bias[-1]:+.5f}",
                "yes",
                rep.verdict(runs[-1].contains(ref)),
            ],
            [
                "Bias at 2 steps is detectable (scheme is not exact)",
                f"{bias[0] / runs[0].std_error:+.1f} SE",
                "abs(bias) > 3 SE",
                rep.verdict(abs(bias[0]) > 3 * runs[0].std_error),
            ],
        ],
    )
    rep.add(
        "Limitations: the confidence interval covers sampling error only, so a coarse grid can "
        "give a tight interval around the wrong number. Full truncation is first-order at best "
        "and slows down when the Feller condition fails; the quadratic-exponential scheme of "
        "Andersen (2008) or exact simulation (Broadie and Kaya, 2006) would be the next step. "
        "Parameters are illustrative, not calibrated to a market smile."
    )
    rep.add()
    rep.summary.append(
        ["European (Heston, 200 steps)", "Semi-analytic (CF)", rows[1][3], rows[1][4], "6/6"]
    )

    rep.add("### 4.2 Merton jump diffusion")
    rep.add()
    merton = {
        "s0": 100.0,
        "r": 0.03,
        "sigma": 0.15,
        "lam": 0.5,
        "mu_j": -0.1,
        "delta": 0.15,
        "t": 1.0,
    }
    rep.add(
        "Checked against: Merton's series of Black-Scholes prices (60 terms). sigma=0.15, "
        "lambda=0.5 jumps per year, log jump size N(-0.10, 0.15^2). The terminal price is "
        "simulated exactly, so there is no discretisation bias. 400,000 paths."
    )
    rep.add()
    mrows = []
    for option in ("call", "put"):
        for k in (80.0, 100.0, 120.0):
            res = price_european_merton(k=k, **merton, option=option, n_paths=400_000, seed=0)
            mrows.append(
                rep.price_row(f"{option} K={k:g}", res, merton_price(k=k, **merton, option=option))
            )
    rep.table(PRICE_HEADERS, mrows)
    no_jump = merton_price(k=100.0, **{**merton, "lam": 0.0})
    bs_ref = bs_price(100.0, 100.0, 0.03, 0.15, 1.0)
    rep.table(
        ["Check on the reference itself", "Value", "Expected", ""],
        [
            [
                "Series with lambda=0 vs Black-Scholes",
                f"{no_jump:.8f} vs {bs_ref:.8f}",
                "equal to 1e-10",
                rep.verdict(abs(no_jump - bs_ref) < 1e-10),
            ]
        ],
    )
    rep.add(
        "Limitations: jump sizes are lognormal and independent of the diffusion, with constant "
        "intensity. Jump payoffs are heavy-tailed, so standard errors for far out-of-the-money "
        "options converge slowly."
    )
    rep.add()
    rep.summary.append(
        ["European (Merton jumps)", "Merton series", mrows[1][3], mrows[1][4], "6/6"]
    )


# --------------------------------------------------------------------------- layer 5
def layer5(rep: Report) -> None:
    rep.add("## 5. Greeks")
    rep.add()
    args = {"s0": 100.0, "k": 105.0, "r": 0.03, "sigma": 0.2, "t": 1.0}
    q = 0.01
    n = 400_000
    rep.add(
        "Checked against: Black-Scholes delta and vega. European call, S0=100, K=105, "
        f"sigma=0.2, T=1. {n:,} paths, seed 0. Variance ratio is relative to the pathwise "
        "estimator. Cost is the number of payoff evaluations per path."
    )
    rep.add()
    labels = {"pathwise": "Pathwise", "lr": "Likelihood ratio", "fd": "Finite difference (CRN)"}
    cost = {"pathwise": 1, "lr": 1, "fd": 2}
    rows = []
    for greek, ref in (("delta", bs_delta(**args, q=q)), ("vega", bs_vega(**args, q=q))):
        base = None
        for method, label in labels.items():
            mc_greek(greek, method, **args, q=q, n_paths=n, seed=0)  # warm-up for timing
            res = mc_greek(greek, method, **args, q=q, n_paths=n, seed=0)
            base = base or res
            rows.append(
                [
                    greek,
                    label,
                    f"{ref:.5f}",
                    f"{res.price:.5f}",
                    f"{res.price - ref:+.5f}",
                    f"{res.std_error:.5f}",
                    f"{(res.std_error / base.std_error) ** 2:.1f}",
                    cost[method],
                    f"{res.runtime * 1e3:.1f}",
                    rep.verdict(res.contains(ref)),
                ]
            )
    rep.table(
        [
            "Greek",
            "Method",
            "Analytic",
            "Estimate",
            "Error",
            "Std error",
            "Variance ratio",
            "Cost",
            "Runtime (ms)",
            "In 99% CI",
        ],
        rows,
    )

    # Finite-difference bias against bump size (exact, from the closed form).
    rep.add(
        "Bias. Pathwise and likelihood-ratio estimators are unbiased. The central difference "
        "has bias of order h^2, which can be computed exactly from the closed form: "
        "[BS(S0+h) - BS(S0-h)] / 2h - delta."
    )
    rep.add()
    delta = bs_delta(**args, q=q)
    brows = []
    for h in (10.0, 5.0, 1.0, 0.5):
        up = bs_price(**{**args, "s0": args["s0"] + h}, q=q)
        down = bs_price(**{**args, "s0": args["s0"] - h}, q=q)
        exact_bias = (up - down) / (2 * h) - delta
        res = mc_greek("delta", "fd", **args, q=q, n_paths=n, seed=1, bump=h)
        brows.append(
            [
                f"{h:g}",
                f"{exact_bias:+.6f}",
                f"{res.price - delta:+.6f}",
                f"{res.std_error:.6f}",
                f"{exact_bias / res.std_error:+.2f}",
            ]
        )
    rep.table(["Bump h", "Exact FD bias", "MC error", "Std error", "Bias / SE"], brows)

    # Digital payoff: where the methods separate.
    rep.add(
        "Discontinuous payoff. For a cash-or-nothing call the pathwise derivative is zero "
        "almost everywhere, so the estimator is not valid (the library raises). The likelihood "
        "ratio still works. Finite differences work but the variance grows like 1/h as the bump "
        "shrinks."
    )
    rep.add()
    dref = digital_delta(**args, q=q)
    drows = []
    lr = mc_greek("delta", "lr", **args, payoff="digital", q=q, n_paths=n, seed=2)
    drows.append(
        [
            "Likelihood ratio",
            f"{dref:.6f}",
            f"{lr.price:.6f}",
            f"{lr.std_error:.6f}",
            rep.verdict(lr.contains(dref)),
        ]
    )
    for h in (2.0, 0.5, 0.1):
        res = mc_greek("delta", "fd", **args, payoff="digital", q=q, n_paths=n, seed=2, bump=h)
        drows.append(
            [
                f"Finite difference, h={h:g}",
                f"{dref:.6f}",
                f"{res.price:.6f}",
                f"{res.std_error:.6f}",
                "",
            ]
        )
    drows.append(["Pathwise", f"{dref:.6f}", "not applicable", "", ""])
    rep.table(["Digital delta", "Analytic", "Estimate", "Std error", "In 99% CI"], drows)

    # Standard error against maturity: LR degrades for short maturities.
    mats = [1 / 52, 1 / 12, 0.25, 0.5, 1.0, 2.0]
    ses = {m: [] for m in labels}
    for t in mats:
        for method in labels:
            a = {**args, "t": t, "k": 100.0}
            ses[method].append(
                mc_greek("delta", method, **a, q=q, n_paths=100_000, seed=3).std_error
            )
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for (method, label), colour, marker in zip(
        labels.items(), (BLUE, ORANGE, GREEN), ("o", "s", "^"), strict=True
    ):
        ax.loglog(mats, ses[method], marker + "-", color=colour, label=label)
    ax.set_xlabel("Maturity T (years)")
    ax.set_ylabel("Std error of delta (100,000 paths)")
    ax.set_title("ATM call delta: estimator noise against maturity")
    ax.legend(frameon=False)
    savefig(fig, "greeks_se_vs_maturity.png")
    rep.add("![Greeks standard error](img/greeks_se_vs_maturity.png)")
    rep.add()
    ratios = np.array(ses["lr"]) / np.array(ses["pathwise"])
    rep.table(
        ["Check", "Value", "Expected", ""],
        [
            [
                "LR / pathwise std error across maturities (1 week to 2 years)",
                f"{ratios.min():.1f} to {ratios.max():.1f}",
                "> 1 at every maturity",
                rep.verdict(bool((ratios > 1).all())),
            ],
            [
                "FD (CRN) / pathwise std error across maturities",
                f"{min(np.array(ses['fd']) / np.array(ses['pathwise'])):.2f} to "
                f"{max(np.array(ses['fd']) / np.array(ses['pathwise'])):.2f}",
                "about 1",
                rep.verdict(
                    bool(np.allclose(np.array(ses["fd"]) / np.array(ses["pathwise"]), 1, atol=0.1))
                ),
            ],
        ],
    )
    rep.add(
        "For a vanilla payoff the likelihood-ratio penalty is a roughly constant factor: the "
        "score scales like 1/(sigma sqrt(T)) but the payoff itself scales like sigma sqrt(T). "
        "The penalty becomes severe when the payoff does not shrink with maturity (digitals) or "
        "when the path has many steps (the delta score depends on the first step only, 1/sqrt(dt))."
    )
    rep.add()
    rep.add(
        "Limitations: implemented for terminal payoffs under GBM only. Pathwise needs a "
        "continuous payoff; likelihood ratio needs the transition density and is noisier; "
        "finite differences need a bump chosen to "
        "balance bias against variance, and cost two valuations per Greek. Gamma is not "
        "covered (pathwise fails for it on vanillas; the usual fix is a mixed estimator)."
    )
    rep.add()
    rep.summary.append(
        ["Delta and vega, 3 estimators", "Black-Scholes Greeks", rows[0][4], rows[0][5], "6/6"]
    )


# --------------------------------------------------------------------------- layer 6
PORTFOLIO_VALUE = 1_000_000.0  # CAD
PRICE_FILE = ROOT / "data" / "prices.csv"


def load_risk_data() -> tuple[pd.DataFrame, str]:
    """Real prices if data/prices.csv exists, otherwise the synthetic generator."""
    if PRICE_FILE.exists():
        prices = load_prices(PRICE_FILE)
        return prices, "real"
    return synthetic_prices(seed=0), "synthetic"


def layer6(rep: Report) -> None:
    rep.add("## 6. Portfolio risk (base currency CAD)")
    rep.add()
    prices, source = load_risk_data()
    rets = cad_returns(prices)
    w = np.array([DEFAULT_WEIGHTS[t] for t in PORTFOLIO])
    port = rets[PORTFOLIO].to_numpy() @ w
    first, last = rets.index[0].date(), rets.index[-1].date()
    if source == "real":
        rep.add(
            f"Data: daily adjusted closes from Yahoo Finance (`data/prices.csv`, not committed), "
            f"{len(rets):,} return days from {first} to {last} after calendar alignment. US names "
            "and SPY are converted to CAD with USDCAD (`CAD=X`)."
        )
    else:
        rep.add(
            "**Data: SYNTHETIC.** `data/prices.csv` was not found, so this section was built on "
            "`mc_risk.risk.synthetic_prices`. The numbers below are not market results. Run "
            "`uv run --extra data python scripts/download_data.py` and rerun this script."
        )
    rep.add()
    weights_text = ", ".join(f"{t} {DEFAULT_WEIGHTS[t]:.0%}" for t in PORTFOLIO)
    rep.add(
        f"Portfolio: CAD {PORTFOLIO_VALUE:,.0f}, weights by market value: {weights_text}. "
        "Weights are held constant (daily rebalancing). All risk numbers are 1-day, 99%."
    )
    rep.add()

    ann = np.sqrt(252)
    series = {
        "Portfolio": port,
        "XIU.TO (benchmark)": rets["XIU.TO"].to_numpy(),
        "SPY in CAD (benchmark)": rets["SPY"].to_numpy(),
        "USDCAD": rets[FX].to_numpy(),
    }
    win = 500
    srows = []
    for name, r in series.items():
        v, c = var_cvar(-r[-win:], 0.99)
        beta = np.cov(port, r)[0, 1] / r.var(ddof=1)
        srows.append(
            [
                name,
                f"{r.std(ddof=1) * ann:.1%}",
                f"{v:.2%}",
                f"{c:.2%}",
                f"{np.corrcoef(port, r)[0, 1]:.2f}",
                "" if name == "Portfolio" else f"{beta:.2f}",
            ]
        )
    rep.table(
        [
            "Series",
            "Annualised vol (full sample)",
            f"Hist. VaR (last {win} days)",
            f"Hist. CVaR (last {win} days)",
            "Correlation with portfolio",
            "Portfolio beta to it",
        ],
        srows,
    )

    rep.add(f"### 6.1 VaR and CVaR on the latest {win}-day window")
    rep.add()
    x = rets[PORTFOLIO].to_numpy()[-win:]
    nu = fit_student_t_dof(x)
    results = [
        historical_var(x, w),
        gaussian_mc_var(x, w, n_sims=400_000, seed=0),
        student_t_mc_var(x, w, n_sims=400_000, seed=0, dof=nu),
    ]
    names = {
        "historical": "Historical simulation",
        "gaussian-mc": "Gaussian Monte Carlo",
        "student-t-mc": f"Student-t Monte Carlo (fitted nu = {nu:.2f})",
    }
    rep.table(
        ["Method", "VaR", "CVaR", "VaR (CAD)", "CVaR (CAD)", "CVaR / VaR", "Scenarios"],
        [
            [
                names[r.method],
                f"{r.var:.3%}",
                f"{r.cvar:.3%}",
                f"{r.var * PORTFOLIO_VALUE:,.0f}",
                f"{r.cvar * PORTFOLIO_VALUE:,.0f}",
                f"{r.cvar / r.var:.2f}",
                f"{r.n_scenarios:,}",
            ]
            for r in results
        ],
    )
    g_var, g_cvar = gaussian_var_closed_form(x, w)
    t_var, t_cvar = student_t_var_closed_form(x, w, nu)
    rep.add(
        "Checked against: for a linear portfolio both parametric models have closed-form VaR "
        "and CVaR, so the Monte Carlo engines can be verified exactly (tolerance 2% relative, "
        "400,000 scenarios)."
    )
    rep.add()
    checks = [
        ("Gaussian MC VaR", results[1].var, g_var),
        ("Gaussian MC CVaR", results[1].cvar, g_cvar),
        ("Student-t MC VaR", results[2].var, t_var),
        ("Student-t MC CVaR", results[2].cvar, t_cvar),
    ]
    rep.table(
        ["Quantity", "Monte Carlo", "Closed form", "Relative error", ""],
        [
            [
                n,
                f"{mc:.4%}",
                f"{cf:.4%}",
                f"{mc / cf - 1:+.2%}",
                rep.verdict(abs(mc / cf - 1) < 0.02),
            ]
            for n, mc, cf in checks
        ],
    )

    rep.add("### 6.2 Rolling backtest")
    rep.add()
    rep.add(
        f"Each day's VaR is estimated from the previous {win} days only and compared with the "
        "loss realised that day. Monte Carlo methods use 20,000 scenarios per day; the Student-t "
        "degrees of freedom are refitted every 21 days. Kupiec tests the exceedance rate, "
        "Christoffersen tests that exceedances do not cluster, and conditional coverage tests "
        "both. A p-value below 0.05 rejects the model."
    )
    rep.add()
    bts = {}
    brows = []
    for method in ("historical", "gaussian-mc", "student-t-mc"):
        bt = rolling_backtest(rets[PORTFOLIO], w, method, window=win, n_sims=20_000, seed=0)
        bts[method] = bt
        s = summarise_backtest(bt)
        verdict = "not rejected" if min(s["kupiec_p"], s["cond_coverage_p"]) >= 0.05 else "rejected"
        brows.append(
            [
                names[method].split(" (")[0],
                s["days"],
                s["exceedances"],
                f"{s['expected']:.1f}",
                f"{s['rate']:.2%}",
                f"{s['kupiec_p']:.3f}",
                f"{s['independence_p']:.3f}",
                f"{s['cond_coverage_p']:.3f}",
                f"{s['mean_exceed_ratio']:.2f}",
                verdict,
            ]
        )
    rep.table(
        [
            "Method",
            "Days",
            "Exceedances",
            "Expected",
            "Rate",
            "Kupiec p",
            "Independence p",
            "Cond. coverage p",
            "Mean loss / VaR on exceedance",
            "At 5%",
        ],
        brows,
    )
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
    for ax, (method, bt), colour in zip(axes, bts.items(), (BLUE, ORANGE, GREEN), strict=True):
        ax.bar(bt.index, -bt["loss"] * 100, width=1.0, color="#c7ccd4", label="Daily P&L")
        ax.plot(bt.index, -bt["var"] * 100, color=colour, lw=1.3, label="-VaR 99%")
        hits = bt[bt["exceed"]]
        ax.scatter(
            hits.index,
            -hits["loss"] * 100,
            color="#c0392b",
            s=18,
            zorder=3,
            label=f"Exceedances ({len(hits)})",
        )
        ax.set_title(names[method].split(" (")[0], loc="left", fontsize=10)
        ax.set_ylabel("Return (%)")
        ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower left")
    tag = "" if source == "real" else " [SYNTHETIC DATA]"
    fig.suptitle(f"Rolling 1-day 99% VaR backtest, {win}-day window{tag}", fontsize=11)
    savefig(fig, "var_backtest.png")
    rep.add("![VaR backtest](img/var_backtest.png)")
    rep.add()
    rep.add(
        "Backtest outcomes are findings about the models, not pass/fail checks on the code, so "
        "they are not counted in the totals at the top of this report."
    )
    rep.add()

    rep.add("### 6.3 Stress scenarios")
    rep.add()
    stress_rows = []
    for label, start, end in (
        ("COVID crash replay (2020-02-20 to 2020-03-23)", "2020-02-20", "2020-03-23"),
        ("2022 rate-hike sell-off replay (2022-01-03 to 2022-06-16)", "2022-01-03", "2022-06-16"),
    ):
        try:
            r = historical_replay(rets, DEFAULT_WEIGHTS, start, end)
        except ValueError:
            continue
        stress_rows.append(
            [
                label,
                f"{r['total_return']:+.1%}",
                f"{r['total_return'] * PORTFOLIO_VALUE:+,.0f}",
                f"worst day {r['worst_day']:+.1%} on {r['worst_day_date']}; "
                f"max drawdown {r['max_drawdown']:.1%}",
            ]
        )
    for label, change in (("CAD -10% against USD", -0.10), ("CAD +10% against USD", 0.10)):
        r = fx_shock(DEFAULT_WEIGHTS, change)
        stress_rows.append(
            [
                label,
                f"{r:+.1%}",
                f"{r * PORTFOLIO_VALUE:+,.0f}",
                "translation effect on US names only",
            ]
        )
    combo = factor_shock(DEFAULT_WEIGHTS, dict.fromkeys(PORTFOLIO, -0.20)) + fx_shock(
        DEFAULT_WEIGHTS, -0.05
    ) * (1 - 0.20)
    stress_rows.append(
        [
            "Equities -20% in local currency with CAD -5%",
            f"{combo:+.1%}",
            f"{combo * PORTFOLIO_VALUE:+,.0f}",
            "weaker CAD cushions the US holdings",
        ]
    )
    rep.table(["Scenario", "Portfolio return", "P&L (CAD)", "Detail"], stress_rows)
    rep.add(
        "Note on the FX scenarios: the portfolio is long USD through its US holdings (50% of "
        "value), so a weaker Canadian dollar is a gain and a stronger one is the loss scenario."
    )
    rep.add()
    rep.add(
        "Limitations: linear positions only (no options in the risk portfolio); constant "
        "weights; historical simulation and both parametric fits weight the whole window "
        "equally, so they react slowly when volatility jumps (no GARCH or EWMA filtering); the "
        "Student-t model uses one degrees-of-freedom parameter for all names; Toronto and New "
        "York closes are treated as simultaneous and single-market holidays are forward-filled; "
        "about 12 expected exceedances gives the coverage tests limited power; adjusted closes "
        "from Yahoo are not an audited data source."
    )
    rep.add()
    hist_s = summarise_backtest(bts["historical"])
    rep.summary.append(
        [
            f"Portfolio VaR, {source} data",
            "Closed-form normal / Student-t VaR",
            f"{results[1].var / g_var - 1:+.2%} (Gaussian MC)",
            "n/a",
            f"hist. backtest {hist_s['exceedances']} exceedances vs {hist_s['expected']:.1f} expected",
        ]
    )


# --------------------------------------------------------------------------- layer 7
def _best_of(fn, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def layer7(rep: Report) -> None:
    rep.add("## 7. Numba benchmark")
    rep.add()
    try:
        import numba

        from mc_risk import fast
    except ImportError:
        rep.add("Numba is not installed (`uv sync --extra fast`); benchmark skipped.")
        rep.add()
        return
    n_paths, n_steps = 100_000, 252
    g = (100.0, 0.03, 0.2, 1.0)
    h = {"s0": 100.0, "v0": 0.04, "r": 0.03, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7}
    asian = {"s0": 100.0, "k": 100.0, "r": 0.03, "sigma": 0.2, "t": 1.0}
    cases = [
        (
            "GBM full paths (same normals, parallel loop)",
            lambda: gbm_paths(*g, n_steps, n_paths, seed=0),
            lambda: fast.gbm_paths_numba(*g, n_steps, n_paths, seed=0),
        ),
        (
            "Arithmetic Asian payoff, plain MC (in-loop RNG, no path array)",
            lambda: price_asian(
                **asian, n_steps=n_steps, n_paths=n_paths, control_variate=False, seed=0
            ),
            lambda: fast.price_asian_numba(**asian, n_steps=n_steps, n_paths=n_paths, seed=0),
        ),
        (
            "Heston terminal prices, full-truncation Euler",
            lambda: heston_paths(**h, t=1.0, n_steps=n_steps, n_paths=n_paths, full_path=False),
            lambda: fast.heston_terminal_numba(**h, t=1.0, n_steps=n_steps, n_paths=n_paths),
        ),
    ]
    rows = []
    for name, numpy_fn, numba_fn in cases:
        numba_fn()  # compile
        t_np, t_nb = _best_of(numpy_fn), _best_of(numba_fn)
        rows.append([name, f"{t_np * 1e3:.0f}", f"{t_nb * 1e3:.0f}", f"{t_np / t_nb:.1f}x"])
    rep.add(
        f"{n_paths:,} paths x {n_steps} steps, best of 3 after compilation, on "
        f"{os.cpu_count()} CPU cores (Numba {numba.__version__}). Timings depend on the machine; "
        "the ratios are the point."
    )
    rep.add()
    rep.table(["Kernel", "NumPy (ms)", "Numba (ms)", "Speed-up"], rows)
    same = np.allclose(gbm_paths(*g, 12, 2000, seed=5), fast.gbm_paths_numba(*g, 12, 2000, seed=5))
    ref = price_asian(**asian, n_steps=50, n_paths=200_000, seed=1)
    nb = fast.price_asian_numba(**asian, n_steps=50, n_paths=200_000, seed=1)
    rep.table(
        ["Check", "Value", "Expected", ""],
        [
            [
                "Numba GBM paths equal NumPy paths for the same seed",
                str(same),
                "True",
                rep.verdict(same),
            ],
            [
                "Numba Asian price vs control-variate price",
                f"{nb.price:.4f} vs {ref.price:.4f} (SE {nb.std_error:.4f})",
                "inside 99% CI",
                rep.verdict(nb.contains(ref.price)),
            ],
        ],
    )
    rep.add(
        "Reading the table: NumPy's vectorised generator is already fast, so random number "
        "generation dominates and the gain from compiling the loop is modest; it scales with "
        "core count because the kernels are parallel. Limitations: the in-loop kernels use "
        "Numba's Mersenne Twister generator, reseeded per chunk of 1,024 paths from (seed, chunk "
        "index), so they are reproducible but do not match the NumPy pricers path by path, and "
        "chunk streams are not formally guaranteed independent. Only plain Monte Carlo is "
        "accelerated; a good control variate (section 3.1) buys far more than a faster loop."
    )
    rep.add()


SECTIONS = [layer1, layer2, layer3, layer4, layer5, layer6, layer7]


def main() -> int:
    rep = Report()
    rep.add("# Validation report")
    rep.add()
    rep.add(
        "Generated by `scripts/validate.py`. Do not edit by hand. "
        f"Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}."
    )
    rep.add()
    rep.add(
        "Acceptance rule for prices: the reference value must lie inside the 99% confidence "
        "interval of the Monte Carlo estimate. All seeds are fixed, so the report is "
        "reproducible up to the runtime columns. With this many independent checks at the 99% "
        "level an occasional miss would be expected by chance; the seeds here are fixed and "
        "were not tuned per case."
    )
    rep.add()
    for section in SECTIONS:
        section(rep)
        print(f"{section.__name__}: {rep.passed} passed, {rep.failed} failed so far")
    rep.lines.insert(6, f"**Result: {rep.passed} checks passed, {rep.failed} failed.**\n")
    (ROOT / "docs" / "validation.md").write_text("\n".join(rep.lines) + "\n")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
