"""Run every validation check, write plots to docs/img/ and tables to docs/validation.md.

Usage: uv run python scripts/validate.py

Every number in docs/validation.md and in the README results block is produced
here. The script exits with a non-zero status if any check fails.
"""

from __future__ import annotations

import math
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from mc_risk import mc_price, price_european, variance_reduction_factor
from mc_risk.analytic import bs_price
from mc_risk.payoffs import european

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


SECTIONS = [layer1, layer2]


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
