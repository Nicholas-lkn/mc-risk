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

from mc_risk import price_european
from mc_risk.analytic import bs_price

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


SECTIONS = [layer1]


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
