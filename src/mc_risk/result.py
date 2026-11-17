"""Result containers shared by every pricer and estimator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

Z99 = 2.5758293035489004  # two-sided 99% standard normal quantile


@dataclass(frozen=True)
class PriceResult:
    """Monte Carlo estimate with its sampling error.

    Attributes:
        price: point estimate (discounted expected payoff, or a Greek).
        std_error: standard error of the estimate.
        ci99: two-sided 99% confidence interval (low, high).
        n_paths: number of simulated paths (function evaluations) used.
        runtime: wall-clock seconds.
        method: label of the estimator that produced the number.
    """

    price: float
    std_error: float
    ci99: tuple[float, float]
    n_paths: int
    runtime: float
    method: str = "plain"

    @classmethod
    def from_samples(
        cls,
        samples: np.ndarray,
        n_paths: int,
        runtime: float,
        method: str = "plain",
        student_t: bool = False,
    ) -> PriceResult:
        """Build a result from i.i.d. samples of an unbiased estimator.

        With ``student_t=True`` the CI uses a t quantile, which is the right
        choice when ``samples`` is a small set of batch means (RQMC scrambles).
        """
        samples = np.asarray(samples, dtype=float)
        n = samples.size
        mean = float(samples.mean())
        se = float(samples.std(ddof=1) / np.sqrt(n))
        crit = float(stats.t.ppf(0.995, n - 1)) if student_t else Z99
        return cls(mean, se, (mean - crit * se, mean + crit * se), int(n_paths), runtime, method)

    def contains(self, reference: float) -> bool:
        """True if ``reference`` lies inside the 99% confidence interval."""
        return self.ci99[0] <= reference <= self.ci99[1]
