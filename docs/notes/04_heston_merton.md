# Note 4: Heston and Merton

## Heston stochastic volatility
**Idea.** Let the variance follow its own mean-reverting (CIR) process,
correlated with the stock:

    dS = (r - q) S dt + sqrt(v) S dW_s
    dv = kappa (theta - v) dt + xi sqrt(v) dW_v,    d<W_s, W_v> = rho dt

Negative `rho` produces the equity skew; `xi` (vol of vol) produces the smile.

**Simulation.** An Euler step can push `v` below zero, where `sqrt(v)` is not
defined. Full truncation keeps the raw `v` but uses `v+ = max(v, 0)` in every
coefficient:

    v_{i+1}    = v_i + kappa (theta - v_i+) dt + xi sqrt(v_i+ dt) Z_v
    ln S_{i+1} = ln S_i + (r - q - v_i+/2) dt + sqrt(v_i+ dt) Z_s

Among the simple fixes (absorb, reflect, truncate) it has the smallest bias
(Lord, Koekkoek and van Dijk, 2010).

**Reference price.** The characteristic function of `ln S_T` is known in closed
form, so `C = S_0 e^{-qT} P_1 - K e^{-rT} P_2` with

    P_2 = 1/2 + (1/pi) int_0^inf Re[ e^{-iu ln K} phi(u) / (iu) ] du

and `P_1` the same with `phi(u - i) / phi(-i)`. The "little Heston trap"
form of `phi` avoids a branch-cut jump in the complex logarithm.

**When it fails.** The scheme is biased for any finite step, and the bias grows
when the Feller condition `2 kappa theta >= xi^2` fails, because the variance
spends time near zero where the square root is steep. A Monte Carlo confidence
interval says nothing about this bias: check against the semi-analytic price,
or halve the step and compare.

## Merton jump diffusion
**Idea.** Add Poisson-driven lognormal jumps to GBM:

    dS/S = (r - q - lambda k) dt + sigma dW + (J - 1) dN,   ln J ~ N(mu_J, delta^2),   k = E[J] - 1

The `-lambda k` term compensates the jumps so the discounted price stays a
martingale.

**Why the series works.** Conditional on `n` jumps, `ln S_T` is normal with
variance `sigma^2 T + n delta^2`, so the price is a Poisson-weighted sum of
Black-Scholes prices:

    V = sum_n e^{-lambda' T} (lambda' T)^n / n! * BS(S_0, K, r_n, sigma_n, T)
    lambda' = lambda (1 + k),  sigma_n^2 = sigma^2 + n delta^2 / T,  r_n = r - lambda k + n ln(1 + k) / T

Simulation is exact at any horizon: draw the number of jumps, then one normal
for their sum.

**When it fails.** The market is incomplete, so the price depends on the chosen
risk-neutral jump parameters, and hedging with the stock alone does not remove
jump risk. Short-dated smiles fit well; the term structure of skew does not.
