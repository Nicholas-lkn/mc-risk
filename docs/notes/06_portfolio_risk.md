# Note 6: Portfolio VaR, CVaR and backtesting

**Definitions.** With loss `L = -(portfolio return)` and confidence `alpha`:

    VaR_alpha  = alpha-quantile of L          (a loss exceeded with probability 1 - alpha)
    CVaR_alpha = E[ L | L >= VaR_alpha ]      (the average loss in that tail)

CVaR is coherent (sub-additive) and looks inside the tail; VaR does neither.

**Currency.** The base currency is CAD. A US stock's CAD return is
`(1 + r_USD)(1 + r_USDCAD) - 1`, so each US position carries equity risk and
currency risk. Converting prices to CAD first handles both with no extra
factor.

## Three models
1. **Historical simulation.** Apply today's weights to each of the last `n`
   days of returns and take the empirical quantile. No distributional
   assumption, but the 99% quantile of 500 points rests on five observations
   and the estimate jumps when a bad day enters or leaves the window.
2. **Gaussian Monte Carlo.** `R ~ N(mu, Sigma)`, simulated as `mu + L Z` with
   `L L' = Sigma` (Cholesky). Thin tails: CVaR/VaR is fixed near 1.15 at 99%.
3. **Student-t Monte Carlo.** `R = mu + sqrt(nu / W) L_s Z`, `W ~ chi^2(nu)`,
   with `L_s L_s' = Sigma (nu - 2)/nu` so the covariance still matches. The
   common mixing variable `W` makes all names have a bad day together (tail
   dependence), which a Gaussian copula cannot do. `nu` is fitted by maximum
   likelihood.

For a linear portfolio models 2 and 3 have closed forms, which this project
uses to verify the simulators. The Monte Carlo machinery is what generalises
to non-linear positions.

## Backtesting
Let `I_t = 1` when the loss on day `t` exceeds the VaR forecast made the day
before, with `x` exceedances in `n` days.

- **Kupiec (unconditional coverage).** Is `x/n` consistent with `p = 1 - alpha`?

      LR_uc = -2 ln[ (1-p)^(n-x) p^x / ((1-x/n)^(n-x) (x/n)^x) ] ~ chi^2(1)

- **Christoffersen (independence).** Does an exceedance make another one more
  likely the next day? Compare a first-order Markov chain with transition
  probabilities `pi_01`, `pi_11` against a constant probability: `LR_ind ~ chi^2(1)`.
- **Conditional coverage.** `LR_cc = LR_uc + LR_ind ~ chi^2(2)`.

**When it fails.** All three models here are unconditional: they weight the
window equally and adapt slowly to volatility regimes, so exceedances cluster
at the start of a sell-off. With 1,260 test days only about 12.6 exceedances
are expected, so the tests have little power against moderate
misspecification. A model that passes is "not rejected", not "validated".

## Stress scenarios
VaR describes ordinary bad days from a recent window. Stress tests ask what
happens in a named scenario regardless of its probability: replaying a
historical episode with today's weights, or an instantaneous shock to one
factor. For a CAD investor holding US stocks, a falling Canadian dollar is a
gain; the currency loss scenario is a CAD rally.
