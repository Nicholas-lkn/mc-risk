# Note 3: Exotics

## Arithmetic Asian with a geometric control
**Idea.** The arithmetic average has no closed-form law, but the geometric
average of lognormals is lognormal. With fixings at `t_i = iT/n`:

    ln G ~ N( ln S_0 + (r - q - sigma^2/2) T (n+1)/(2n),  sigma^2 T (n+1)(2n+1)/(6n^2) )

so the geometric Asian has a Black-type price. As `n -> infinity` this becomes
the Kemna-Vorst formula (volatility `sigma / sqrt(3)`).

**Why it works.** Arithmetic and geometric averages of the same path are almost
perfectly correlated (`rho > 0.999` here), and the control variate removes a
fraction `rho^2` of the variance.
**When it fails.** The correlation drops for high volatility, long maturities
or few fixings. The control mean must match the simulated fixing dates exactly:
using the continuous Kemna-Vorst value as the mean of a discretely sampled
control would bias the price.

## Barrier options and the continuity correction
**Idea.** Simulation checks the barrier only at `m` dates, so a path can cross
and return unseen. A discretely monitored knock-out is therefore worth more
than the continuous closed form says. Broadie, Glasserman and Kou (1997):

    V_discrete(H) ~ V_continuous( H exp(-/+ beta sigma sqrt(T/m)) ),   beta = -zeta(1/2)/sqrt(2 pi) ~ 0.5826

with the minus sign for a down barrier (shift it away from the spot).
**Why it works.** The overshoot of a random walk past a level has a known mean
in the limit, which translates into a barrier shift.
**When it fails.** Small `m`, or a barrier within a standard deviation or two of
one step from the spot. Monte Carlo error is also large for barriers because
the payoff is discontinuous; QMC helps little.

## American options by Longstaff-Schwartz
**Idea.** Exercise is optimal when the immediate payoff exceeds the
continuation value `C_i(S) = E[ discounted future cash flow | S_{t_i} = S ]`.
LSM estimates that conditional expectation by regressing realised discounted
cash flows on basis functions of the state, backwards in time:

    C_i(S) ~ sum_k beta_k phi_k(S),   exercise if payoff(S) > C_i(S)

Only in-the-money paths enter the regression: out-of-the-money paths never
exercise, and dropping them lets the polynomial fit the region that matters.
The price is the average of the discounted cash flows under the fitted rule.

**Why it works.** Regression is the L2 projection onto the basis, which is the
best approximation of a conditional expectation in that span. Any exercise rule
gives a lower bound; a good rule gives a tight one.
**When it fails.** Too few paths or too rich a basis overfits (the rule then
uses future information and biases the price up). Too poor a basis biases it
down. With `n` exercise dates the result is a Bermudan price, slightly below
the American one. Polynomial bases do not scale past a few dimensions.
