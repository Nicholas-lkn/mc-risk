# Note 5: Monte Carlo Greeks

A Greek is a derivative of an expectation, `d/dtheta E[f(S_T(theta))]`. The
three methods differ in where they put the derivative.

## Finite differences with common random numbers
    Delta ~ [V(S_0 + h) - V(S_0 - h)] / (2h)

Using the **same normals** in both valuations makes the two estimates highly
correlated, so their difference has small variance. With independent draws the
variance is of order `1/h^2`.
**Bias** `O(h^2)`; **cost** two valuations per Greek.
**Fails when** the payoff is discontinuous: only paths that cross the jump
between the two bumps contribute, so the variance grows like `1/h`.

## Pathwise
Move the derivative inside the expectation and differentiate the payoff along
each path:

    Delta = E[ e^{-rT} 1{S_T > K} S_T / S_0 ]
    Vega  = E[ e^{-rT} 1{S_T > K} S_T (sqrt(T) Z - sigma T) ]

using `dS_T/dS_0 = S_T/S_0` and `dS_T/dsigma = S_T (sqrt(T) Z - sigma T)`.
**Unbiased, lowest variance, one valuation.**
**Fails when** the payoff is not continuous in `S_T`. For a digital the
pathwise derivative is zero almost surely, which gives delta = 0: wrong, with
zero reported error. The same problem hits gamma of a vanilla.

## Likelihood ratio (score function)
Differentiate the density of `S_T` instead of the payoff:

    d/dtheta E[f(S_T)] = E[ f(S_T) * d ln p_theta(S_T) / dtheta ]
    score_delta = Z / (S_0 sigma sqrt(T)),     score_vega = (Z^2 - 1)/sigma - Z sqrt(T)

**Unbiased for any payoff**, including digitals and barriers.
**Fails when** the score is large relative to the payoff. The score scales like
`1/(sigma sqrt(T))`. For a vanilla this is offset by the payoff, which scales
like `sigma sqrt(T)`, and the penalty against pathwise is a constant factor
(about 2 in standard error for delta, 3 for vega in the validation report). For
a short-dated digital there is no offset. For a path simulated in `m` steps the
delta score depends on the first step only, `Z_1 / (S_0 sigma sqrt(T/m))`, so
the variance grows linearly in `m`.

## Rule of thumb
Pathwise when the payoff is continuous, likelihood ratio when it is not, finite
differences as an independent check on both.
