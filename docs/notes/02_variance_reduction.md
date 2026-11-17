# Note 2: Variance reduction

Plain Monte Carlo error is `sigma_Y / sqrt(N)`. Every method here leaves the
`sqrt(N)` alone (except Sobol) and shrinks `sigma_Y`.

## Antithetic variates
**Idea.** Use each normal draw twice, as `Z` and `-Z`, and average the pair.

    Var((Y + Y') / 2) = (Var(Y) / 2) (1 + rho),   rho = Corr(f(Z), f(-Z))

**Works when** the payoff is monotone in `Z` (calls, puts), so `rho < 0`.
**Fails when** the payoff is symmetric in `Z` (straddle at the forward): `rho`
is near +1 and the second evaluation is wasted.

## Control variates
**Idea.** Subtract a correlated quantity `X` whose mean is known exactly:

    Y_cv = Y - beta (X - E[X]),   beta* = Cov(Y, X) / Var(X),   Var(Y_cv) = Var(Y) (1 - rho^2)

Here `X` is the discounted terminal stock price, with `E[X] = S_0 e^{-qT}`.
**Works when** `|rho|` is close to 1. The gain is modest for an out-of-the-money
call against the underlying, and very large for an arithmetic Asian against the
geometric Asian (note 3).
**Fails when** the control is weakly correlated, or its "known" mean is itself
approximate, which silently biases the price.

## Randomised quasi-Monte Carlo (scrambled Sobol)
**Idea.** Replace random points with a low-discrepancy sequence that fills the
unit cube evenly. The error rate improves from `N^{-1/2}` towards `N^{-1}` for
smooth, low-dimensional integrands. A deterministic sequence has no error
estimate, so the points are scrambled (Owen scrambling): each scramble is
unbiased, and the spread of `m` independent scramble means gives a standard
error.

**Brownian bridge.** Sobol points are best in their leading dimensions. The
bridge spends the first coordinate on `W(T)`, the second on `W(T/2)` given the
ends, and so on:

    W(t_m) | W(t_l), W(t_r) ~ N( ((t_r - t_m) W(t_l) + (t_m - t_l) W(t_r)) / (t_r - t_l),
                                 (t_m - t_l)(t_r - t_m) / (t_r - t_l) )

This lowers the effective dimension of the integrand without changing its law.

**Fails when** the payoff is discontinuous (barriers, digitals) or genuinely
high-dimensional: the gain shrinks towards plain Monte Carlo. Use powers of two
for the number of points; other sizes lose the balance properties.

**Error estimate caveat.** The scramble means are not exactly normal. In this
project a 99% t interval from 16 scrambles covered the true price only about
96% of the time; 64 scrambles brought coverage back to about 99% (see the
validation report). More scrambles of fewer points trade a little accuracy for
an honest error bar.
