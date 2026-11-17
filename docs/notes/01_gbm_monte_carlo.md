# Note 1: GBM and plain Monte Carlo

**Idea.** Under the risk-neutral measure the price of a payoff is its discounted
expectation, `V = e^{-rT} E[f(S)]`. Monte Carlo replaces the expectation with a
sample mean over simulated paths.

**Key formulas.** GBM solves `dS = (r - q) S dt + sigma S dW` exactly:

    S_T = S_0 exp((r - q - sigma^2 / 2) T + sigma sqrt(T) Z),   Z ~ N(0, 1)

so paths are simulated without discretisation error. The estimator and its
error are

    V_hat = (1/N) sum_i Y_i,   SE = s_Y / sqrt(N),   99% CI = V_hat +/- 2.576 SE

**Why it works.** The law of large numbers gives convergence and the central
limit theorem gives the error: it shrinks like `1/sqrt(N)` regardless of the
dimension of the problem. That is the slope checked in the convergence plot.

**When it fails or struggles.**
- Slow: one more digit of accuracy costs 100 times more paths. Layer 2 attacks the constant.
- The CLT interval is unreliable for heavy-tailed payoffs (deep out-of-the-money
  options with few paths finishing in the money).
- The model, not the method, is usually the larger error: constant volatility
  and no jumps do not match observed smiles (see notes 4).
