"""Semi-analytic Heston price by Fourier inversion of the characteristic function."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad


def heston_cf(
    u: complex,
    s0: float,
    v0: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    t: float,
    q: float = 0.0,
) -> complex:
    """Characteristic function E[exp(i u ln S_T)] under the risk-neutral measure.

    Uses the formulation of Albrecher et al. (2007), "the little Heston trap",
    which avoids the branch-cut discontinuity of the original 1993 formula.
    """
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b * b + xi * xi * (iu + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * t)
    c_term = (r - q) * iu * t + kappa * theta / xi**2 * (
        (b - d) * t - 2.0 * np.log((1.0 - g * e) / (1.0 - g))
    )
    d_term = (b - d) / xi**2 * (1.0 - e) / (1.0 - g * e)
    return np.exp(c_term + d_term * v0 + iu * math.log(s0))


def heston_price(
    s0: float,
    k: float,
    v0: float,
    r: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    t: float,
    option: str = "call",
    q: float = 0.0,
    upper: float = 250.0,
) -> float:
    """European option price under Heston.

        C = S0 e^{-qT} P1 - K e^{-rT} P2
        P2 = 1/2 + (1/pi) int_0^inf Re[ e^{-iu ln K} phi(u) / (iu) ] du
        P1 = 1/2 + (1/pi) int_0^inf Re[ e^{-iu ln K} phi(u - i) / (iu phi(-i)) ] du

    The integrals are truncated at ``upper`` and evaluated by adaptive quadrature.
    """
    args = (s0, v0, r, kappa, theta, xi, rho, t, q)
    log_k = math.log(k)
    forward = s0 * math.exp((r - q) * t)  # phi(-i)

    def integrand_p2(u: float) -> float:
        return (np.exp(-1j * u * log_k) * heston_cf(u, *args) / (1j * u)).real

    def integrand_p1(u: float) -> float:
        return (np.exp(-1j * u * log_k) * heston_cf(u - 1j, *args) / (1j * u * forward)).real

    p1 = 0.5 + quad(integrand_p1, 1e-10, upper, limit=500)[0] / math.pi
    p2 = 0.5 + quad(integrand_p2, 1e-10, upper, limit=500)[0] / math.pi
    call = s0 * math.exp(-q * t) * p1 - k * math.exp(-r * t) * p2
    if option == "call":
        return call
    if option == "put":
        return call - s0 * math.exp(-q * t) + k * math.exp(-r * t)
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")
