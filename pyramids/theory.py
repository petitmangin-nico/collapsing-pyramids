"""Closed-form results of the paper (Propositions 1 to 5), used to check the simulations."""

import numpy as np

from .parameters import REFERENCE, Params


def fold_threshold(p: Params = REFERENCE) -> float:
    """Proposition 1: minimal exposure q_c = 4 delta / theta."""
    return p.q_c


def living_and_critical_levels(q: float, p: Params = REFERENCE):
    """Proposition 1: k_+ and k_- for an exposure q >= q_c (None below the fold)."""
    if q < p.q_c:
        return None
    root = np.sqrt(1 - p.q_c / q)
    return 0.5 * (1 + root), 0.5 * (1 - root)


def folding_loss(p: Params = REFERENCE, q_threshold: float | None = None) -> float:
    """Share of jobs lost when the junior band reaches the threshold: p* = (1 - q_c) s_J."""
    q = p.q_c if q_threshold is None else q_threshold
    return (1 - q) * p.s_J


def self_taught_floor(omega: float, p: Params = REFERENCE) -> float:
    """Expertise kept alive without any base: k_f = omega / (omega + delta)."""
    return omega / (omega + p.delta)


def remedy_price(omega: float, p: Params = REFERENCE) -> float:
    """Proposition 3: effective exposure needed for expertise to settle above one half,
    q_omega = q_c (1 - omega / delta) = 4 (delta - omega) / theta."""
    return 4 * (p.delta - omega) / p.theta


def omega_star(exposure: float, p: Params = REFERENCE, grid: int = 200_001) -> float:
    """Largest self-teaching rate at which the expertise equation still has several equilibria;
    above it, expertise has a single stable level (Proposition 3).

    Several equilibria end where two merge: f(k) = 0 and f'(k) = 0, with
    f(k) = theta q k^2 (1 - k) + omega (1 - k) - delta k. The first condition gives omega(k),
    the second picks the k where the merge happens; omega* is the largest such omega >= 0."""
    a = p.theta * exposure
    k = np.linspace(1e-6, 1 - 1e-6, grid)
    omega = (p.delta * k - a * k**2 * (1 - k)) / (1 - k)
    slope = a * (2 * k - 3 * k**2) - omega - p.delta
    merge = np.nonzero(np.diff(np.sign(slope)) != 0)[0]
    values = [float(omega[i]) for i in merge if omega[i] >= 0]
    return max(values, default=0.0)


def omega_star_max(p: Params = REFERENCE) -> float:
    """Supremum of omega* over all exposures: delta / 8, reached at exposure 27 delta / (8 theta),
    where the three equilibria merge at k = 1/3."""
    return p.delta / 8


def deadline_without_self_teaching(k_c: float, exposure: float, p: Params = REFERENCE) -> float:
    """Appendix A.4: years between the end of the base and the last useful policy,
    T_k = ln(k_c / k_-) / delta."""
    _, k_minus = living_and_critical_levels(exposure, p)
    return float(np.log(k_c / k_minus) / p.delta)


def minimal_redistribution(eta: float, eta0: float) -> float:
    """Proposition 5: tau* = (eta0 - eta) / (1 - eta), whatever the spending rates."""
    return (eta0 - eta) / (1 - eta)
