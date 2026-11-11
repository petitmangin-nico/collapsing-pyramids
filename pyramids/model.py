"""The dynamical model, integrated with an explicit Euler scheme from 2026.

State variables
    frontier a, capability c        cognitive frontier (manual work has its own, a_R and c_R)
    invention X_j                   ground won by new tasks in each profession
    expertise k_j                   share of each profession's know-how still alive
    skills s_H, s_L                 skill of the augmented and of the substituted
    mobilisation pi_j               how far each profession has mobilised
    energy E                        energy available to AI (2026 = 1)
    budget B                        share of the 2 degC carbon budget left (2026 = 1)
    intensity iota                  carbon intensity of output (2026 = 1)
"""

import numpy as np

from .parameters import MANUAL, REFERENCE, WEIGHTS, Params

W = WEIGHTS

# Series recorded at every step. Scalars first, then one column per profession.
SCALAR_SERIES = [
    "a",
    "c",
    "K",
    "s",
    "x",
    "v",
    "Y",
    "Yr",
    "u",
    "E",
    "B",
    "T",
    "D",
    "Pi",
    "Q",
    "Xdot",
    "sH",
    "sL",
    "zeta",
    "wage",
]
PROFESSION_SERIES = ["q", "L", "k", "pi"]
SERIES_DOC = {
    "a": "deployed cognitive frontier",
    "c": "AI capability",
    "K": "average expertise",
    "s": "average skill s-bar",
    "x": "share of jobs delegated to AI",
    "v": "share of delegated work checked",
    "Y": "potential output",
    "Yr": "output sold",
    "u": "share of output sold",
    "E": "energy for AI",
    "B": "carbon budget left",
    "T": "warming (degC)",
    "D": "damage factor",
    "Pi": "policy in place (0/1)",
    "Q": "qualified employment",
    "Xdot": "speed of invention",
    "sH": "skill of the augmented",
    "sL": "skill of the substituted",
    "zeta": "value of unchecked AI work",
    "wage": "wage bill",
    "q": "junior band open to people",
    "L": "share of jobs still human",
    "k": "expertise",
    "pi": "mobilisation",
}


# --- Building blocks (one per equation of the paper) ------------------------------------------
def warming(B, p: Params):
    """Warming rises linearly from T0 in 2026 (B = 1) to 2 degC when the budget is spent (B = 0)."""
    return 2.0 - (2.0 - p.T0) * B


def damage_factor(T, p: Params):
    """Share of output left after climate damages: DICE quadratic term times a tipping loss."""
    tipping = p.Omega / (1 + np.exp(-(T - p.T_tip) / p.w_T))
    return (1 - tipping) / (1 + p.psi2 * T**2)


def human_share(front, base, ell):
    """L_j: share of a profession's jobs above the frontier (exponential pyramid)."""
    return np.exp(-np.maximum(front - base, 0) / ell)


def junior_open(L, s_J):
    """q_j: share of the junior band still open, 1 - p_j / s_J, clipped to [0, 1]."""
    return np.clip((L - (1 - s_J)) / s_J, 0, 1)


def expertise_rate(k, exposure, skill, omega, p: Params, D):
    """dk/dt: transmission (with critical mass) + self-teaching - attrition - climate erosion."""
    mass = k if p.allee else 1.0
    transmission = p.theta * exposure * skill * k * mass * (1 - k)
    return transmission + omega * (1 - k) - p.delta * k - p.nu_D * (1 - D) * k


def full_base_expertise(p: Params):
    """k_+ with the whole junior band open and no self-teaching (k_0 in the paper, about 0.89)."""
    if p.allee:
        return 0.5 * (1 + np.sqrt(1 - p.q_c))
    return 1 - p.delta / p.theta


def initial_expertise(p: Params, beta):
    """Each profession starts in 2026 at the stable upper root of its own expertise equation."""
    q0 = junior_open(human_share(0.0, beta, p.ell), p.s_J)
    exposure = q0 + p.phi * (1 - q0)
    D0 = damage_factor(warming(1.0, p), p)
    grid = np.linspace(1, 0, 20001)
    return np.array([grid[np.argmax(expertise_rate(grid, e, 1.0, p.omega_A, p, D0) >= 0)] for e in exposure])


# --- Simulation -------------------------------------------------------------------------------
class Trajectory(dict):
    """Model output: arrays indexed by time step, also readable as attributes (traj.K)."""

    __getattr__ = dict.__getitem__

    def at(self, name, year):
        """Value of a series at the step closest to a calendar year."""
        return self[name][np.argmin(abs(self["t"] - year))]

    def first_year(self, condition):
        """First calendar year at which a boolean series is true, or None."""
        return float(self["t"][np.argmax(condition)]) if condition.any() else None


def simulate(p: Params = REFERENCE, years: float = 100.0, dt: float = 0.02) -> Trajectory:
    """Integrate the model over `years` from 2026 and return every series."""
    n = int(years / dt) + 1
    lag = int(round(p.d / dt))
    beta = p.entry_levels()
    k_full = full_base_expertise(p)

    # Initial state
    a = a_R = 0.0
    c = p.v0 / p.kappa
    c_R = c * p.robot_speed
    X = np.zeros(5)
    k = initial_expertise(p, beta)
    sH = sL = 1.0
    pi = np.zeros(5)
    E = B = iota = 1.0

    out = Trajectory({name: np.zeros(n) for name in SCALAR_SERIES})
    out.update({name: np.zeros((n, 5)) for name in PROFESSION_SERIES})
    losses = np.zeros((n, 5))
    Q0 = Y0 = C0Y0 = None
    t_policy = None

    for i in range(n):
        t = i * dt

        # Frontier and pyramid
        front = np.array([a, a, a, a, a_R])
        L = human_share(front, beta + X, p.ell)
        q = junior_open(L, p.s_J)
        losses[i] = 1 - L

        # Policy and minds
        policy = 1.0 if (p.react and float(W @ pi) > p.m_pol) else 0.0
        if policy and t_policy is None:
            t_policy = t
        h = max(p.h, p.h_pol) if policy else p.h
        x = 1 - float(W @ L)
        K = float(W @ k)
        s = h * sH + (1 - h) * sL

        # Verification
        v = min(1.0, p.v0_check * K * s / max(x, 1e-9))
        zeta = 1 - (1 - p.zeta0) * np.exp(-p.lam_zeta * a)

        # Climate and abatement
        T = warming(B, p)
        D = damage_factor(T, p)
        mu = p.mu_pol * policy
        s_c = p.s_c_pol if policy else p.s_c_market
        abatement = p.abate_level * np.exp(-p.abate_decay * t) * mu**p.abate_convexity

        # Output, wages and demand
        Q = float(W @ (L * k))
        Y = (1 - abatement) * D * (Q / k_full + p.A * x * (v + zeta * (1 - v)))
        if i == 0:
            Q0, Y0 = Q, Y
        wage = p.eta0 * Y0 * Q / Q0
        profit = max(Y - wage, 0)
        tau = p.tau * policy
        C = p.m_L * (wage + tau * profit) + p.m_K * (1 - tau) * profit
        if C0Y0 is None:
            C0Y0 = C / Y
        u = min(1.0, (C / Y) / C0Y0)
        Yr = u * Y

        # Invention: own expertise plus spillovers from the other professions
        Xdot = p.rho0 * s * (k + p.sigma * (K - W * k))

        for name, value in zip(
            SCALAR_SERIES,
            [a, c, K, s, x, v, Y, Yr, u, E, B, T, D, policy, Q, float(W @ Xdot), sH, sL, zeta, wage],
        ):
            out[name][i] = value
        out["q"][i], out["L"][i], out["k"][i], out["pi"][i] = q, L, k, pi
        if i == n - 1:
            break

        # Mobilisation: each profession reacts to its own losses, seen with a delay d
        if p.react and i >= lag:
            seen = losses[i - lag].copy()
            if p.transversal:
                seen = np.maximum(seen, np.concatenate([[0.0], seen[:-1]]))
            pi = pi + dt * p.r_pi * (1 - pi) * (seen > p.p_react)

        # Energy, compute and frontier
        dE = p.chi * u * E * (1 - E / p.E_bar) - p.delta_E * E
        compute_growth = dE / E + p.g_H
        c += dt * p.gamma * compute_growth
        c_R += dt * p.gamma * compute_growth * p.robot_speed
        a += dt * p.kappa * max(c - a, 0)
        a_R += dt * p.kappa * max(c_R - a_R, 0)

        # Invention, expertise and skills
        X = X + dt * Xdot
        exposure = q + max(p.phi, p.z * policy) * (1 - q)
        k = np.clip(k + dt * expertise_rate(k, exposure, s, p.lam_A * h * sH, p, D), 0, 1)
        sH += dt * p.eps * (1 + p.alpha * x - sH)
        sL += dt * p.eps * (1 - p.alpha * x - sL)

        # Energy stock and carbon
        E = max(E + dt * dE, 1e-6)
        B -= dt * p.F0 * iota * (1 - mu) * Yr
        iota *= np.exp(-(p.g0 + p.lam_AI * s_c * policy) * dt)

    out["t"] = 2026 + np.arange(n) * dt
    out["t_policy"] = None if t_policy is None else 2026 + t_policy
    return out
