"""Calibration of the model: the five professions and every parameter, with the paper's notation.

Time is in years from 2026. The task axis is measured in junior bands: one unit is the range of
tasks a beginner handles in the first years.
"""

from dataclasses import dataclass, replace, asdict

import numpy as np

# --- Professions -----------------------------------------------------------------------------
PROFESSIONS = ["support", "development", "research", "business", "manual"]
MANUAL = 4  # index of the profession that faces the robotic frontier

# Employment weights w_j (BLS OEWS 2025), renormalised to sum to one.
WEIGHTS = np.array([0.15, 0.02, 0.035, 0.30, 0.49])
WEIGHTS = WEIGHTS / WEIGHTS.sum()

# Share of each junior band still open to people in 2026, q_j(0)
# (fall of early-career employment in exposed occupations, Brynjolfsson, Chandar and Chen 2025).
JUNIOR_OPEN_2026 = np.array([0.60, 0.80, 0.95, 0.90, 1.00])


@dataclass(frozen=True)
class Params:
    """All model parameters. Field names follow the paper's symbols."""

    # Pyramid
    s_J: float = 0.15  # junior share of a profession's jobs

    # Frontier (AI capability c and deployed frontier a)
    kappa: float = 0.5  # adoption speed: da/dt = kappa (c - a), about 40% of the gap closed in a year
    v0: float = 0.127  # deployed speed in 2026, sets the initial gap c - a = v0 / kappa
    gamma: float = 0.58  # capability gained per e-fold of effective compute
    g_H: float = 1.0  # efficiency gains of chips and algorithms, per year
    robot_speed: float = 0.3  # robotic frontier speed, as a share of the cognitive one
    robot_offset: float = 1.0  # extra entry level of manual work, in junior bands

    # Invention
    rho0: float = 0.0694  # invention productivity (about 1.2% of employment into new work per year)
    sigma: float = 0.3  # spillover of know-how between professions

    # Expertise
    theta: float = 0.7  # transmission efficiency (q_c = 4 delta / theta = 0.4)
    delta: float = 0.07  # attrition of know-how, per year
    lam_A: float = 0.02 / 0.3  # self-teaching per augmented worker: omega_A = lam_A * h
    allee: bool = True  # critical-mass (k^2) transmission; False gives the logistic variant
    nu_D: float = 0.2  # extra erosion of expertise from climate damages

    # Minds
    eps: float = 1.0  # adjustment speed of skills, per year
    h: float = 0.3  # share of augmented users
    alpha: float = 0.5  # skill shift: s_H -> 1 + alpha x, s_L -> 1 - alpha x

    # Verification and output
    zeta0: float = 0.75  # value of unchecked AI work relative to checked work, 2026
    lam_zeta: float = 0.5  # speed at which zeta closes its gap to 1, per junior band of frontier
    v0_check: float = 1.0  # volume of delegated work one unit of expertise can check
    A: float = 1.3  # productivity of AI work relative to a person

    # Training and myopia
    phi: float = 0.1  # share of employment in firms that keep training
    react: bool = True  # professions mobilise and a policy can be adopted
    p_react: float = 0.15  # job loss at which a profession reacts
    d: float = 2.0  # delay between pressure and action, years
    r_pi: float = 2.0  # speed of mobilisation once a profession reacts
    m_pol: float = 0.5  # share of the workforce a coalition needs
    transversal: bool = False  # professions also react to the losses of their neighbour

    # Policy levers (active once the coalition has formed)
    z: float = 0.0  # share of the junior band restored by training positions
    h_pol: float = 0.3  # augmented share under the policy (AI as a tutor)
    tau: float = 0.0  # share of capital income returned to households
    mu_pol: float = 0.0  # share of emissions cut on purpose
    s_c_pol: float = 0.5  # clean share of research under the policy

    # Income
    eta0: float = 0.528  # labour share in 2026
    m_L: float = 0.7  # spending rate out of wages
    m_K: float = 0.08  # spending rate out of capital income

    # Energy for AI
    chi: float = 0.205  # investment rate: energy for AI x2.3 in six years when all output is sold (IEA)
    delta_E: float = 0.05  # depreciation
    E_bar: float = 20.0  # ceiling set by grids and generation

    # Carbon and climate
    T0: float = 1.40  # warming in 2026: 1.37 degC (human-induced, 2025) + 0.27 degC per decade
    F0: float = 0.0406  # share of the 2 degC budget used per unit of 2026 output
    g0: float = 0.021  # decoupling of carbon intensity, per year
    lam_AI: float = 0.0  # boost AI gives to clean research, active once the policy is in place
    s_c_market: float = 0.5  # clean share of research without policy
    psi2: float = 0.003467  # DICE quadratic damages
    Omega: float = 0.15  # tipping loss
    T_tip: float = 2.0  # tipping temperature, degC
    w_T: float = 0.2  # width of the tipping transition, degC
    abate_level: float = (
        0.10  # cost of cutting emissions: abate_level * exp(-abate_decay t) * mu^abate_convexity
    )
    abate_decay: float = 0.017
    abate_convexity: float = 2.6

    def with_(self, **changes) -> "Params":
        """A copy with some parameters changed."""
        return replace(self, **changes)

    def as_dict(self) -> dict:
        return asdict(self)

    # --- Derived quantities ----------------------------------------------------------------
    @property
    def ell(self) -> float:
        """Height of the pyramid, in junior bands (6.15 for s_J = 0.15)."""
        return -1.0 / np.log(1 - self.s_J)

    @property
    def q_c(self) -> float:
        """Minimal junior exposure for know-how to survive by transmission alone."""
        return 4 * self.delta / self.theta

    @property
    def omega_A(self) -> float:
        """Self-teaching rate in 2026."""
        return self.lam_A * self.h

    def entry_levels(self) -> np.ndarray:
        """Entry level beta_j of each profession, set so that a frontier at 0 leaves q_j(0) open."""
        r = 1 / self.ell
        L = np.exp(-r) + JUNIOR_OPEN_2026 * (1 - np.exp(-r))
        beta = self.ell * np.log(L)
        beta[MANUAL] += self.robot_offset
        return beta


REFERENCE = Params()
