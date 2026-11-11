"""Checks of the model. Run from the repository root:  python -m pytest tests"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pyramids import REFERENCE, simulate, theory  # noqa: E402


def flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def test_numbers_match_reference():
    """Every number quoted in the paper is reproduced exactly (tests/reference_numbers.json)."""
    import compute_numbers

    R = {}
    nr = compute_numbers.run(react=False)
    for section in (
        compute_numbers.section_2,
        compute_numbers.section_3,
        compute_numbers.section_4,
        compute_numbers.section_5,
    ):
        section(R, nr)
    compute_numbers.section_5_climate(R)
    R = json.loads(json.dumps(R, default=float))
    ref = json.loads((ROOT / "tests" / "reference_numbers.json").read_text())
    for key, value in flatten(ref):
        got = dict(flatten(R))[key]
        if value is None or got is None:
            assert got == value, key
        else:
            assert np.allclose(got, value, atol=1e-9), (key, got, value)


def test_published_numbers_are_current():
    """results/numbers.json, which the paper quotes, is the reference output."""
    published = json.loads((ROOT / "results" / "numbers.json").read_text())
    ref = json.loads((ROOT / "tests" / "reference_numbers.json").read_text())
    assert published == ref


def test_initial_state_is_an_equilibrium():
    """With the frontier, invention and emissions frozen, expertise stays where it starts in 2026."""
    frozen = REFERENCE.with_(react=False, g_H=0.0, chi=0.0, v0=1e-12, rho0=0.0, F0=0.0)
    traj = simulate(frozen, years=10)
    assert np.allclose(traj["k"][-1], traj["k"][0], atol=1e-3)


def test_fold_threshold():
    """Proposition 1: q_c = 0.4 and the folding loss is 9% of jobs."""
    assert np.isclose(theory.fold_threshold(), 0.4)
    assert np.isclose(theory.folding_loss(), 0.09)


def test_remedy_price_closed_form():
    """Proposition 3: with omega = 0.02 the price of the remedy is 4 (delta - omega) / theta."""
    assert np.isclose(theory.remedy_price(0.02), 4 * 0.05 / 0.7)
    assert theory.omega_star(0.4) < 0.007


def test_minimal_redistribution_is_independent_of_spending():
    """Proposition 5: tau* gives back the 2026 consumption share for any spending rates."""
    eta0, eta = 0.528, 0.2
    tau = theory.minimal_redistribution(eta, eta0)
    for m_L, m_K in [(0.7, 0.08), (0.9, 0.02), (0.6, 0.3)]:
        c0 = m_L * eta0 + m_K * (1 - eta0)
        c = m_L * eta + m_K * (1 - eta) + tau * (m_L - m_K) * (1 - eta)
        assert np.isclose(c, c0)


def test_invention_follows_skill_and_expertise():
    """Proposition 2: the speed of invention is X(0) s-bar K-tilde / K-tilde(0) at every date."""
    from pyramids import WEIGHTS

    p = REFERENCE
    traj = simulate(p.with_(react=False), years=24)
    w = WEIGHTS * (1 + p.sigma * (1 - WEIGHTS))
    k_tilde = traj["k"] @ w / w.sum()
    predicted = traj["Xdot"][0] * traj["s"] * k_tilde / k_tilde[0]
    assert np.allclose(traj["Xdot"], predicted, rtol=1e-12)


def test_living_and_critical_levels_are_equilibria():
    """Proposition 1: k+ and k- are zeros of the expertise equation without self-teaching."""
    from pyramids.model import expertise_rate

    p = REFERENCE.with_(lam_A=0.0, nu_D=0.0)
    for q in [0.45, 0.6, 1.0]:
        for k in theory.living_and_critical_levels(q, p):
            assert abs(expertise_rate(k, q, 1.0, 0.0, p, 1.0)) < 1e-12


def test_omega_star():
    """Proposition 3: omega* never exceeds delta / 8, and the reference self-teaching is above it."""
    for exposure in np.linspace(0.05, 3.0, 60):
        assert theory.omega_star(exposure) <= theory.omega_star_max() + 1e-4
    assert abs(theory.omega_star(0.34) - theory.omega_star_max()) < 5e-4
    assert REFERENCE.omega_A > theory.omega_star_max()


def test_myopic_policy_comes_after_the_fold():
    """Proposition 4 (i): with p_react above the folding loss counting training (10%), when a myopic
    policy is adopted, professions holding more than m_pol of employment have already crossed the fold,
    with or without political delay."""
    from pyramids import WEIGHTS

    for m_pol in [0.5, 0.8]:
        for d in [0.0, 2.0, 4.0]:
            for p_react in [0.11, 0.15, 0.25]:
                p = REFERENCE.with_(d=d, m_pol=m_pol, p_react=p_react)
                date = simulate(p, years=40)["t_policy"]
                if date is None:
                    continue
                nr = simulate(p.with_(react=False), years=40)
                i = np.argmin(abs(nr["t"] - date))
                exposure = nr["q"][i] + p.phi * (1 - nr["q"][i])
                assert WEIGHTS @ (exposure < p.q_c) > m_pol, (m_pol, d, p_react)


def test_late_policy_still_restores_expertise():
    """Corollary of Proposition 3: a policy fifteen years late still brings expertise back above one half."""
    from pyramids import FULL_POLICY

    traj = simulate(REFERENCE.with_(**FULL_POLICY, d=15), years=30)
    assert traj["K"].min() < 0.6 and traj["K"][-1] > 0.8
