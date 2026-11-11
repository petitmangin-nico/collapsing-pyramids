"""Every number quoted in the paper, written to results/numbers.json (about 15 s).

Run from the repository root:  python scripts/compute_numbers.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pyramids import FULL_POLICY, KNOWLEDGE, PLANET, REFERENCE, SOCIETIES, WEIGHTS, simulate  # noqa: E402
from pyramids import theory  # noqa: E402

HORIZON = 24  # 2026 to 2050
DT = 0.02
DEV = 1  # index of software development, the profession of Figure 1


def run(**changes):
    return simulate(REFERENCE.with_(**changes), years=HORIZON, dt=DT)


def lost_output(traj, until):
    """Production-years (years of 2026 output sold) lost below the 2026 level, from 2026 to `until`."""
    y, t = traj["Yr"] / traj["Yr"][0], traj["t"]
    return float(np.sum(np.clip(1 - y[t <= until], 0, None)) * DT)


def effective_exposure(traj, p=REFERENCE):
    """Junior exposure counting the firms that keep training, q + phi (1 - q)."""
    return traj["q"] + p.phi * (1 - traj["q"])


def folded_share(traj, year, p=REFERENCE):
    """Share of employment whose base has crossed the fold by a given year."""
    i = np.argmin(abs(traj["t"] - year))
    return float(WEIGHTS @ (effective_exposure(traj, p)[i] < p.q_c))


def crossing_year(traj):
    """Year 2 degC is crossed, or None."""
    return traj.first_year(traj["T"] > 2.0)


def section_2(R, nr):
    """Pyramid, fold and invention (no reaction)."""
    i = np.argmin(abs(nr["t"] - 2028))
    R["F1_2028_lost"] = 1 - nr["L"][i, DEV]
    R["F1_2028_q"] = nr["q"][i, DEV]
    exposure = effective_exposure(nr)
    R["fold"] = {j: nr.first_year(exposure[:, j] < REFERENCE.q_c) for j in range(5)}
    R["react15"] = {j: nr.first_year(1 - nr["L"][:, j] > REFERENCE.p_react) for j in range(5)}
    R["positions_down_fifth"] = nr.first_year(nr["x"] > 0.2)
    for year in [2030, 2035, 2040, 2050]:
        R[f"inv{year}"] = nr.at("Xdot", year) / nr["Xdot"][0]
    R["K0"] = float(nr["K"][0])
    R["k_init"] = [float(v) for v in nr["k"][0]]
    R["K2050"] = nr.at("K", 2050)
    R["K2050_rel"] = R["K2050"] / nr["K"][0]
    R["Q2050_rel"] = nr.at("Q", 2050) / nr["Q"][0]
    R["folding_loss"] = theory.folding_loss()
    p = REFERENCE
    R["folding_loss_with_training"] = theory.folding_loss(q_threshold=(p.q_c - p.phi) / (1 - p.phi))
    # With self-teaching, the share of jobs lost at which expertise falls through one half
    R["half_expertise_loss_self_taught"] = theory.folding_loss(q_threshold=theory.remedy_price(p.omega_A))


def section_3(R, nr):
    """Skills, remedy and verification."""
    R["x2050"] = nr.at("x", 2050)
    R["sH2050"] = nr.at("sH", 2050)
    R["sL2050"] = nr.at("sL", 2050)
    R["s2050"] = nr.at("s", 2050)
    R["zeta095"] = nr.first_year(nr["zeta"] > 0.95)
    tutor = run(react=False, h=0.6)
    R["s2050_tutor"] = tutor.at("s", 2050)
    R["inv2050_tutor"] = tutor.at("Xdot", 2050) / tutor["Xdot"][0]
    R["K2050_tutor"] = tutor.at("K", 2050)
    # Price of the remedy with self-teaching, q_omega / s-bar (Proposition 3)
    for label, traj, h in [("today", nr, REFERENCE.h), ("tutor", tutor, 0.6)]:
        for year in [2026, 2050]:
            omega = REFERENCE.lam_A * h * traj.at("sH", year)
            R[f"price_{label}_{year}"] = theory.remedy_price(omega) / traj.at("s", year)
    R["q_omega_2026"] = theory.remedy_price(REFERENCE.omega_A)
    R["q_omega_2050"] = theory.remedy_price(REFERENCE.omega_A * R["sH2050"])
    R["floor2026"] = theory.self_taught_floor(REFERENCE.omega_A)
    R["omega_star_max"] = theory.omega_star_max()


def section_4(R, nr):
    """Training dilemma, the price of myopia (Proposition 4) and the deadline without self-teaching."""
    for phi in [0.1, 0.5]:
        R[f"K2050_phi{phi}"] = run(react=False, phi=phi).at("K", 2050)
    for m_pol in [0.5, 0.8]:
        dates = {}
        for transversal in [False, True]:
            key = "tr" if transversal else "my"
            dates[key] = run(d=0.0, m_pol=m_pol, transversal=transversal)["t_policy"]
            R[f"coal_m{m_pol}_{key}"] = dates[key]
            R[f"lost_m{m_pol}_{key}"] = lost_output(nr, dates[key])
        R[f"myopia_years_m{m_pol}"] = dates["my"] - dates["tr"]
        R[f"folded_share_m{m_pol}"] = folded_share(nr, dates["my"])
    # Appendix A.4: development without self-teaching
    st0 = run(react=False, lam_A=0.0)
    gone = np.argmax(st0["q"][:, DEV] <= 1e-6)
    R["base_gone_dev"] = st0.first_year(st0["q"][:, DEV] <= 1e-6)
    R["k0_dev_no_self_teaching"] = float(st0["k"][0, DEV])
    R["k_c_dev_no_self_teaching"] = float(st0["k"][gone, DEV])
    R["deadline_years_dev"] = theory.deadline_without_self_teaching(R["k_c_dev_no_self_teaching"], 1.0)
    R["deadline_year_dev"] = R["base_gone_dev"] + R["deadline_years_dev"]


def section_5(R, nr):
    """Demand, cost of delay, the five societies and redistribution."""
    policy_date = R["coal_m0.5_my"]
    for d in [0, 5, 10, 15]:
        R[f"lost_d{d}"] = lost_output(nr, policy_date + d)
        traj = run(**FULL_POLICY, d=d)
        R[f"Kmin_d{d}"] = float(traj["K"].min())
        R[f"K2050_d{d}"] = float(traj["K"][-1])
        R[f"tp_d{d}"] = traj["t_policy"]
    R["lost_by_year"] = {d: lost_output(nr, policy_date + d) for d in range(16)}
    R["u2050_nr"] = nr.at("u", 2050)
    R["Yr2050_nr"] = nr.at("Yr", 2050)
    for name, changes in SOCIETIES.items():
        traj = run(**changes)
        R["T1_" + name] = dict(
            K=traj["K"][-1],
            Q=traj["Q"][-1] / traj["Q"][0],
            u=traj["u"][-1],
            Yr=traj["Yr"][-1],
            T=traj["T"][-1],
        )
    eta0 = nr["wage"][0] / nr["Y"][0]
    R["eta0"] = eta0
    R["eta2050"] = nr.at("wage", 2050) / nr.at("Y", 2050)
    R["tau_star_2050"] = theory.minimal_redistribution(R["eta2050"], eta0)
    with_policy = run(**KNOWLEDGE, **PLANET, tau=0.0)
    R["eta2050_pol"] = with_policy.at("wage", 2050) / with_policy.at("Y", 2050)
    R["tau_star_2050_pol"] = theory.minimal_redistribution(R["eta2050_pol"], eta0)
    for tau in [0.1, 0.3, 0.5]:
        R[f"u2050_tau{tau}"] = run(**KNOWLEDGE, **PLANET, tau=tau)["u"][-1]


def section_5_climate(R):
    """Climate columns of Tables 1 and 2 and the decoupling numbers, followed to 2100."""

    def long(**changes):
        return simulate(REFERENCE.with_(**changes), years=74, dt=DT)

    for name, changes in SOCIETIES.items():
        R["T1_" + name]["cross2C"] = crossing_year(long(**changes))
    for planet in [False, True]:
        for tau in [0.0, 0.3, 0.6]:
            for d in [2, 12]:
                changes = dict(**KNOWLEDGE, tau=tau, d=d, **(PLANET if planet else {}))
                traj = long(**changes)
                R[f"T2_planet{int(planet)}_tau{tau}_d{d}"] = dict(
                    T2050=traj.at("T", 2050), cross2C=crossing_year(traj), Tmax=float(traj["T"].max())
                )
    # Decoupling pace that keeps the overshoot society below 2 degC until 2100
    for g0 in np.arange(0.021, 0.08, 0.001):
        if crossing_year(long(**SOCIETIES["Overshoot"], g0=g0)) is None:
            R["g0_needed"] = float(g0)
            break
    R["Tmax_planet_no_AI_boost"] = float(long(**{**FULL_POLICY, "lam_AI": 0.0})["T"].max())
    R["T2100_planet"] = float(long(**FULL_POLICY)["T"][-1])


def main():
    nr = run(react=False)
    R = {}
    for section in (section_2, section_3, section_4, section_5):
        section(R, nr)
    section_5_climate(R)
    out = ROOT / "results" / "numbers.json"
    out.write_text(json.dumps(R, indent=1, default=float))
    print(f"{len(R)} numbers written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
