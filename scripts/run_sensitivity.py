"""Monte Carlo sensitivity analysis (Appendix A.8), written to results/sensitivity.json.

Every parameter that is not measured directly is drawn uniformly over its range (Table A1),
all at once, 300 times. Takes about five minutes.

Run from the repository root:  python scripts/run_sensitivity.py [--draws 300]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pyramids import KNOWLEDGE, REFERENCE, simulate  # noqa: E402

# Uniform ranges (low, high). Order matters: it fixes the random draws.
RANGES = dict(
    # pyramid, frontier, invention
    q_c=(0.3, 0.6),
    s_J=(0.10, 0.20),
    delta=(0.05, 0.10),
    gamma=(0.45, 0.75),
    kappa=(0.3, 0.7),
    g_H=(0.7, 1.3),
    robot_speed=(0.2, 0.5),
    sigma=(0.15, 0.45),
    # minds and verification
    h=(0.2, 0.4),
    alpha=(0.3, 0.7),
    lam_A_mult=(0.5, 1.5),
    zeta0=(0.65, 0.85),
    A=(1.0, 1.6),
    # training and myopia
    phi=(0.0, 0.2),
    p_react=(0.10, 0.25),
    m_pol=(0.4, 0.7),
    d=(1.0, 4.0),
    # income, energy, climate
    eta0=(0.48, 0.58),
    m_L=(0.6, 0.8),
    m_K=(0.04, 0.15),
    chi=(0.15, 0.30),
    Omega=(0.10, 0.25),
    lam_AI=(0.0, 0.04),
)
SOCIETY_KEYS = {
    "none": "World without AI",
    "hollow": "Hollow economy",
    "rent": "Concentrated wealth",
    "auto": "Autonomous society",
    "over": "Overshoot",
}
HORIZON = 24
DT = 0.02


def params_for(draw):
    """Reference parameters with one draw applied."""
    changes = {k: v for k, v in draw.items() if k not in ("q_c", "lam_A_mult", "lam_AI")}
    changes.update(
        theta=4 * draw["delta"] / draw["q_c"],
        v0=REFERENCE.v0 * draw["kappa"] / REFERENCE.kappa,  # keep the initial gap c - a = v0 / kappa
        lam_A=REFERENCE.lam_A * draw["lam_A_mult"],
    )
    return REFERENCE.with_(**changes)


def crossing_year(traj):
    hot = traj["T"] > 2.0
    return float(traj["t"][np.argmax(hot)]) if hot.any() else float("nan")


def evaluate(draw):
    """All outcomes of one draw."""
    p = params_for(draw)
    nr = simulate(p.with_(react=False), years=HORIZON)
    p_star = (1 - draw["q_c"]) * draw["s_J"]
    # Folding loss counting the firms that keep training: effective exposure q + phi (1 - q) = q_c
    p_star_phi = (1 - (draw["q_c"] - draw["phi"]) / (1 - draw["phi"])) * draw["s_J"]
    R = dict(
        pstar=p_star,
        pstar_phi=p_star_phi,
        fold_before_react=float(p_star < draw["p_react"]),
        fold_before_react_phi=float(p_star_phi < draw["p_react"]),
        inv2050=nr["Xdot"][-1] / nr["Xdot"][0],
        K2050=nr["K"][-1] / nr["K"][0],
        Q2050=nr["Q"][-1] / nr["Q"][0],
        u2050=nr["u"][-1],
    )
    eta0, eta = nr["wage"][0] / nr["Y"][0], nr["wage"][-1] / nr["Y"][-1]
    R["tau_star"] = (eta0 - eta) / (1 - eta)

    # Cost of delay: output sold lost before the policy, for 0 to 15 extra years of delay
    coalition = simulate(p.with_(d=0.0), years=HORIZON)
    if coalition["t_policy"] is not None:
        tp = coalition["t_policy"]
        R["tpol"] = tp
        long = simulate(p.with_(react=False), years=HORIZON + 16)
        y, t = long["Yr"] / long["Yr"][0], long["t"]
        lost = [float(np.sum(np.clip(1 - y[t <= tp + k], 0, None)) * DT) for k in range(16)]
        R["lost0"], R["lost5"], R["lost10"], R["lost15"] = lost[0], lost[5], lost[10], lost[15]
        steps = np.diff(lost)
        R["delay_increasing"] = float(np.all(steps[1:] >= steps[:-1] - 1e-9) and np.all(steps > 0))

    # The five societies: values in 2050 and year 2 degC is crossed, up to 2100
    planet = dict(mu_pol=0.5, s_c_pol=0.9, lam_AI=draw["lam_AI"])
    societies = {
        "none": dict(react=False, g_H=0.0, chi=0.0),
        "hollow": dict(react=False),
        "rent": dict(**KNOWLEDGE, **planet, tau=0.0),
        "auto": dict(**KNOWLEDGE, **planet, tau=0.6),
        "over": dict(**KNOWLEDGE, tau=0.6),
    }
    for key, changes in societies.items():
        traj = simulate(p.with_(**changes), years=74)
        j = np.argmin(abs(traj["t"] - 2050))
        R[f"{key}_K"], R[f"{key}_Q"] = traj["K"][j], traj["Q"][j] / traj["Q"][0]
        R[f"{key}_u"], R[f"{key}_T"] = traj["u"][j], traj["T"][j]
        R[f"{key}_cross"] = crossing_year(traj)
    R["K2050_saved"] = R["auto_K"] / nr["K"][0]
    R["omega_A_below_star"] = float(p.omega_A < p.delta / 8)
    return R


def summarise(draws):
    keys = [
        "pstar",
        "pstar_phi",
        "inv2050",
        "K2050",
        "Q2050",
        "u2050",
        "tau_star",
        "tpol",
        "lost0",
        "lost5",
        "lost10",
        "lost15",
        "K2050_saved",
    ] + [f"{s}_{v}" for s in SOCIETY_KEYS for v in ["K", "Q", "u", "T", "cross"]]
    summary = {}
    for k in keys:
        v = np.array([o[k] for o in draws if k in o], dtype=float)
        finite = v[np.isfinite(v)]
        summary[k] = dict(
            p5=float(np.percentile(finite, 5)) if finite.size else None,
            p50=float(np.median(finite)) if finite.size else None,
            p95=float(np.percentile(finite, 95)) if finite.size else None,
            min=float(finite.min()) if finite.size else None,
            max=float(finite.max()) if finite.size else None,
            n=int(finite.size),
            n_missing=int(v.size - finite.size),
        )
    with_policy = [o for o in draws if "tpol" in o]
    late = [o for o in with_policy if o["tpol"] + o["d"] >= 2043]
    early = [o for o in with_policy if o["tpol"] + o["d"] < 2043]
    checks = dict(
        fold_before_react=np.mean([o["fold_before_react"] for o in draws]),
        fold_before_react_phi=np.mean([o["fold_before_react_phi"] for o in draws]),
        invention_below_three_quarters=np.mean([o["inv2050"] < 0.75 for o in draws]),
        expertise_below_0_6=np.mean([o["K2050"] < 0.6 for o in draws]),
        delay_increasing=np.mean([o["delay_increasing"] for o in with_policy]),
        second_five_years_cost_more=np.mean(
            [o["lost10"] - o["lost5"] > o["lost5"] - o["lost0"] for o in with_policy]
        ),
        n_policy=len(with_policy),
        n_policy_before_2043=len(early),
        early_policy_restores=np.mean([o["K2050_saved"] > 0.9 for o in early]),
        n_policy_after_2043=len(late),
        n_late_policy_restored_by_2050=sum(o["K2050_saved"] > 0.98 for o in late),
        n_omega_A_below_star=sum(o["omega_A_below_star"] for o in draws),
        n_none_crosses_by2100=sum(np.isfinite(o["none_cross"]) for o in draws),
        n_over_crosses_by2100=sum(np.isfinite(o["over_cross"]) for o in draws),
        n_auto_crosses_by2100=sum(np.isfinite(o["auto_cross"]) for o in draws),
        n_hollow_crosses_by2100=sum(np.isfinite(o["hollow_cross"]) for o in draws),
        n_rent_crosses_by2100=sum(np.isfinite(o["rent_cross"]) for o in draws),
        auto_cooler_than_over=np.mean([o["auto_T"] < o["over_T"] for o in draws]),
    )
    return summary, {k: float(v) for k, v in checks.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=300)
    args = parser.parse_args()
    rng = np.random.default_rng(2026)
    draws = []
    for n in range(args.draws):
        draw = {k: rng.uniform(*bounds) for k, bounds in RANGES.items()}
        draws.append({**draw, **evaluate(draw)})
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{args.draws} draws")
    summary, checks = summarise(draws)
    for k, s in summary.items():
        if s["n"]:
            print(
                f"{k:12s} range [{s['min']:.3f}, {s['max']:.3f}]  median {s['p50']:.3f}  missing {s['n_missing']}"
            )
    print(checks)
    out = ROOT / "results" / "sensitivity.json"
    out.write_text(
        json.dumps(dict(summary=summary, checks=checks, ranges=RANGES, draws=draws), indent=1, default=float)
    )
    print(f"written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
