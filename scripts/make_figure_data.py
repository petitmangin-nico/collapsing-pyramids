"""Data behind Figures 1 to 5 and 8, written to figures/data.json for figures/template.html.

Run from the repository root:  python scripts/make_figure_data.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pyramids import REFERENCE, WEIGHTS, simulate  # noqa: E402
from pyramids import theory  # noqa: E402

HORIZON, DT, EVERY = 24, 0.02, 25  # one point every half year
DEV = 1


def invention_weighted_expertise(k, p):
    """K-tilde of Proposition 2: expertise weighted by w_i (1 + sigma (1 - w_i)), weights summing to one."""
    w = WEIGHTS * (1 + p.sigma * (1 - WEIGHTS))
    return float(w @ k / w.sum())


def rounded(values):
    return [round(float(v), 4) for v in values]


def main():
    p = REFERENCE
    nr = simulate(p.with_(react=False), years=HORIZON, dt=DT)
    tutor = simulate(p.with_(react=False, h=0.6), years=HORIZON, dt=DT)
    no_self_teaching = simulate(p.with_(react=False, lam_A=0.0), years=HORIZON, dt=DT)
    t = nr["t"]
    idx = np.arange(0, len(t), EVERY)

    def relative(traj, name):
        return traj[name] / traj[name][0]

    def price(traj, h):
        """Share of the junior band a policy must restore, q_omega / s-bar (Figure 5), in %."""
        omega = p.lam_A * h * traj["sH"]
        return theory.remedy_price(omega, p) / traj["s"] * 100

    D = {"t": rounded(t[idx]), "ELL": p.ell}
    # Figure 1: the development pyramid
    D["pyrT"] = [
        dict(
            g=round(float(-p.ell * np.log(nr["L"][i, DEV])), 4),
            lost=round(float(1 - nr["L"][i, DEV]), 4),
            q=round(float(nr["q"][i, DEV]), 4),
        )
        for i in idx
    ]
    # Figure 3: invention, expertise, employment
    D.update(
        inv=rounded(relative(nr, "Xdot")[idx]),
        K=rounded(relative(nr, "K")[idx]),
        Q=rounded(relative(nr, "Q")[idx]),
        invT=rounded(relative(tutor, "Xdot")[idx]),
        inv0=rounded(relative(no_self_teaching, "Xdot")[idx]),
        floor=float(theory.self_taught_floor(p.omega_A) / invention_weighted_expertise(nr["k"][0], p)),
    )
    # Figure 4: skills
    D.update(sH=rounded(nr["sH"][idx]), sL=rounded(nr["sL"][idx]), s=rounded(nr["s"][idx]))
    # Figure 5: price of the remedy
    D.update(pToday=rounded(price(nr, p.h)[idx]), pTutor=rounded(price(tutor, 0.6)[idx]))
    # Figure 8: output sold and production lost before the policy
    y = nr["Yr"] / nr["Yr"][0]
    policy_date = simulate(p.with_(d=0.0), years=HORIZON, dt=DT)["t_policy"]
    D["Yr"] = rounded(y[idx])
    D["delays"] = []
    for d in [0, 5, 10, 15]:
        tp = policy_date + d
        lost = float(np.sum(np.clip(1 - y[t <= tp], 0, None)) * DT)
        D["delays"].append(dict(d=d, tp=float(tp), lost=round(lost, 3)))

    out = ROOT / "figures" / "data.json"
    out.write_text(json.dumps(D))
    print(f"written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
