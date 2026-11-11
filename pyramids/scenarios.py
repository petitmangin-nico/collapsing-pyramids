"""Policy levers and the five societies of Table 1."""

from .parameters import REFERENCE, Params

# Levers pulled once the coalition has formed.
KNOWLEDGE = dict(z=1.0, h_pol=0.6)  # base given back, AI used as a tutor
PLANET = dict(mu_pol=0.5, s_c_pol=0.9, lam_AI=0.02)  # emission cuts, clean research, AI boost

SOCIETIES = {
    "World without AI": dict(react=False, g_H=0.0, chi=0.0),
    "Hollow economy": dict(react=False),
    "Concentrated wealth": dict(**KNOWLEDGE, **PLANET, tau=0.0),
    "Autonomous society": dict(**KNOWLEDGE, **PLANET, tau=0.6),
    "Overshoot": dict(**KNOWLEDGE, tau=0.6),
}

FULL_POLICY = dict(**KNOWLEDGE, **PLANET, tau=0.6)  # every lever, used for the cost of delay


def society(name: str, base: Params = REFERENCE) -> Params:
    """Parameters of one of the five societies."""
    return base.with_(**SOCIETIES[name])


NO_AI_PROGRESS = 1.0  # capability gained over the run below which AI is said to have stalled (junior bands)
HOLLOW_EXPERTISE = 0.5  # average expertise below which the economy is hollow
SOLD_SHARE_RENT = 0.95  # share of output sold below which wealth stays concentrated


def classify(traj) -> str:
    """Which society a trajectory ends in. Follow it to 2100 (years=74): overshoot shows only after 2050."""
    if traj["c"].max() - traj["c"][0] < NO_AI_PROGRESS:
        return "World without AI"
    if traj.at("K", 2050) < HOLLOW_EXPERTISE:
        return "Hollow economy"
    if traj["T"].max() > 2.0:
        return "Overshoot"
    if traj.at("u", 2050) < SOLD_SHARE_RENT:
        return "Concentrated wealth"
    return "Autonomous society"
