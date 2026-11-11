# Collapsing pyramids — model and results

Code behind *Collapsing pyramids*, an essay on what generative AI does to the base of professions: the
junior band where beginners become experts, the invention of new tasks, the skills of those who use AI,
and the two loops of income and carbon that decide which society we reach by 2050.

Paper: arXiv link to come.
Author: Nicolas Petitmangin.

Every number, table and figure of the paper is computed by the scripts below from the `pyramids`
package, and the tests check that they still are.

## Contents

```
pyramids/                 the model, as a small Python package
├── parameters.py         professions and every parameter (Params), with the paper's notation
├── model.py              the equations and the simulation: simulate(params) -> Trajectory
├── scenarios.py          policy levers and the five societies of Table 1
└── theory.py             closed-form results (Propositions 1 to 5)
scripts/
├── compute_numbers.py    -> results/numbers.json      every number quoted in the paper (~15 s)
├── run_sensitivity.py    -> results/sensitivity.json  Appendix A.8, 300 joint draws (~5 min)
└── make_figure_data.py   -> figures/data.json         data behind the figures
results/                  outputs of the scripts, as used in the paper
figures/
├── template.html         the figures, drawn with d3 from data.json
├── export_figures.js     renders each figure to figures/png/ (Node + Playwright)
├── png/                  Figures 1 to 6 and 8
└── static/               Figure 7 (the two loops), drawn by hand
tests/                    the code reproduces the paper (reference_numbers.json) and its propositions
```

## Reproduce

Python 3.10 or later.

```
pip install -r requirements.txt
python scripts/compute_numbers.py
python scripts/run_sensitivity.py
python scripts/make_figure_data.py
python -m pytest tests
```

Figures (optional, needs Node):

```
npm install playwright && npx playwright install chromium
node figures/export_figures.js
```

## Using the model

```python
from pyramids import REFERENCE, simulate, society

traj = simulate(REFERENCE.with_(react=False), years=24)     # 2026 to 2050, no policy
traj.at("K", 2050)                                          # average expertise in 2050
simulate(society("Autonomous society"), years=24).at("u", 2050)
```

`simulate` returns a `Trajectory`: one array per series (see `pyramids.model.SERIES_DOC`), with `t` in
calendar years and `t_policy` the date the policy takes effect.

The model simulates mechanisms under stated assumptions; it is not a forecast.

## Licence

Code under the MIT licence (see `LICENSE`). If you use it, please cite the paper (`CITATION.cff`).
