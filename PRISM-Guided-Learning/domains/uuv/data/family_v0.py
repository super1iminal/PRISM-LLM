"""The UUV family of Rajan's checklist, item 8 (`uuv_family_v0.csv`): members that differ in structure, not only in
probabilities, so that no fixed simple controller covers them (positioning Check 2 found the failure-rate family
trivial). PRISM only, no LLM.

Members: the North Sea dynamics (current probability 0.6, min_visib 1) with
- the visibility range W = max_visib - min_visib in 7..12, which moves the band edges (`med_visib = W/3`,
  `high_visib = 2W/3`) and so where each altitude is allowed;
- the inspection length in 6, 8, 10, 12, 14 m (the range of `d_insp`);
- the deadline 3 x the inspection length (North Sea: 10 m in 30 steps).

Thresholds, per member and requirement: the best value over all controllers moved towards the worst by the fraction
SIGMA = 0.1 of the range between them (`threshold = best - 0.1 (best - worst)`; for energy, best + 0.1 (worst - best)).
The paper's own thresholds sit at 6-20% of these ranges. With the other domains' "optimum minus 5%" every controller
would pass `no_thruster_failure` here, whose whole range is about 0.02. Bounds come from the exact solvers (interval
iteration, LP where it stalls), on the bare model.

Splits (column `split`): three small members for training (`train`: W 7, 8, 9 with inspection 6, 8, 6 m; the ranges
where `stay`, `always_med` and `always_high` fail respectively), the other 27 for validation (`validation`). The
sealed test members (longer inspections) are not generated here: building them is the test pass's (rule 3).

Run from PRISM-Guided-Learning/:  python domains/uuv/data/family_v0.py   (writes uuv_family_v0.csv next to it)
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from config import load_config  # noqa: E402
from core.domain import load_domain  # noqa: E402
from core.verifier import PolicyVerifier  # noqa: E402

SIGMA = 0.1
WIDTHS = [7, 8, 9, 10, 11, 12]
INSPECT = [6, 8, 10, 12, 14]
TRAIN = {(7, 6), (8, 8), (9, 6)}


def members():
    for w in WIDTHS:
        for i in INSPECT:
            yield {"name": f"w{w}_i{i}_d{3 * i}", "min_visib": 1, "max_visib": 1 + w, "current_prob": 0.6,
                   "inspect": i, "deadline": 3 * i, "split": "train" if (w, i) in TRAIN else "validation"}


def bounds(domain, instance, cfg):
    """Each requirement's best and worst value over all controllers (exact solvers, bare model)."""
    v = PolicyVerifier(domain, instance, cfg)
    res, solver = v.verify_exact(v.empty_policy())
    return {r.name: (res.best[r.name], res.worst[r.name], r.bound) for r in v.spec.requirements}, solver


def main():
    cfg = load_config("U1")
    domain = load_domain("uuv")
    rows = list(members())
    tmp = Path(tempfile.mkdtemp()) / "family.csv"
    # thresholds the bounds do not depend on; the energy requirement is on (a threshold is set)
    pd.DataFrame([{**r, "safe_threshold": 0.0, "on_time_threshold": 0.0, "energy_threshold": 1e9} for r in rows]
                 ).to_csv(tmp, index=False)
    out = []
    for row, inst in zip(rows, domain.load_instances(str(tmp))):
        b, solver = bounds(domain, inst, cfg)
        th = {name: best - SIGMA * (best - worst) for name, (best, worst, _) in b.items()}
        out.append({**row, "safe_threshold": round(th["no_thruster_failure"], 6),
                    "on_time_threshold": round(th["done_in_time"], 6), "energy_threshold": round(th["energy_budget"], 6),
                    **{f"best_{k}": v[0] for k, v in b.items()}, **{f"worst_{k}": v[1] for k, v in b.items()},
                    "bounds_solver": solver})
        print(out[-1]["name"], out[-1]["split"], {k: out[-1][k] for k in ("safe_threshold", "on_time_threshold",
                                                                          "energy_threshold")}, solver, flush=True)
    path = Path(__file__).with_name("uuv_family_v0.csv")
    pd.DataFrame(out).to_csv(path, index=False)
    print(path)


if __name__ == "__main__":
    main()
