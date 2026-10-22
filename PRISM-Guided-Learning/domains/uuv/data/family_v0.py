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
where `stay`, `always_med` and `always_high` fail respectively), the other 27 for validation (`validation`).

Appended later, by the same rule, without touching the rows before them (Asher's UUV plan, Oct 22):
- `--append test`: ids 30-41, the sealed test members, W 7-12 at inspection 16 and 18 m (`split` test; sealed in
  configs/transfer/default.yaml). Only their bare models' bounds are computed here.
- `--append sets`: ids 42-47, training-only members for two more training sets (`split` train): W 8, 10, 12 at
  inspection 9 m (set 1) and W 7, 9, 11 at inspection 11 m (set 2), inspection lengths the validation grid lacks.

Run from PRISM-Guided-Learning/:
    python domains/uuv/data/family_v0.py                  writes rows 0-29 (uuv_family_v0.csv next to it)
    python domains/uuv/data/family_v0.py --append test    appends ids 30-41
    python domains/uuv/data/family_v0.py --append sets    appends ids 42-47
"""
import argparse
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


TEST_INSPECT = [16, 18]
EXTRA_SETS = [(8, 9), (10, 9), (12, 9), (7, 11), (9, 11), (11, 11)]     # (W, inspection): set 1, then set 2


def member(w: int, i: int, split: str) -> dict:
    return {"name": f"w{w}_i{i}_d{3 * i}", "min_visib": 1, "max_visib": 1 + w, "current_prob": 0.6,
            "inspect": i, "deadline": 3 * i, "split": split}


def members():
    for w in WIDTHS:
        for i in INSPECT:
            yield member(w, i, "train" if (w, i) in TRAIN else "validation")


def appended(group: str):
    if group == "test":
        return [member(w, i, "test") for w in WIDTHS for i in TEST_INSPECT]
    if group == "sets":
        return [member(w, i, "train") for w, i in EXTRA_SETS]
    raise ValueError(group)


def bounds(domain, instance, cfg):
    """Each requirement's best and worst value over all controllers (exact solvers, bare model)."""
    v = PolicyVerifier(domain, instance, cfg)
    res, solver = v.verify_exact(v.empty_policy())
    return {r.name: (res.best[r.name], res.worst[r.name], r.bound) for r in v.spec.requirements}, solver


def with_thresholds(rows, cfg, domain):
    """`rows` with each requirement's threshold from its exact bounds on the bare model (SIGMA of the range)."""
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
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--append", choices=["test", "sets"], default=None)
    a = p.parse_args()
    cfg = load_config("U1")
    domain = load_domain("uuv")
    path = Path(__file__).with_name("uuv_family_v0.csv")
    if a.append is None:
        pd.DataFrame(with_thresholds(list(members()), cfg, domain)).to_csv(path, index=False)
        print(path)
        return
    existing = pd.read_csv(path)
    new = [r for r in appended(a.append) if r["name"] not in set(existing.name)]
    if not new:
        print("nothing to append")
        return
    out = pd.DataFrame(with_thresholds(new, cfg, domain))[existing.columns]
    if (out.bounds_solver != "interval iteration").any():
        raise SystemExit(f"bounds not from interval iteration: {out[out.bounds_solver != 'interval iteration'].name.tolist()}")
    with open(path, "a", encoding="utf-8", newline="") as f:            # rows before stay byte-identical
        out.to_csv(f, index=False, header=False, lineterminator="\n")
    print(f"{path}: appended {len(out)} rows (ids {len(existing)}-{len(existing) + len(out) - 1})")


if __name__ == "__main__":
    main()
