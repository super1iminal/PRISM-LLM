"""Asher's UUV plan, step 3's sanity check (run by cluster/checklist.sh item8-repick, from PRISM-Guided-Learning/):

- runs 1 and 2 (no `any` rules in item 8): the re-pick's NoRepair list (norepair.json, resolved) equals item 8's
  stored list (out/results/prepick/port_uuv/<run>/determinized.json), and its checks equal item 8's (outcome, solver
  and worst values per target);
- run 3 may differ (its `any` rules are now resolved by the fast checker): reported, not an error;
- no check of the re-picked runs is inconclusive, timed out, an error, or decided by the Gauss-Seidel fallback.
Exits 1 on any failure, so the job reads as failed.
"""
import json
import sys
from pathlib import Path

import pandas as pd

OLD, NEW = Path("out/results/prepick/port_uuv"), Path("out/results/core/port_uuv")
STEM = "transfer_exact_uuv_family_v0"
WORST = ["worst_no_thruster_failure", "worst_done_in_time", "worst_energy_budget"]
problems = []

for k in (1, 2, 3):
    run = f"seed_none_{k}"
    stored = json.loads((OLD / run / "determinized.json").read_text())["0"]["rules"]
    norepair = json.loads((NEW / run / "norepair.json").read_text())["0"]["determinized"]
    same = stored == norepair
    print(f"{run}: NoRepair list == item 8's list: {same}")
    if k in (1, 2) and not same:
        problems.append(f"{run}: the NoRepair list differs from item 8's")
    old = pd.read_csv(OLD / run / f"{STEM}_determinized.csv").set_index("target").sort_index()
    new = pd.read_csv(NEW / run / f"{STEM}_norepair_determinized.csv").set_index("target").sort_index()
    if k in (1, 2):
        if list(old.index) != list(new.index):
            problems.append(f"{run}: NoRepair checks other targets than item 8's")
        else:
            outcome = (old.outcome != new.outcome).sum()
            solver = (old.solver != new.solver).sum()
            worst = (old[WORST] - new[WORST]).abs().max().max()
            print(f"  checks vs item 8: {outcome} outcomes and {solver} solvers differ, largest worst-value "
                  f"difference {worst:.2e}")
            if outcome or solver or worst > 1e-9:
                problems.append(f"{run}: NoRepair checks differ from item 8's")
    for path in sorted((NEW / run).glob(f"{STEM}*.csv")):
        df = pd.read_csv(path)
        bad = df[~df.outcome.isin(["certified", "failed"]) | df.fallback.astype(bool)
                 | (df.solver != "interval iteration") & (df.solver != "linear programming")]
        print(f"  {path.name}: {df.outcome.value_counts().to_dict()}, solvers {df.solver.value_counts().to_dict()}")
        if len(bad):
            problems.append(f"{path}: {len(bad)} checks inconclusive, timed out or by the fallback")

print("\n".join(["PROBLEMS:"] + problems) if problems else "step 3 sanity check passed")
sys.exit(1 if problems else 0)
