"""One row per final list of every Haiku run that resolves `any` in the loop: benchmark, condition, training values
(exact check), the keep-best score, margins, and held-out certification from the run's transfer CSV (PRISM)."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]          # PRISM-Guided-Learning/
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import pandas as pd

from config import load_run_config
from core.domain import load_domain
from results_io import sample_records
from transfer import parse_ids

BENCH = {"grid_20_large.csv": ("transfer_grid_20_large.csv", {None: "12-19"}),
         "families_v0.csv": ("transfer_families_v0.csv", {0: "16-35", 1: "76-95", 2: "136-155"}),
         "pacman_transfer.csv": ("transfer_pacman_transfer.csv", {None: "0-7,9"}),
         "mer_transfer.csv": ("transfer_mer_transfer.csv", {None: "0-8"}),
         "mer_n_transfer.csv": ("transfer_mer_n_transfer.csv", {None: "0-8"})}
GROUPS = {0: "rooms", 1: "crossing", 2: "adjacent"}


def name(dataset, sid, hidden):
    if dataset == "families_v0.csv":
        return "families/" + GROUPS[sid]
    if dataset == "grid_20_large.csv":
        return "grids_noclose" if "closer" in hidden else "grids"
    return {"pacman_transfer.csv": "pacman", "mer_transfer.csv": "mer", "mer_n_transfer.csv": "mer_n"}[dataset]


def rows_of(run: Path, suffix=""):
    cfg = load_run_config(run)
    if cfg.rules.any_order not in ("all", "rotations") or "haiku" not in cfg.llm.model:
        return []
    ds = cfg.domain.dataset
    if ds not in BENCH:
        return []
    csv = run / BENCH[ds][0].replace(".csv", f"{suffix}.csv")
    if not csv.exists():
        return []
    df = pd.read_csv(csv)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    instances = {int(i.id): i for i in domain.load_instances(ds)}
    out = []
    for sid, rec in sample_records(run).items():
        if "final_rules" not in rec or "members" not in rec:
            continue
        groups = BENCH[ds][1]
        targets = parse_ids(groups[sid] if sid in groups else groups[None])
        sub = df[(df.source == sid) & df.target.isin(targets) & ~df.trained]
        if sub.target.nunique() < len(set(targets) - {int(m["instance"]) for m in rec["members"]}):
            continue
        margins, fails, short = [], 0, 0.0
        per_member = []
        for m in rec["members"]:
            spec = domain.spec(instances[int(m["instance"])])
            mm = []
            for r in spec.requirements:
                v = m["final_worst"][r.name]
                g = (v - r.threshold if r.maximize else r.threshold - v) / (abs(r.threshold) if r.reward else 1)
                mm.append(g)
                fails += not r.satisfied(v)
                short += r.shortfall(v)
            per_member.append(min(mm))
            margins += mm
        out.append({"run": run.relative_to(ROOT / "out/results").as_posix(), "sid": sid, "condition": cfg.condition,
                    "bench": name(ds, sid, cfg.rules.hidden_features), "success": rec["success"],
                    "fails": fails, "shortfall": short, "min_margin": min(margins), "sum_margin": sum(margins),
                    "large_margin": per_member[len([m for m in rec["members"] if not m.get("hidden")]) - 1],
                    "num_rules": len(rec["final_rules"]),
                    "certified": int(sub.certified.fillna(False).astype(bool).sum()),
                    "n_targets": int(sub.target.nunique()),
                    "tokens_out": sum(it["llm_output_tokens"] for it in rec["iterations"]),
                    "rounds": len(rec["iterations"])})
    return out


def main(out_path):
    rows = []
    for cfgp in sorted(list((ROOT / "out/results/core").glob("*/seed*/config.json")) +
                       list((ROOT / "out/results/ablations").glob("*/seed*/config.json"))):
        try:
            rows += rows_of(cfgp.parent)
        except Exception as e:
            print("skip", cfgp.parent, type(e).__name__, e)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(len(rows), "lists ->", out_path)


if __name__ == "__main__":
    main(sys.argv[1])
