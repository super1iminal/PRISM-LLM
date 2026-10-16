"""One row per gridworld list (Haiku runs; chain runs left out): training values and held-out certification, both
with the list's action orders resolved (as frozen) and with its ties left open (core.rules.open_ties: the worst case
over every action an `any` rule allows). Open training values: numpy worst case on each member (tie_analysis);
held-out: the runs' PRISM transfer CSVs (`transfer_*_open.csv`; for runs that keep `any`, the final CSV is already
the worst case). Usage: python open_table.py <out.csv>"""
import json
import re
import os
import sys
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
ROOT = Path(__file__).resolve().parents[4]          # PRISM-Guided-Learning/
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import pandas as pd

from baselines.gridworld_mdp import ACTIONS, GridMDP, RuleScorer
from config import load_run_config
from core.domain import load_domain
from core.rules import ANY, open_ties
from results_io import sample_records
from tie_analysis import allowed, extreme
from transfer import parse_ids

TARGETS = {"grid_20_large.csv": {None: "12-19"}, "families_v0.csv": {0: "16-35", 1: "76-95", 2: "136-155"}}
# grid ablations that resolve `any`: their open values cost ~30 s per list; the resolved analysis covers them
SLOW = re.compile(r"ablations/haiku_ts_(order\+|order_r8|trace|grow|hidden|declared)")
GROUPS = {0: "rooms", 1: "crossing", 2: "adjacent"}


def worst(mdp, allow):
    """Worst case of every requirement when any allowed action may be taken (tie_analysis.worst_best, worst only)."""
    out = {}
    for name, (mon, safety) in mdp.monitors().items():
        out[name] = (1 - extreme(mdp, allow, mon, mon.reject, minimize=False, tol=1e-10) if safety
                     else extreme(mdp, allow, mon, mon.accept, minimize=True, tol=1e-10))
    return out


def allow_of(domain, cfg, inst, rules, mdp):
    try:
        return allowed(mdp, rules)
    except KeyError:                       # conditions beyond feature literals: the core rule evaluator
        spec = domain.spec(inst)
        pol = spec.policy(rules, spec.vocabulary(cfg.rules))
        allow = np.zeros((mdp.S, 4), dtype=bool)
        for i, state in enumerate(RuleScorer(inst.data, spec.variables).valuations):
            acts = pol.allowed(state)
            allow[i, [ACTIONS.index(a) for a in acts] if acts is not None else slice(None)] = True
        return allow


def stats(values, reqs):
    """(members passing, total shortfall, smallest margin, smallest margin on the largest member)."""
    passed, short, margins, per = 0, 0.0, [], []
    for vals, rs in zip(values, reqs):
        m = [r.margin(vals[r.name]) for r in rs]
        passed += min(m) >= 0
        short += sum(r.shortfall(vals[r.name]) for r in rs)
        margins += m
        per.append(min(m))
    return passed, short, min(margins), per[-1]


def certified(csv: Path, sid, targets):
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    sub = df[(df.source == sid) & df.target.isin(targets) & ~df.trained]
    if sub.target.nunique() < len(targets):
        return None
    return int(sub.certified.fillna(False).astype(bool).sum())


def rows_of(run: Path):
    cfg = load_run_config(run)
    ds = cfg.domain.dataset
    if ds not in TARGETS or "haiku" not in cfg.llm.model or cfg.planner.chains > 1:
        return []
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    instances = {int(i.id): i for i in domain.load_instances(ds)}
    stem = Path(ds).stem
    resolves = cfg.rules.any_order in ("all", "rotations", "declared")
    out = []
    for sid, rec in sample_records(run).items():
        if "final_rules" not in rec or "members" not in rec:
            continue
        group = TARGETS[ds]
        members = [int(m["instance"]) for m in rec["members"]]
        targets = [t for t in parse_ids(group[sid] if sid in group else group[None]) if t not in members]
        rules = [(r["condition"], r["action"]) for r in rec["final_rules"]]
        opened = open_ties(rules, ACTIONS)
        reqs = [domain.spec(instances[i]).requirements for i in members]
        resolved_vals = [m["final_worst"] for m in rec["members"]]
        open_vals = resolved_vals
        if resolves:
            mdps = [GridMDP(instances[i].data) for i in members]
            open_vals = [worst(mdp, allow_of(domain, cfg, instances[i], opened, mdp)) for i, mdp in zip(members, mdps)]
        rp, rs, rm, rl = stats(resolved_vals, reqs)
        op, os_, om, ol = stats(open_vals, reqs)
        bench = ("families/" + GROUPS[sid]) if ds == "families_v0.csv" else \
            ("grids_noclose" if "closer" in cfg.rules.hidden_features else "grids")
        out.append({"run": run.relative_to(ROOT / "out/results").as_posix(), "sid": sid, "condition": cfg.condition,
                    "bench": bench, "any_order": cfg.rules.any_order, "num_rules": len(rules),
                    "num_any": sum(a == ANY for _, a in opened),
                    "res_passed": rp, "res_short": rs, "res_margin": rm, "res_large": rl,
                    "open_passed": op, "open_short": os_, "open_margin": om, "open_large": ol,
                    "cert_resolved": certified(run / f"transfer_{stem}.csv", sid, targets),
                    "cert_open": certified(run / (f"transfer_{stem}_open.csv" if resolves else f"transfer_{stem}.csv"),
                                           sid, targets),
                    "n_targets": len(targets), "members": len(members)})
    return out


def main(out_path):
    out_path = Path(out_path)
    rows = []
    done = set()
    if out_path.exists():
        old = pd.read_csv(out_path)
        rows = old.to_dict("records")
        done = set(old.run)
    runs = sorted(list((ROOT / "out/results/core").glob("*/seed*/config.json")) +
                  list((ROOT / "out/results/ablations").glob("*/seed*/config.json")))
    for k, cfgp in enumerate(runs):
        run = cfgp.parent
        if run.relative_to(ROOT / "out/results").as_posix() in done or SLOW.search(run.as_posix()):
            continue
        try:
            new = rows_of(run)
        except Exception as e:                 # old configs that no longer load, etc.
            print("skip", run.name, type(e).__name__, str(e)[:80], flush=True)
            continue
        if new:
            rows += new
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(k, run.relative_to(ROOT / "out/results").as_posix(), len(new), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
