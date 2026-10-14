"""Rule lists for the transfer diagnosis (docs/transfer_plan.md, Phase 1): collect the final rule sets of runs, and
rewrite their `any` rules into a fixed direction order. The lists are then certified with src/run_baseline.py
(method `rule_lists`).

Usage (from PRISM-Guided-Learning/):
    python src/diagnose_rules.py orders <run dir glob> ... --out <file.yaml> [--orders down,right,up,left ...]
Writes {name: [[condition, action], ...]} with, per distinct rule set, the original (`<name>@any`) and one list per
order (`<name>@<order>`), where each `c -> any` becomes `c -> d1`, `c -> d2`, ... in that order: the first action
in the order that the rule allows.
"""
import argparse
import glob
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml

from results_io import sample_records

Rules = List[Tuple[str, str]]
ROTATIONS = [["down", "right", "up", "left"], ["right", "up", "left", "down"],
             ["up", "left", "down", "right"], ["left", "down", "right", "up"]]


def collect(run_dirs: Sequence[Path]) -> Dict[str, Rules]:
    """Distinct final rule sets of the runs, named <condition>/<run>/<sample id> after their first occurrence."""
    found: Dict[Tuple, str] = {}
    for run in run_dirs:
        for sid, record in sample_records(run).items():
            if "final_rules" in record:
                rules = tuple((r["condition"], r["action"]) for r in record["final_rules"])
                found.setdefault(rules, f"{run.parent.name}/{run.name}/{sid}")
    return {name: list(rules) for rules, name in found.items()}


def fixed_order(rules: Rules, order: Sequence[str]) -> Rules:
    """`rules` with each `c -> any` replaced by `c -> d` for every action d in `order`."""
    return [(c, d) for c, a in rules for d in (order if a == "any" else [a])]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    orders = sub.add_parser("orders", help="original rule sets and their fixed-order rewrites")
    orders.add_argument("runs", nargs="+", help="run directories or globs")
    orders.add_argument("--out", type=Path, required=True)
    orders.add_argument("--orders", nargs="*", default=None, help="comma-separated orders (default: 4 rotations)")
    args = parser.parse_args(argv)

    runs = sorted(Path(p) for pattern in args.runs for p in glob.glob(pattern) if (Path(p) / "outputs").is_dir())
    rule_sets = collect(runs)
    order_list = [o.split(",") for o in args.orders] if args.orders else ROTATIONS
    lists = {}
    for name, rules in rule_sets.items():
        lists[f"{name}@any"] = [list(r) for r in rules]
        if any(a == "any" for _, a in rules):
            for order in order_list:
                lists[f"{name}@{'-'.join(order)}"] = [list(r) for r in fixed_order(rules, order)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(lists, sort_keys=False, width=200), encoding="utf-8")
    print(f"{len(rule_sets)} rule sets from {len(runs)} runs -> {len(lists)} lists in {args.out}")


if __name__ == "__main__":
    main()
