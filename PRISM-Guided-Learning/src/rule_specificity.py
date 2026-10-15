"""Instance-specific rules per kind of feedback (plan 04, step 5): how many conditions of the runs' rule lists single
out a place or an instance, read with the rule parser, and whether the next answer rewrites the rules the feedback
blamed more often than the others.

A condition is counted once per list (by its parsed form, so spacing and the per-action copies that `any` resolution
writes count once) under each kind it contains:
  * position: a position variable compared with a constant or with another position (`y = G3C`, `x > G1R`, `xG0 = xP`);
  * offset: a constant plus or minus a number (`G2C - 1`, `N - 1`);
  * constants: a comparison of constants only (`G2R = G3R + 1`).
Comparisons with a number alone (`x = 0`, a border) are none of these; conditions that read a position at all are
counted separately (`positions`). Positions are gridworld's `x`, `y`, `obs_idx`
and Pac-Man's `xP`, `yP`, `xG0`, `yG0`, `xG1`, `yG1`; constants are the domain's instance constants (`N`, `G1R`,
`MAXSTEPS`).

Lists: each training set's final list, and every round's candidate from round 2 on (the rounds that follow feedback).
Blamed rules: in each REFINE prompt, the listed rules the blame section names; a listed rule counts as changed when
its condition is not among the conditions of the answer to that prompt.

Usage (from PRISM-Guided-Learning/):
    python src/rule_specificity.py configs/experiments/rq2_verifier_output.yaml [more experiment files]
        [--cells NAME ...] [--out out/results/experiments/rq2_verifier_output/instance_specific]
Writes <out>.csv (one row per run) and <out>.md (one table per benchmark: means over runs of the per-list means).
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from core.domain import load_domain
from core.rules import Binary, Const, Expr, RuleError, Unary, Var, _Parser
from experiments.base import experiment
from experiments.store import recorded_config
from results_io import sample_records

POSITIONS = {"gridworld": {"x", "y", "obs_idx"}, "pacman": {"xP", "yP", "xG0", "yG0", "xG1", "yG1"}}
COMPARISONS = {"=", "!=", "<", "<=", ">", ">="}
KINDS = ("position", "offset", "constants")
BLAME = re.compile(r"^- \d+%: rule (\d+) `", re.M)
LISTED = re.compile(r"^\d+\. (\{.*\})$", re.M)


def parse(condition: str) -> Optional[Expr]:
    """The condition's syntax tree, names unresolved (constants and features stay names); None if it does not parse."""
    try:
        return _Parser(condition).parse()
    except RuleError:
        return None


def names(expr: Expr) -> Set[str]:
    if isinstance(expr, Var):
        return {expr.name}
    if isinstance(expr, Unary):
        return names(expr.arg)
    if isinstance(expr, Binary):
        return names(expr.left) | names(expr.right)
    return set()


def kinds(expr: Expr, positions: Set[str], constants: Set[str]) -> Set[str]:
    """Which of KINDS the condition contains."""
    found: Set[str] = set()

    def walk(e: Expr) -> None:
        if isinstance(e, Unary):
            walk(e.arg)
        if not isinstance(e, Binary):
            return
        left, right = names(e.left), names(e.right)
        if e.op in COMPARISONS:
            places = positions | constants
            if (left & positions and right & places) or (right & positions and left & places):
                found.add("position")
            if (left | right) and (left | right) <= constants:
                found.add("constants")
        if e.op in ("+", "-") and any(isinstance(a, Var) and a.name in constants and isinstance(b, Const)
                                      for a, b in ((e.left, e.right), (e.right, e.left))):
            found.add("offset")
        walk(e.left)
        walk(e.right)

    walk(expr)
    return found


def list_counts(conditions: Iterable[str], positions: Set[str], constants: Set[str]) -> Dict[str, int]:
    """Distinct conditions of one list, and how many contain each kind (and any of them)."""
    trees = {parse(c) or c for c in conditions}
    counts = {"conditions": len(trees), "positions": 0, "specific": 0, **{k: 0 for k in KINDS}}
    for tree in trees:
        if isinstance(tree, str):
            continue
        found = kinds(tree, positions, constants)
        for k in found:
            counts[k] += 1
        counts["specific"] += bool(found)
        counts["positions"] += bool(names(tree) & positions)
    return counts


def blame_follow(prompt: str, answer: List[str]) -> Optional[Tuple[int, int, int, int]]:
    """(blamed changed, blamed, others changed, others) for one REFINE prompt and the conditions of its answer."""
    if "## Where probability is lost" not in prompt:
        return None
    shown = prompt[prompt.index("## Your current rules"):]
    shown = shown[:shown.index("\n## ", 1)]
    listed = [json.loads(m)["condition"] for m in LISTED.findall(shown)]
    section = prompt[prompt.index("## Where probability is lost"):]
    blamed = {int(n) - 1 for n in BLAME.findall(section)}
    kept = {parse(c) or c for c in answer}
    out = [0, 0, 0, 0]
    for i, cond in enumerate(listed):
        changed = (parse(cond) or cond) not in kept
        slot = 0 if i in blamed else 2
        out[slot] += changed
        out[slot + 1] += 1
    return tuple(out)


_CONSTANTS: Dict[Tuple[str, str], Set[str]] = {}


def constant_names(domain_name: str, dataset: str) -> Set[str]:
    """Names of the domain's instance constants, from the first instance's spec."""
    key = (domain_name, dataset)
    if key not in _CONSTANTS:
        domain = load_domain(domain_name)
        _CONSTANTS[key] = {c.name for c in domain.spec(domain.load_instances(dataset)[0]).constants}
    return _CONSTANTS[key]


def run_row(run_dir: Path) -> Dict[str, float]:
    """Per-list means of one run (final lists and rounds 2+), and its blame-follow counts."""
    cfg = recorded_config(run_dir)
    positions, constants = POSITIONS.get(cfg.domain.name, set()), constant_names(cfg.domain.name, cfg.domain.dataset)
    final, later, follow = [], [], [0, 0, 0, 0]
    for record in sample_records(run_dir).values():
        final.append(list_counts([r["condition"] for r in record.get("final_rules") or []], positions, constants))
        for it in record["iterations"]:
            conditions = [r["condition"] for r in it.get("rules") or []]
            if it["iteration"] >= 2 and conditions:
                later.append(list_counts(conditions, positions, constants))
            if it["mode"] == "refine" and conditions:
                counts = blame_follow(it["prompt"], conditions)
                if counts:
                    follow = [a + b for a, b in zip(follow, counts)]
    row: Dict[str, float] = {}
    for prefix, lists in (("final", final), ("later", later)):
        frame = pd.DataFrame(lists)
        for col in ("conditions", "positions", "specific", *KINDS):
            row[f"{prefix}_{col}"] = float(frame[col].mean()) if len(frame) else float("nan")
        row[f"{prefix}_lists"] = len(frame)
    row.update(blamed_changed=follow[0], blamed=follow[1], others_changed=follow[2], others=follow[3])
    return row


def collect(paths: List[Path], cells: Optional[List[str]]) -> pd.DataFrame:
    rows, seen = [], set()
    for path in paths:
        exp = experiment(path)
        for u in exp.units:
            if cells and u.cell.name not in cells:
                continue
            for run in u.runs:
                if (u.cell.name, u.bench.name, run) in seen:
                    continue
                seen.add((u.cell.name, u.bench.name, run))
                rows.append({"experiment": exp.name, "cell": u.cell.name, "benchmark": u.bench.name,
                             "run": f"{run.parent.name}/{run.name}", **run_row(run)})
    return pd.DataFrame(rows)


def table(frame: pd.DataFrame) -> str:
    """One markdown table per benchmark: per cell, means over runs."""
    out = []
    for bench, rows in frame.groupby("benchmark", sort=False):
        out += [f"## {bench}", "",
                "| feedback | runs | final: conditions | reading a position | specific | position | offset "
                "| constants | rounds 2+: specific / conditions | blamed rules rewritten | other rules rewritten |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for cell, r in rows.groupby("cell", sort=False):
            def pct(changed, total):
                total = r[total].sum()
                return f"{100 * r[changed].sum() / total:.0f}% of {total}" if total else "–"
            out.append(f"| {cell} | {len(r)} | {r.final_conditions.mean():.1f} | {r.final_positions.mean():.1f} "
                       f"| {r.final_specific.mean():.1f} "
                       f"| {r.final_position.mean():.1f} | {r.final_offset.mean():.1f} | {r.final_constants.mean():.1f} "
                       f"| {r.later_specific.mean():.1f} / {r.later_conditions.mean():.1f} "
                       f"| {pct('blamed_changed', 'blamed')} | {pct('others_changed', 'others')} |")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiments", nargs="+", type=Path)
    parser.add_argument("--cells", nargs="*", help="cell names to include (default: every cell)")
    parser.add_argument("--out", type=Path, default=Path("out/results/experiments/rq2_verifier_output/instance_specific"))
    args = parser.parse_args(argv)
    frame = collect(args.experiments, args.cells)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out.with_suffix(".csv"), index=False)
    page = table(frame)
    args.out.with_suffix(".md").write_text(page, encoding="utf-8")
    print(page)


if __name__ == "__main__":
    main()
