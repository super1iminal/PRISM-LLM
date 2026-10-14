"""What a run (or a baseline) achieved: training sets solved, cost, and how its frozen rules do on held-out targets,
from its outputs and the transfer CSVs of src/transfer.py (or src/run_baseline.py). Shared by the experiment
harnesses (RQ1 comparison, RQ2 ablations) and viz/transfer_summary.py.

Transfer numbers, over held-out targets only (a source's own training instances are left out):
  * certified: targets certified per rule list, on its own group's targets when there are groups (e.g. a family's
    list on that family), else on every target; mean over the lists;
  * certified_all: the same on every target, whatever the group (a family's list on all families);
  * certified_best: the best single list's `certified`;
  * library: targets certified by at least one of the lists (certification at run time picks one that passes);
  * met_share: share of requirements met in the worst case.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from results_io import sample_records
from transfer import csv_path, parse_ids


def transfer_numbers(path: Path, targets: List[int], groups: Optional[Dict[str, str]] = None,
                     source_groups: Optional[Dict[str, str]] = None, sources: Optional[List[str]] = None) -> dict:
    """The transfer numbers of one CSV; `checked` is False when it is missing or does not cover every target.
    `source_groups`: source -> the group it is checked on ("*": a list for every group, e.g. the template, which
    counts once per group on that group's targets). `sources`: only these sources (default: all)."""
    if not path.exists():
        return {"checked": False}
    df = pd.read_csv(path)
    if sources is not None:
        df = df[df.source.astype(str).isin([str(s) for s in sources])]
    checked = set(targets) <= set(df.target)
    df = df[df.target.isin(targets) & ~df.trained.astype(bool)]
    df = df.assign(certified=df.certified.fillna(False).astype(bool) if "certified" in df else False)
    everywhere = df.groupby(df.source.astype(str)).certified.sum()
    library = int(df[df.certified].target.nunique())
    group_of = {}
    if groups:
        members = {g: set(parse_ids(spec)) for g, spec in groups.items()}
        of_target = {t: g for g, ts in members.items() for t in ts}
        wanted = {str(s): (source_groups or {}).get(str(s)) for s in df.source.unique()}
        df = df.assign(source=[f"{s}@{of_target.get(t)}" if wanted[str(s)] == "*" else str(s)
                               for s, t in zip(df.source, df.target)])
        group_of = {s: (s.rsplit("@", 1)[1] if "@" in s and wanted.get(s.rsplit("@", 1)[0]) == "*"
                        else wanted.get(s)) for s in df.source.unique()}
        df = df[[group_of[s] is not None and t in members.get(group_of[s], ()) for s, t in zip(df.source, df.target)]]
    per_source = df.certified.groupby(df.source.astype(str)).sum()
    met = (df.met.fillna(0) / df.requirements.fillna(1)).mean() if "met" in df else 0.0
    by_group = {g: per_source[[s for s in per_source.index if group_of.get(s) == g]].mean() for g in (groups or {})}
    return {"checked": checked, "certified": per_source.mean(), "certified_best": per_source.max(),
            "certified_all": everywhere.mean(), "library": library, "met_share": met,
            "by_target": df.certified.groupby(df.target).mean().to_dict(), "by_group": by_group,
            "per_source": per_source.to_dict()}


def model_of(run_dir: Path) -> str:
    path = run_dir / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))["llm"]["model"] if path.exists() else ""


def run_numbers(run_dir: Path, prices: Dict[str, List[float]]) -> dict:
    """Training sets solved, rounds, and tokens and dollars per set (prompt and output tokens over every round, at
    `prices`: model -> [$ per M prompt tokens, $ per M output tokens])."""
    records = sample_records(run_dir)
    sets = max(len(records), 1)
    prompt = sum(it.get("llm_prompt_tokens") or 0 for r in records.values() for it in r.get("iterations", []))
    output = sum(it.get("llm_output_tokens") or 0 for r in records.values() for it in r.get("iterations", []))
    price_in, price_out = prices.get(model_of(run_dir), [float("nan"), float("nan")])
    return {"run": run_dir.name, "sets": len(records), "solved": sum(bool(r.get("success")) for r in records.values()),
            "rounds": sum(len(r.get("iterations", [])) for r in records.values()) / sets,
            "prompt_tokens": prompt / sets, "output_tokens": output / sets,
            "dollars": (prompt * price_in + output * price_out) / 1e6 / sets}


def run_row(run_dir: Path, dataset: str, targets: List[int], prices: Dict[str, List[float]], rules: str = "final",
            groups: Optional[Dict[str, str]] = None, source_groups: Optional[Dict[str, str]] = None) -> dict:
    """One run's numbers: `run_numbers` and the transfer numbers of its `rules` on `targets` of `dataset`."""
    return {**run_numbers(run_dir, prices),
            **transfer_numbers(csv_path(run_dir, dataset, rules), targets, groups, source_groups)}
