"""Transfer-search summary: per condition, training sets solved and how the frozen rules do on held-out targets.

Usage (from PRISM-Guided-Learning/):
    python viz/transfer_summary.py configs/plot/transfer_search.yaml [--set key=value]
Settings: the plot config (PlotConfig below; configs/plot/transfer_search.yaml documents each key). Reads each
condition's runs (<root>/<condition>/seed_*/): outputs/sample_*.json and transfer_<dataset stem>.csv from
src/transfer.py. Baselines (src/baselines/) are read from their own transfer-format CSVs.

Per condition, over its runs (mean, and the range over runs):
  * sets solved: training sets (or instances) whose rules pass every requirement of every member;
  * certified: held-out targets certified per rule set (targets the rule set was trained on are left out);
  * met: share of requirements met in the worst case on those targets;
  * tokens and dollars per rule set (prompt and output tokens over every round, at the configured prices).
And a target x condition table: how often each target is certified, as a share of the condition's rule sets.
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import plot_config  # noqa: E402
from experiments.metrics import run_row, transfer_numbers  # noqa: E402,F401  (transfer_numbers: re-exported)
from run_ablation import existing_root  # noqa: E402
from transfer import parse_ids  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent


@dataclass
class PlotConfig:
    """A configs/plot/*.yaml file for this script (configs/plot/transfer_search.yaml documents each key)."""
    script: str
    root: Optional[str]              # runs: <root>/<condition>/seed_*/; null: out/results/core or ablations (the comparison experiments, configs/experiments/)
    dataset: str                     # target dataset of the transfer CSVs
    targets: str                     # held-out target ids, e.g. "12-19"
    rules: str                       # final | determinized (src/determinize.py): which transfer CSVs to read
    conditions: List[str]            # in table order
    prices: Dict[str, List[float]]   # model -> [$ per M prompt tokens, $ per M output tokens]
    baselines: Dict[str, str]        # name -> a transfer-format CSV
    groups: Dict[str, str]           # group name -> its target ids; {}: every source counts on every target
    source_groups: Dict[str, str]    # source (sample id or baseline source name) -> the group it is checked on
    title: str
    out: Optional[str]               # the page; null: print only


def run_dirs(root: Path, condition: str) -> List[Path]:
    """The condition's finished runs (those with per-sample outputs), in name order."""
    return sorted(p for p in (root / condition).glob("seed_*") if (p / "outputs").is_dir())


def summarize(cfg: PlotConfig) -> pd.DataFrame:
    """One row per run of every condition, plus one per baseline CSV."""
    targets = parse_ids(cfg.targets)
    rows = [{"condition": c, **run_row(d, cfg.dataset, targets, cfg.prices, cfg.rules, cfg.groups, cfg.source_groups)}
            for c in cfg.conditions for d in run_dirs(PROJECT / cfg.root if cfg.root else existing_root(c), c)]
    rows += [{"condition": name, "run": "-", **transfer_numbers(PROJECT / path, targets, cfg.groups, cfg.source_groups)}
             for name, path in cfg.baselines.items()]
    return pd.DataFrame(rows)


def _span(values: pd.Series, fmt: str) -> str:
    values = values.dropna()
    if values.empty:
        return "–"
    mean = format(values.mean(), fmt)
    return mean if len(values) == 1 or values.min() == values.max() else \
        f"{mean} ({format(values.min(), fmt)}–{format(values.max(), fmt)})"


def markdown(cfg: PlotConfig, runs: pd.DataFrame) -> str:
    """The summary page: the condition table and the target table."""
    targets = parse_ids(cfg.targets)
    lines = [f"# {cfg.title}", "",
             f"Held-out targets: {cfg.targets} of `{cfg.dataset}` ({len(targets)}). Mean over runs, range in "
             f"parentheses. Certified: held-out targets certified per rule set. Met: share of requirements met in "
             f"the worst case on them.", "",
             "| condition | runs | sets solved | certified (%s) | best rule set | met | prompt tokens / set | "
             "output tokens / set | $ / set |" % ("on its own group" if cfg.groups else f"of {len(targets)}"),
             "|---|---|---|---|---|---|---|---|---|"]
    for name, g in runs.groupby("condition", sort=False):
        unchecked = int((~g.checked.fillna(False).astype(bool)).sum()) if "checked" in g else len(g)
        solved = _span(g.solved / g.sets, ".0%") if "sets" in g and g.sets.notna().any() else "–"
        lines.append(f"| {name}{f' ({unchecked} unchecked)' if unchecked else ''} | {len(g)} | {solved} | "
                     f"{_span(g.get('certified', pd.Series(dtype=float)), '.2f')} | "
                     f"{_span(g.get('certified_best', pd.Series(dtype=float)), '.0f')} | "
                     f"{_span(g.get('met_share', pd.Series(dtype=float)), '.0%')} | "
                     f"{_span(g.get('prompt_tokens', pd.Series(dtype=float)), ',.0f')} | "
                     f"{_span(g.get('output_tokens', pd.Series(dtype=float)), ',.0f')} | "
                     f"{_span(g.get('dollars', pd.Series(dtype=float)), '.3f')} |")
    if cfg.groups:
        lines += ["", "Certified per rule set on its own group's targets (mean over runs):", "",
                  "| condition | " + " | ".join(f"{g} (of {len(parse_ids(t))})" for g, t in cfg.groups.items()) + " |",
                  "|---|" + "---|" * len(cfg.groups)]
        for name, g in runs.groupby("condition", sort=False):
            by_group = [d for d in g.get("by_group", pd.Series(dtype=object)).dropna()]
            cells = [pd.Series([d.get(k) for d in by_group], dtype=float).mean() for k in cfg.groups]
            lines.append(f"| {name} | " + " | ".join("–" if pd.isna(c) else f"{c:.1f}" for c in cells) + " |")
    lines += ["", "Share of rule sets certifying each held-out target:", "",
              "| condition | " + " | ".join(str(t) for t in targets) + " |",
              "|---|" + "---|" * len(targets)]
    for name, g in runs.groupby("condition", sort=False):
        by_target = [d for d in g.get("by_target", pd.Series(dtype=object)).dropna()]
        shares = [pd.Series([d.get(t) for d in by_target], dtype=float).mean() for t in targets]
        lines.append(f"| {name} | " + " | ".join("–" if pd.isna(s) else f"{s:.0%}" for s in shares) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    cfg = plot_config(PlotConfig, "transfer_summary", argv)
    page = markdown(cfg, summarize(cfg))
    if cfg.out:
        out = PROJECT / cfg.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page, encoding="utf-8")
        print(out)
    print(page)


if __name__ == "__main__":
    main()
