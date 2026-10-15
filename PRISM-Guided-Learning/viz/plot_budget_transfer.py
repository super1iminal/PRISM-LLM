"""Budget curves at equal cost (plan 03): held-out targets certified per rule list, and training sets solved, against
the spend per training set, one line per cell of an experiment (mean over runs, band: range over runs), one column
per benchmark. Data: each run's budget_transfer.csv (src/budget_transfer.py). Keep-best makes every line rise, so
read the lines against each other at equal spend, never a line's rise on its own.

Usage (from PRISM-Guided-Learning/):
    python viz/plot_budget_transfer.py configs/plot/budget_transfer.yaml
The experiment's report (src/run_experiment.py) draws the same figure when its file has a `budget:` block.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter  # noqa: E402

from config import plot_config  # noqa: E402
from theme import GRID, INK, INK_2, SURFACE, style  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]      # the reference palette's first three slots (all pairs distinct)


@dataclass
class PlotConfig:
    """A configs/plot/*.yaml file for this script (configs/plot/budget_transfer.yaml documents each key)."""
    script: str
    experiment: str
    cost: str = "dollars"
    points: int = 121
    colors: Dict[str, int] = field(default_factory=dict)
    dashed: List[str] = field(default_factory=list)
    dotted: List[str] = field(default_factory=list)
    out: Optional[str] = None


def _line_style(cfg: PlotConfig, code: str) -> dict:
    slot = cfg.colors.get(code)
    color = SLOTS[slot - 1] if slot else INK_2
    dash = "--" if code in cfg.dashed else ":" if code in cfg.dotted else "-"
    return {"color": color, "linestyle": dash, "linewidth": 2.0 if slot else 1.3, "zorder": 3 if slot else 2}


def _money(cost: str):
    if cost == "dollars":
        return FuncFormatter(lambda v, _: f"${v:.3f}")
    return FuncFormatter(lambda v, _: f"{v / 1000:.0f}k")


def draw(data: pd.DataFrame, exp, cfg: PlotConfig, marks: Dict[str, List[float]], firsts: pd.DataFrame) -> Path:
    """The figure from `curves` rows (bench, cell, code, run, x, certified, solved); `marks`: bench -> budgets to
    draw as vertical lines (the planned comparisons); `firsts`: the LLM's first answers per bench."""
    from budget_transfer import COSTS
    benches = [b for b in exp.spec["benchmarks"].values() if (data.bench == b.name).any()]
    cells = [c for c in exp.spec["cells"] if (data.cell == c.name).any()]
    fig, axes = plt.subplots(2, len(benches), figsize=(4.6 * len(benches) + 0.6, 7.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    of = getattr(exp, "_of", lambda b: "")
    for col, bench in enumerate(benches):
        b = data[data.bench == bench.name]
        for row, (metric, ylabel) in enumerate([("certified", f"held-out targets certified per list ({of(bench)})"),
                                                ("solved", "training sets solved")]):
            ax = axes[row][col]
            style(ax, (bench.label or bench.name).split(" (")[0] if row == 0 else "", ylabel if col == 0 else None)
            if col:
                ax.set_ylabel(ylabel, color=INK_2, fontsize=8)
            ax.grid(axis="x", visible=False)
            for cell in cells:
                g = b[b.cell == cell.name]
                if g.empty:
                    continue
                stats = g.groupby("x")[metric].agg(["mean", "min", "max"]).reset_index()
                st = _line_style(cfg, cell.code)
                runs = g.run.nunique()
                ax.fill_between(stats.x, stats["min"], stats["max"], color=st["color"], alpha=0.10, linewidth=0)
                ax.plot(stats.x, stats["mean"], label=f"{cell.name} ({runs} runs)", **st)
            for x in marks.get(bench.name, []):
                ax.axvline(x, color=INK_2, linestyle=(0, (2, 3)), linewidth=0.9, zorder=1)
            f = firsts[firsts.bench == bench.name] if not firsts.empty else firsts
            if not f.empty and cfg.cost in f:
                ax.scatter(f[cfg.cost], f[metric], marker="x", s=64, color=INK, linewidths=1.6, zorder=5,
                           label="LLM's first answer, no feedback (all cells)")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
            ax.xaxis.set_major_formatter(_money(cfg.cost))
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            if metric == "solved":
                ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
                ax.set_xlabel(COSTS.get(cfg.cost, cfg.cost), color=INK_2, fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), frameon=False, fontsize=8.5,
               labelcolor=INK)
    fig.suptitle(f"{exp.title}: certified and solved against spend (equal cost)", x=0.01, ha="left", fontsize=11,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.04 + 0.025 * ((len(labels) + 2) // 3), 1, 0.97))
    out = Path(cfg.out) if cfg.out else exp.out / "budget_transfer.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv=None):
    from budget_transfer import curves, first_answers, planned
    from experiments.base import experiment
    cfg = plot_config(PlotConfig, "plot_budget_transfer", argv)
    exp = experiment(Path(cfg.experiment))
    table = planned(exp) if exp.spec.get("budget") else pd.DataFrame()
    marks = {b: sorted(set(g.budget.dropna())) for b, g in table.groupby("bench")} if not table.empty else {}
    print(draw(curves(exp, cfg.cost, cfg.points), exp, cfg, marks, first_answers(exp)))


if __name__ == "__main__":
    main()
