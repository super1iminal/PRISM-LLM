"""Transfer heatmaps: each frozen rule set (row, the instance it was written for) on every target instance (column).

Usage (from PRISM-Guided-Learning/):
    python viz/plot_transfer.py configs/plot/transfer_grid.yaml
Settings: the plot config (PlotConfig below). Reads <run>/transfer_<dataset stem>.csv from src/transfer.py.

Cell colour: requirements met in the worst case, grey from none to all but one; certified cells (every requirement
met) in blue. The diagonal (each rule set on its own instance) is outlined; rows without rules (the instance ended
with an error) are hatched. With `group`, ticks name groups of consecutive instances (e.g. grid size).
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import plot_config  # noqa: E402
from core.domain import load_domain  # noqa: E402
from results_io import run_facts  # noqa: E402
from theme import GRID, INK, INK_2, SURFACE  # noqa: E402

CERTIFIED = "#2a78d6"
LOW, HIGH = "#f1f0ec", "#9e9d98"   # grey ramp: no requirement met .. all but one


@dataclass
class PlotConfig:
    """A configs/plot/*.yaml file for this script (configs/plot/transfer_grid.yaml documents each key)."""
    script: str
    runs: Dict[str, str]          # panel title -> run dir (with a transfer CSV)
    dataset: str                  # target dataset: reads transfer_<stem>.csv
    group: Optional[str]          # instance field that groups the ticks (e.g. "n"); null: instance ids
    group_label: str              # tick text for a group, from its value {v}
    title: str
    out: str


def matrix(csv: Path, n: int) -> np.ndarray:
    """Requirements met per (source, target); NaN for sources without rules. Certified cells hold the maximum."""
    df = pd.read_csv(csv)
    m = np.full((n, n), np.nan)
    m[df.source, df.target] = df.met.fillna(0)
    return m


def colormap(requirements: int) -> ListedColormap:
    lo, hi = np.array(to_rgb(LOW)), np.array(to_rgb(HIGH))
    greys = [lo + (hi - lo) * k / max(requirements - 1, 1) for k in range(requirements)]
    return ListedColormap(greys + [to_rgb(CERTIFIED)])


def groups(instances, field: Optional[str]):
    """(start, end, value) per run of consecutive instances with the same `field`; one per instance if None."""
    values = [inst.data[field] if field else inst.id for inst in instances]
    out, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            out.append((start, i, values[start]))
            start = i
    return out


def panel(ax, m: np.ndarray, requirements: int, title: str, spans, label: str):
    n = len(m)
    ax.imshow(np.ma.masked_invalid(m), cmap=colormap(requirements), vmin=-0.5, vmax=requirements + 0.5,
              interpolation="nearest")
    for row in np.where(np.isnan(m).all(axis=1))[0]:
        ax.add_patch(Rectangle((-0.5, row - 0.5), n, 1, facecolor=SURFACE, edgecolor=GRID, hatch="///", lw=0))
    for i in range(n):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=0.8))
    for start, _, _ in spans[1:]:
        ax.axhline(start - 0.5, color=SURFACE, lw=2)
        ax.axvline(start - 0.5, color=SURFACE, lw=2)
    ticks = [(s + e - 1) / 2 for s, e, _ in spans]
    names = [label.format(v=v) for _, _, v in spans]
    ax.set_xticks(ticks, names)
    ax.set_yticks(ticks, names)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    own = np.nansum(np.diag(m) == requirements)
    off = ~np.eye(n, dtype=bool) & ~np.isnan(m)
    ax.set_title(f"{title}\nown instance {int(own)}/{int((~np.isnan(np.diag(m))).sum())}, "
                 f"other instances {int((m[off] == requirements).sum())}/{int(off.sum())}",
                 loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.set_xlabel("certified on (target)", color=INK_2, fontsize=9)
    ax.set_ylabel("rules written for (source)", color=INK_2, fontsize=9)


def main(argv=None):
    cfg = plot_config(PlotConfig, "plot_transfer", argv)
    stem = Path(cfg.dataset).stem
    first = Path(next(iter(cfg.runs.values())))
    instances = load_domain(run_facts(first).domain).load_instances(cfg.dataset)
    spans = groups(instances, cfg.group)
    requirements = int(pd.read_csv(first / f"transfer_{stem}.csv").requirements.max())

    fig, axes = plt.subplots(1, len(cfg.runs), figsize=(5.6 * len(cfg.runs), 6.4), facecolor=SURFACE, squeeze=False)
    missing = False
    for ax, (title, run) in zip(axes[0], cfg.runs.items()):
        m = matrix(Path(run) / f"transfer_{stem}.csv", len(instances))
        missing |= bool(np.isnan(m).all(axis=1).any())
        panel(ax, m, requirements, title, spans, cfg.group_label)
    ramp = colormap(requirements).colors
    handles = [Patch(color=ramp[0]), Patch(color=ramp[requirements - 1]), Patch(color=CERTIFIED),
               Patch(facecolor=SURFACE, edgecolor=INK, lw=0.8)]
    labels = ["no requirement met", f"{requirements - 1} of {requirements} met", f"certified (all {requirements})",
              "own instance"]
    if missing:
        handles.append(Patch(facecolor=SURFACE, edgecolor=GRID, hatch="///"))
        labels.append("no rules (run error)")
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(cfg.title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
