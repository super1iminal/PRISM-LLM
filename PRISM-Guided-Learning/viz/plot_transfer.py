"""Transfer heatmaps: each frozen rule set (row) certified on every target instance (column).

Usage (from PRISM-Guided-Learning/):
    python viz/plot_transfer.py configs/plot/transfer_grid.yaml
Settings: the plot config (PlotConfig below). Reads <run>/transfer_<dataset stem>.csv from src/transfer.py.

A row is the rule set written for one instance, or for one training set (run.train_sets). Cell colour:
requirements met in the worst case, grey from none to all but one; certified cells (every requirement met) in
blue. The instances a rule set was written for are outlined; rows without rules (the run ended with an error)
are hatched. With `group`, column ticks (and row ticks of a per-instance run) name groups of consecutive
instances (e.g. grid size).
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
from transfer import training_instances  # noqa: E402

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


@dataclass
class Transfer:
    """One run's transfer results as matrices: rows are its rule sets (samples), columns the targets."""
    sources: List[int]
    trained_on: Dict[int, List[int]]  # source -> the instances its rules were written for
    met: np.ndarray                   # requirements met; NaN for sources without rules
    trained: np.ndarray               # bool: the target is one the rule set was written for

    @classmethod
    def load(cls, run: Path, stem: str, targets: int) -> "Transfer":
        df = pd.read_csv(run / f"transfer_{stem}.csv")
        trained_on = training_instances(run)
        sources = sorted(trained_on)
        row = {s: i for i, s in enumerate(sources)}
        met, trained = np.full((len(sources), targets), np.nan), np.zeros((len(sources), targets), bool)
        met[[row[s] for s in df.source], df.target] = df.met.fillna(0)
        for s, ids in trained_on.items():
            trained[row[s], [i for i in ids if i < targets]] = True
        return cls(sources, trained_on, met, trained)

    def certified(self, requirements: int, on_trained: bool) -> str:
        """'certified/checked' over the trained-on cells, or over the held-out ones."""
        cells = (self.trained if on_trained else ~self.trained) & ~np.isnan(self.met)
        return f"{int((self.met[cells] == requirements).sum())}/{int(cells.sum())}"


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


def panel(ax, t: Transfer, requirements: int, title: str, spans, label: str):
    rows, cols = t.met.shape
    ax.imshow(np.ma.masked_invalid(t.met), cmap=colormap(requirements), vmin=-0.5, vmax=requirements + 0.5,
              interpolation="nearest", aspect="auto" if rows != cols else "equal")
    for r in np.where(np.isnan(t.met).all(axis=1))[0]:
        ax.add_patch(Rectangle((-0.5, r - 0.5), cols, 1, facecolor=SURFACE, edgecolor=GRID, hatch="///", lw=0))
    for r, c in zip(*np.where(t.trained)):
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=INK, lw=0.8))
    for start, _, _ in spans[1:]:
        ax.axvline(start - 0.5, color=SURFACE, lw=2)
    ticks, names = [(s + e - 1) / 2 for s, e, _ in spans], [label.format(v=v) for _, _, v in spans]
    ax.set_xticks(ticks, names)
    if rows == cols and all(t.trained_on[s] == [s] for s in t.sources):   # one rule set per instance
        for start, _, _ in spans[1:]:
            ax.axhline(start - 0.5, color=SURFACE, lw=2)
        ax.set_yticks(ticks, names)
        ax.set_ylabel("rules written for (source)", color=INK_2, fontsize=9)
    else:
        ax.set_yticks(range(rows), [f"set {s}: " + ", ".join(map(str, t.trained_on[s])) for s in t.sources])
        ax.set_ylabel("rules trained on (instance ids)", color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(f"{title}\ncertified: trained-on {t.certified(requirements, True)}, "
                 f"held-out {t.certified(requirements, False)}", loc="left", fontsize=10, color=INK, fontweight="bold")
    ax.set_xlabel("certified on (target)", color=INK_2, fontsize=9)


def main(argv=None):
    cfg = plot_config(PlotConfig, "plot_transfer", argv)
    stem = Path(cfg.dataset).stem
    first = Path(next(iter(cfg.runs.values())))
    instances = load_domain(run_facts(first).domain).load_instances(cfg.dataset)
    spans = groups(instances, cfg.group)
    requirements = int(pd.read_csv(first / f"transfer_{stem}.csv").requirements.max())
    transfers = {title: Transfer.load(Path(run), stem, len(instances)) for title, run in cfg.runs.items()}

    rows = max(len(t.sources) for t in transfers.values())
    height = 6.4 if rows == len(instances) else 2.6 + 0.35 * rows
    fig, axes = plt.subplots(1, len(transfers), figsize=(5.6 * len(transfers), height), facecolor=SURFACE,
                             squeeze=False)
    for ax, (title, t) in zip(axes[0], transfers.items()):
        panel(ax, t, requirements, title, spans, cfg.group_label)
    ramp = colormap(requirements).colors
    handles = [Patch(color=ramp[0]), Patch(color=ramp[requirements - 1]), Patch(color=CERTIFIED),
               Patch(facecolor=SURFACE, edgecolor=INK, lw=0.8)]
    labels = ["no requirement met", f"{requirements - 1} of {requirements} met", f"certified (all {requirements})",
              "written for this instance"]
    if any(np.isnan(t.met).all(axis=1).any() for t in transfers.values()):
        handles.append(Patch(facecolor=SURFACE, edgecolor=GRID, hatch="///"))
        labels.append("no rules (run error)")
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(cfg.title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.06 if rows == len(instances) else 0.12, 1, 0.95))
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
