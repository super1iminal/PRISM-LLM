"""Transfer-search figures: held-out instances certified per frozen rule set, our method against the baselines, on
the random grids and the gridworld families; Pac-Man's worst-case crash probability per horizon; and the ablations.

Usage (from PRISM-Guided-Learning/):
    python viz/plot_transfer_results.py configs/plot/transfer_results.yaml
Settings: the plot config (PlotConfig below; configs/plot/transfer_results.yaml documents each key).

A method is an LLM condition (its runs; the bar is the mean over runs of the certified targets per rule set, the
whisker the range over runs), a baseline's transfer CSV (`csv` and its `sources`; the bar is the mean over sources,
the whisker their range), or explicit `values` from the numpy model (drawn hatched: PRISM certification pending).
Our methods are blue, baselines grey.
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import plot_config  # noqa: E402
from run_ablation import existing_root  # noqa: E402
from theme import GRID, INK, INK_2, SURFACE, style  # noqa: E402
from transfer import csv_path, parse_ids  # noqa: E402
from transfer_summary import run_dirs, transfer_numbers  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
OURS, OURS_STRONG, BASE, BASE_LIGHT = "#2a78d6", "#184f95", "#9e9d98", "#d6d5d0"


@dataclass
class PlotConfig:
    """A configs/plot/*.yaml file for this script (configs/plot/transfer_results.yaml documents each key)."""
    script: str
    grid: Dict[str, Any]
    families: Dict[str, Any]
    pacman: Dict[str, Any]
    ablations: Dict[str, Any]
    library: Dict[str, Any]
    out_dir: str


# ---------------------------------------------------------------- data

def method_values(m: dict, dataset: str, targets: List[int], group: str = None, groups=None,
                  source_groups=None) -> List[float]:
    """Per run (LLM condition) or per source (baseline), certified held-out targets per rule set."""
    if "values" in m:
        v = m["values"]
        return list(v[group] if isinstance(v, dict) else v)
    if "condition" in m:
        out = []
        for run in run_dirs(existing_root(m["condition"]), m["condition"]):
            n = transfer_numbers(csv_path(run, dataset), targets, groups, source_groups)
            if not n.get("checked"):
                continue
            out.append(n["by_group"][group] if group else n["certified"])
        return out
    df = pd.read_csv(PROJECT / m["csv"])
    df = df[df.target.isin(targets) & ~df.trained.astype(bool)]
    if group:
        df = df[df.target.isin(parse_ids(groups[group]))]
    sources = m["sources"][group] if isinstance(m["sources"], dict) else m["sources"]
    return [float(df[df.source.astype(str) == str(s)].certified.fillna(False).astype(bool).sum()) for s in sources]


# ---------------------------------------------------------------- drawing

def bars(ax, methods: List[dict], values: List[List[float]], top: float, title: str):
    """Horizontal bars, one per method: mean with the range as a whisker, the mean written at the bar's end."""
    y = np.arange(len(methods))[::-1]
    for yi, m, v in zip(y, methods, values):
        if not v:
            ax.text(0.1, yi, "not run yet", va="center", fontsize=8, color=INK_2)
            continue
        mean = float(np.mean(v))
        color = (OURS_STRONG if m.get("strong") else OURS) if m.get("ours") else BASE
        pending = "values" in m
        ax.barh(yi, mean, height=0.62, color=BASE_LIGHT if pending and not m.get("ours") else color,
                hatch="////" if pending else None, edgecolor=BASE if pending else SURFACE, linewidth=0.8, zorder=2)
        if len(v) > 1 and min(v) != max(v):
            ax.plot([min(v), max(v)], [yi, yi], color=INK, linewidth=1.2, zorder=3)
            ax.plot([min(v)] * 2, [yi - 0.13, yi + 0.13], color=INK, linewidth=1.2, zorder=3)
            ax.plot([max(v)] * 2, [yi - 0.13, yi + 0.13], color=INK, linewidth=1.2, zorder=3)
        ax.text(max(max(v), mean) + top * 0.02, yi, f"{mean:.1f}" if mean % 1 else f"{mean:.0f}", va="center",
                fontsize=8.5, color=INK, fontweight="bold" if m.get("ours") else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([m["label"] for m in methods], fontsize=8.5, color=INK)
    ax.set_xlim(0, top * 1.12)
    ax.set_xticks(range(0, int(top) + 1, 2 if top <= 10 else 5))
    style(ax, title)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines["bottom"].set_visible(False)


def grid_figure(cfg: dict, out: Path):
    targets = parse_ids(cfg["targets"])
    panels = cfg["panels"]
    fig, axes = plt.subplots(1, len(panels), figsize=(max(9.0, 5.2 * len(panels)), 0.42 * max(len(p) for p in panels.values()) + 1.6),
                             facecolor=SURFACE, sharex=True)
    for ax, (title, methods) in zip(np.atleast_1d(axes), panels.items()):
        bars(ax, methods, [method_values(m, cfg["dataset"], targets) for m in methods], len(targets), title)
        ax.set_xlabel(f"validation grids certified per rule set (of {len(targets)})", fontsize=8.5, color=INK_2)
    fig.suptitle(cfg["title"], x=0.01, y=0.995, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, cfg["note"], fontsize=7.5, color=INK_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def family_figure(cfg: dict, out: Path):
    targets, groups = parse_ids(cfg["targets"]), cfg["groups"]
    methods = cfg["methods"]
    fig, axes = plt.subplots(1, len(groups), figsize=(4.4 * len(groups), 0.42 * len(methods) + 1.6),
                             facecolor=SURFACE, sharey=True)
    for i, (ax, group) in enumerate(zip(axes, groups)):
        vals = [method_values(m, cfg["dataset"], targets, group, groups, cfg["source_groups"]) for m in methods]
        size = len(parse_ids(groups[group]))
        bars(ax, methods, vals, size, group)
        ax.set_xlabel(f"members certified (of {size})", fontsize=8.5, color=INK_2)
        if i:
            ax.tick_params(labelleft=False)
    fig.suptitle(cfg["title"], x=0.01, y=0.995, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, cfg["note"], fontsize=7.5, color=INK_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def pacman_figure(cfg: dict, out: Path):
    horizons = cfg["horizons"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor=SURFACE)
    ax.axhline(cfg["optimum"], color=GRID, linewidth=2, zorder=1)
    ax.text(horizons[-1] + 0.4, cfg["optimum"] - 0.004, f"optimum {cfg['optimum']}", fontsize=8, color=INK_2, va="top")
    ax.axhline(cfg["threshold"], color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(horizons[-1] + 0.4, cfg["threshold"] + 0.004, f"threshold {cfg['threshold']}", fontsize=8, color=INK_2,
            va="bottom")
    for i, series in enumerate(cfg["series"]):
        df = pd.read_csv(PROJECT / series["csv"])
        df = df[df.source == series["source"]].sort_values("target")
        xs = [horizons[t] for t in df.target]
        color = OURS_STRONG if i or len(cfg["series"]) == 1 else OURS
        ax.plot(xs, df.worst_crash, color=color, linewidth=2, marker="o", markersize=5, zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(xs[-1] + 0.3, df.worst_crash.iloc[-1], series["label"], fontsize=8.5, color=INK, va="center")
    for p in cfg["published"]:
        ax.plot([p["horizon"]], [p["value"]], marker="D", markersize=8, color=BASE, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(p["horizon"] + 0.3, p["value"], p["label"], fontsize=8.5, color=INK, va="center")
    ax.set_xticks(horizons)
    ax.set_xlim(horizons[0] - 0.5, horizons[-1] + 6.5)
    ax.set_ylim(cfg["optimum"] - 0.03, None)
    ax.set_xlabel("horizon (moves); training horizons marked in the labels", fontsize=8.5, color=INK_2)
    style(ax, cfg["title"], "worst-case crash probability (lower is better)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def ablation_figure(cfg: dict, out: Path):
    targets, xmin = parse_ids(cfg["targets"]), cfg.get("xmin", 0)
    rows = [(group, m) for group, ms in cfg["groups"].items() for m in ms]
    # one empty slot above each group for its heading
    slots, heads, pos = [], [], 0
    for group, ms in cfg["groups"].items():
        heads.append((group, pos))
        pos += 1
        slots += list(range(pos, pos + len(ms)))
        pos += len(ms)
    y = pos - 1 - np.array(slots)
    fig, ax = plt.subplots(figsize=(8.4, 0.34 * pos + 1.4), facecolor=SURFACE)
    for yi, (group, m) in zip(y, rows):
        v = method_values(m, cfg["dataset"], targets)
        if not v:
            ax.text(xmin + 0.1, yi, "running", fontsize=8, color=INK_2, va="center")
            continue
        color = OURS if group == cfg["highlight"] else BASE
        ax.plot([min(v), max(v)], [yi, yi], color=color, linewidth=2.2, alpha=0.45, zorder=2, solid_capstyle="round")
        ax.plot([np.mean(v)], [yi], marker="o", markersize=8, color=color, zorder=3, markeredgecolor=SURFACE,
                markeredgewidth=1.5)
        ax.text(max(v) + 0.12, yi, f"{np.mean(v):.2f}", fontsize=8, color=INK, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([m["label"] for _, m in rows], fontsize=8.5, color=INK)
    ax.set_xlim(xmin, len(targets))
    ax.set_ylim(-0.6, pos - 0.4)
    ax.set_xlabel(f"validation grids certified per rule set (of {len(targets)})", fontsize=8.5, color=INK_2)
    style(ax, "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines["bottom"].set_visible(False)
    for group, slot in heads:
        ax.text(xmin + 0.05, pos - 1 - slot, group, fontsize=8.5, va="center", fontweight="bold",
                color=OURS if group == cfg["highlight"] else INK_2)
    fig.suptitle(cfg["title"], x=0.01, y=0.99, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Dot: mean over runs; bar: range over runs.", fontsize=7.5, color=INK_2, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def library_values(m: dict, dataset: str, targets: List[int]):
    """Per run (LLM condition) or for the baseline's listed sources: (targets certified per rule list, targets
    certified by at least one of the lists), every target counting whatever group it is in."""
    if "condition" in m:
        nums = [transfer_numbers(csv_path(run, dataset), targets)
                for run in run_dirs(existing_root(m["condition"]), m["condition"])]
    else:
        nums = [transfer_numbers(PROJECT / m["csv"], targets, sources=[str(s) for s in m["sources"]])]
    nums = [n for n in nums if n.get("checked")]
    return [n["certified"] for n in nums], [n["library"] for n in nums]


def library_figure(cfg: dict, out: Path):
    """Two bars per method: one rule list on every target, and a run's lists used as a library."""
    targets = parse_ids(cfg["targets"])
    methods = cfg["methods"]
    fig, ax = plt.subplots(figsize=(8.4, 0.62 * len(methods) + 1.5), facecolor=SURFACE)
    y = np.arange(len(methods))[::-1] * 1.0
    for yi, m in zip(y, methods):
        per_list, library = library_values(m, cfg["dataset"], targets)
        for dy, v, light in ((0.17, per_list, True), (-0.17, library, False)):
            if not v:
                continue
            mean = float(np.mean(v))
            color = (OURS if light else OURS_STRONG) if m.get("ours") else (BASE_LIGHT if light else BASE)
            ax.barh(yi + dy, mean, height=0.32, color=color, edgecolor=SURFACE, linewidth=0.8, zorder=2)
            if len(v) > 1 and min(v) != max(v):
                ax.plot([min(v), max(v)], [yi + dy] * 2, color=INK, linewidth=1.1, zorder=3)
            ax.text(max(max(v), mean) + len(targets) * 0.015, yi + dy, f"{mean:.1f}" if mean % 1 else f"{mean:.0f}",
                    va="center", fontsize=8, color=INK, fontweight="bold" if m.get("ours") else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([m["label"] for m in methods], fontsize=8.5, color=INK)
    ax.set_xlim(0, len(targets) * 1.12)
    ax.set_xlabel(f"held-out members certified (of {len(targets)}, all families)", fontsize=8.5, color=INK_2)
    style(ax, "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines["bottom"].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BASE_LIGHT, label="one rule list (mean over lists)"),
                       Patch(color=BASE, label="a run's lists together (library)")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8, frameon=False)
    fig.suptitle(cfg["title"], x=0.01, y=0.99, ha="left", fontsize=12, fontweight="bold", color=INK)
    import textwrap
    fig.text(0.01, 0.01, textwrap.fill(cfg["note"], 150), fontsize=7.5, color=INK_2, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv=None):
    cfg = plot_config(PlotConfig, "plot_transfer_results", argv)
    out_dir = PROJECT / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn, section in [("transfer_grid_results.png", grid_figure, cfg.grid),
                              ("transfer_family_results.png", family_figure, cfg.families),
                              ("transfer_pacman_results.png", pacman_figure, cfg.pacman),
                              ("transfer_ablations.png", ablation_figure, cfg.ablations),
                              ("transfer_library.png", library_figure, cfg.library)]:
        fn(section, out_dir / name)
        print(out_dir / name)


if __name__ == "__main__":
    main()
