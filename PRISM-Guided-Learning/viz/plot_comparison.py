"""Legacy vs symbolic, grid by grid.

Usage (from PRISM-Guided-Learning/):
    python viz/plot_comparison.py --legacy out/results/legacy_grid20 --symbolic out/results/symbolic_grid20_capped
    python viz/plot_comparison.py ... --samples 0-4 --include-partial --out viz/figures/first5.png

Symbolic probabilities are worst case (a guarantee over every completion of the rules); the
black tick marks the best case. Samples missing from either run are dropped.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from loaders import add_summary_metrics, load_legacy, load_symbolic, requirement_names  # noqa: E402

# Reference categorical palette, slots 1-2 (validated all-pairs)
LEGACY, SYMBOLIC = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BAR_W = 0.36


def parse_samples(text):
    if not text:
        return None
    ids = set()
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        ids.update(range(int(lo), int(hi or lo) + 1))
    return sorted(ids)


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def paired_bars(ax, labels, legacy, symbolic, fmt, symbolic_best=None):
    x = np.arange(len(labels))
    bars = [ax.bar(x - BAR_W / 2 - 0.01, legacy, BAR_W, color=LEGACY, label="Legacy (per-state policy)"),
            ax.bar(x + BAR_W / 2 + 0.01, symbolic, BAR_W, color=SYMBOLIC, label="Symbolic (rules, worst case)")]
    if symbolic_best is not None:
        ax.scatter(x + BAR_W / 2 + 0.01, symbolic_best, marker="_", s=300, color=INK, linewidths=2, zorder=3,
                   label="Symbolic, best case")
    ax.set_xticks(x, labels)
    tops = [None, None if symbolic_best is None else np.asarray(symbolic_best, dtype=float)]
    for group, top in zip(bars, tops):
        for i, bar in enumerate(group):
            h = bar.get_height()
            if np.isfinite(h):
                y = max(h, top[i]) if top is not None and np.isfinite(top[i]) else h  # clear the best-case tick
                ax.annotate(fmt.format(h), (bar.get_x() + bar.get_width() / 2, y), ha="center", va="bottom",
                            xytext=(0, 3), textcoords="offset points", fontsize=7.5, color=INK_2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", required=True, type=Path)
    parser.add_argument("--symbolic", required=True, type=Path)
    parser.add_argument("--data", default="grid_20_balanced.csv")
    parser.add_argument("--samples", default=None, help="e.g. 0-4 or 0,2,5-7")
    parser.add_argument("--include-partial", action="store_true", help="include symbolic samples that did not finish")
    parser.add_argument("--title", default=None)
    parser.add_argument("--out", type=Path, default=Path("viz/figures/comparison.png"))
    args = parser.parse_args()

    legacy = load_legacy(args.legacy, args.data)
    symbolic = load_symbolic(args.symbolic, legacy["size"].to_dict(), args.include_partial)
    common = sorted(set(legacy.index) & set(symbolic.index))
    wanted = parse_samples(args.samples)
    if wanted is not None:
        common = [s for s in common if s in wanted]
    if not common:
        raise SystemExit("no samples present in both runs")
    legacy, symbolic = add_summary_metrics(legacy.loc[common]), add_summary_metrics(symbolic.loc[common])
    reqs = requirement_names(legacy)

    has_tokens = symbolic.output_tokens.notna().all() and legacy.output_tokens.notna().all()
    if len(common) > 8:
        # Many grids: one group per grid size, showing the mean over that size's grids
        counts = legacy.groupby("size").size()
        labels = [f"{s}x{s}\n(n={counts[s]})" for s in counts.index]
        leg_g, sym_g = legacy.groupby("size"), symbolic.groupby("size")
        unit, fmt_met = "mean per grid", "{:.1f}"

        def agg(g, col):
            return g[col].mean()
    else:
        labels = [f"#{s}\n{int(legacy.loc[s, 'size'])}x{int(legacy.loc[s, 'size'])}"
                  + ("" if symbolic.loc[s, "complete"] else f"\n(partial: {int(symbolic.loc[s, 'attempts'])}/5)")
                  for s in common]
        leg_g, sym_g = legacy, symbolic
        unit, fmt_met = "per grid", "{:.0f}"

        def agg(g, col):
            return g[col]

    fig = plt.figure(figsize=(13, 13 if has_tokens else 9), facecolor=SURFACE)
    grid = fig.add_gridspec(3 if has_tokens else 2, 2)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0])]
    if has_tokens:
        axes.append(fig.add_subplot(grid[1, 1]))
        req_axis = fig.add_subplot(grid[2, :])
    else:
        req_axis = fig.add_subplot(grid[1, 1])

    paired_bars(axes[0], labels, agg(leg_g, "met"), agg(sym_g, "met"), fmt_met, agg(sym_g, "met_best"))
    style(axes[0], f"Requirements met (of {len(reqs)}), {unit}", "requirements")
    axes[0].set_ylim(0, len(reqs) + 0.6)

    paired_bars(axes[1], labels, agg(leg_g, "shortfall"), agg(sym_g, "shortfall"), "{:.2f}")
    style(axes[1], f"Shortfall below thresholds, {unit} (lower is better)", "probability")

    paired_bars(axes[2], labels, agg(leg_g, "time_s") / 60, agg(sym_g, "time_s") / 60, "{:.1f}")
    style(axes[2], f"Wall time (5 attempts), {unit}", "minutes")

    if has_tokens:
        paired_bars(axes[3], labels, agg(leg_g, "output_tokens") / 1000, agg(sym_g, "output_tokens") / 1000, "{:.1f}k")
        style(axes[3], f"LLM output tokens, {unit}", "thousand tokens")
    paired_bars(req_axis, [r.replace("_", " ").replace("avoid moving ", "avoid ") for r in reqs],
                [legacy[f"p_{r}"].mean() for r in reqs], [symbolic[f"p_{r}"].mean() for r in reqs], "{:.2f}",
                [symbolic[f"p_best_{r}"].mean() for r in reqs])
    style(req_axis, f"Final probability per requirement (mean over {len(common)} grids)", "probability")
    req_axis.set_ylim(0, 1.12)
    if not has_tokens:
        plt.setp(req_axis.get_xticklabels(), rotation=35, ha="right")

    handles, names = axes[0].get_legend_handles_labels()
    order = sorted(range(len(names)), key=lambda i: "best" in names[i])  # bars first, best-case tick last
    handles, names = [handles[i] for i in order], [names[i] for i in order]
    fig.legend(handles, names, loc="upper center", ncol=3, frameon=False, fontsize=10, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.955))
    title = args.title or (f"Legacy vs symbolic policies: qwen3:14b, {len(common)} gridworlds"
                           + (", grouped by grid size" if len(common) > 8 else ""))
    fig.suptitle(title, x=0.012, ha="left", y=0.99, fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.925), h_pad=3, w_pad=2.5)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    table = pd.concat({"legacy": legacy, "symbolic": symbolic}, names=["approach"])
    table.to_csv(args.out.with_suffix(".csv"))
    print(args.out)


if __name__ == "__main__":
    main()
