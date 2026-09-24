"""Compare legacy against one or more symbolic runs (e.g. prompt ablations), by grid size.

Usage (from PRISM-Guided-Learning/):
    python viz/plot_runs.py --legacy out/results/legacy_grid20 \
        --symbolic "catch-all asked=out/results/symbolic_grid20_capped" \
        --symbolic "no catch-all=out/results/symbolic_grid20_nocatchall" \
        --out viz/figures/catchall_ablation.png

Writes the figure, a CSV of per-grid numbers, and a markdown table (same stem).
Symbolic probabilities are worst case; black ticks mark the best case.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from loaders import add_summary_metrics, load_legacy, load_symbolic, requirement_names  # noqa: E402
from legacy.requirements import get_threshold_for_key  # noqa: E402

# Reference categorical palette, slots 1-3 (validated all-pairs in light and dark)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODES = ["first", "retry", "refine", "extend"]  # "retry" = blind retry (initial prompt after round 1)


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


def grouped_bars(ax, labels, series, fmt, ticks=None, colors=None):
    """series: list of (name, values); ticks: optional per-series best-case markers (or None)."""
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(labels))
    colors = colors or COLORS
    for k, (name, values) in enumerate(series):
        pos = x - 0.4 + width * (k + 0.5)
        values = np.asarray(values, dtype=float)
        ax.bar(pos, values, width * 0.94, color=colors[k], label=name)
        top = values.copy()
        if ticks is not None and ticks[k] is not None:
            t = np.asarray(ticks[k], dtype=float)
            ax.scatter(pos, t, marker="_", s=180, color=INK, linewidths=2, zorder=3,
                       label="Symbolic, best case" if k == 1 else None)
            top = np.fmax(values, t)
        for p, v, y in zip(pos, values, top):
            if np.isfinite(v):
                ax.annotate(fmt.format(v), (p, y), ha="center", va="bottom", xytext=(0, 3),
                            textcoords="offset points", fontsize=7, color=INK_2)
    ax.set_xticks(x, labels)


def symbolic_rounds(run_dir: Path) -> pd.DataFrame:
    """Per-round rows with mode, coverage and whether the round improved the kept policy."""
    df = pd.read_parquet(run_dir / "SYMBOLIC_results.parquet").reset_index().sort_values(["sample_id", "iteration"])
    reqs = [c[len("prob_worst_"):] for c in df.columns if c.startswith("prob_worst_")]
    thr = {r: get_threshold_for_key(r) for r in reqs}
    df["key"] = list(zip(
        sum((df[f"prob_worst_{r}"] < thr[r]).astype(int) for r in reqs),
        sum((df[f"prob_best_{r}"] < thr[r]).astype(int) for r in reqs),
        sum((thr[r] - df[f"prob_worst_{r}"]).clip(lower=0) for r in reqs),
        sum((thr[r] - df[f"prob_best_{r}"]).clip(lower=0) for r in reqs)))
    improved = []
    for _, g in df.groupby("sample_id"):
        best = None
        for key in g.key:
            improved.append(best is None or key < best)
            best = key if best is None or key < best else best
    df["improved"] = improved
    df["uncovered_frac"] = df.uncovered_situations / df.reachable_situations
    df["mode"] = np.where(df.iteration == 1, "first", np.where(df["mode"] == "initial", "retry", df["mode"]))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", required=True, type=Path)
    parser.add_argument("--symbolic", action="append", required=True, help="label=path, repeatable (max 2)")
    parser.add_argument("--data", default="grid_20_balanced.csv")
    parser.add_argument("--title", default=None)
    parser.add_argument("--out", type=Path, default=Path("viz/figures/runs.png"))
    args = parser.parse_args()
    if len(args.symbolic) > 2:
        raise SystemExit("at most two symbolic runs (three series stay distinguishable)")

    legacy = load_legacy(args.legacy, args.data)
    runs = []
    for spec in args.symbolic:
        label, _, path = spec.partition("=")
        runs.append((label, Path(path), load_symbolic(Path(path), legacy["size"].to_dict())))
    common = sorted(set(legacy.index).intersection(*[set(r[2].index) for r in runs]))
    legacy = add_summary_metrics(legacy.loc[common])
    runs = [(label, path, add_summary_metrics(df.loc[common])) for label, path, df in runs]
    rounds = {label: symbolic_rounds(path) for label, path, _ in runs}
    for r in rounds.values():
        r["size"] = r.sample_id.map(legacy["size"])
    reqs = requirement_names(legacy)

    sizes = sorted(legacy["size"].unique())
    labels = [f"{s}x{s}" for s in sizes]

    def by_size(df, col):
        return df.groupby("size")[col].mean().reindex(sizes).values

    names = ["Legacy"] + [f"Symbolic: {label}" for label, _, _ in runs]
    frames = [legacy] + [df for _, _, df in runs]

    fig = plt.figure(figsize=(14, 15), facecolor=SURFACE)
    grid = fig.add_gridspec(4, 2)
    ax = {k: fig.add_subplot(grid[i, j]) for k, (i, j) in
          {"met": (0, 0), "short": (0, 1), "tokens": (1, 0), "time": (1, 1), "modes": (3, 0), "uncov": (3, 1)}.items()}
    ax["req"] = fig.add_subplot(grid[2, :])

    grouped_bars(ax["met"], labels, [(n, by_size(f, "met")) for n, f in zip(names, frames)], "{:.1f}",
                 [None] + [by_size(f, "met_best") for f in frames[1:]])
    style(ax["met"], f"Requirements met (of {len(reqs)}), mean per grid", "requirements")
    ax["met"].set_ylim(0, len(reqs) + 0.6)
    grouped_bars(ax["short"], labels, [(n, by_size(f, "shortfall")) for n, f in zip(names, frames)], "{:.2f}")
    style(ax["short"], "Shortfall below thresholds, mean per grid (lower is better)", "probability")
    grouped_bars(ax["tokens"], labels, [(n, by_size(f, "output_tokens") / 1000) for n, f in zip(names, frames)], "{:.1f}")
    style(ax["tokens"], "LLM output tokens, mean per grid", "thousand tokens")
    grouped_bars(ax["time"], labels, [(n, by_size(f, "time_s") / 60) for n, f in zip(names, frames)], "{:.1f}")
    style(ax["time"], "Wall time (5 rounds), mean per grid", "minutes")

    grouped_bars(ax["req"], [r.replace("_", " ").replace("avoid moving ", "avoid ") for r in reqs],
                 [(n, [f[f"p_{r}"].mean() for r in reqs]) for n, f in zip(names, frames)], "{:.2f}",
                 [None] + [[f[f"p_best_{r}"].mean() for r in reqs] for f in frames[1:]])
    style(ax["req"], f"Final probability per requirement (mean over {len(common)} grids)", "probability")
    ax["req"].set_ylim(0, 1.12)

    # Loop behaviour (symbolic only): rounds per mode, labelled with the share that improved the kept policy
    width = 0.8 / len(runs)
    for k, (label, _, _) in enumerate(runs):
        r = rounds[label]
        for i, m in enumerate(MODES):
            sub = r[r["mode"] == m]
            pos = i - 0.4 + width * (k + 0.5)
            ax["modes"].bar(pos, len(sub), width * 0.94, color=COLORS[1 + k])
            text = f"{len(sub)}" + (f"\n{100 * sub.improved.mean():.0f}%" if len(sub) and m != "first" else "")
            ax["modes"].annotate(text, (pos, len(sub)), ha="center", va="bottom", xytext=(0, 3),
                                 textcoords="offset points", fontsize=7, color=INK_2)
    ax["modes"].set_xticks(range(len(MODES)), ["first round", "blind retry", "refine", "extend"])
    ax["modes"].margins(y=0.2)
    style(ax["modes"], "Rounds per loop mode (label: count, % that improved the kept policy)", "rounds")

    grouped_bars(ax["uncov"], labels,
                 [(f"Symbolic: {label}", rounds[label].groupby("size").uncovered_frac.mean().reindex(sizes).values * 100)
                  for label, _, _ in runs], "{:.0f}%", colors=COLORS[1:])
    style(ax["uncov"], "Uncovered reachable situations, mean over rounds", "% of reachable situations")

    handles, hl = [], []
    for a in (ax["met"],):
        h, l = a.get_legend_handles_labels()
        handles += h
        hl += l
    order = sorted(range(len(hl)), key=lambda i: "best" in hl[i])
    fig.legend([handles[i] for i in order], [hl[i] for i in order], loc="upper center", ncol=4, frameon=False,
               fontsize=10, labelcolor=INK, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(args.title or f"Legacy vs symbolic runs: qwen3:14b, {len(common)} gridworlds, grouped by grid size",
                 x=0.012, ha="left", y=0.995, fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3, w_pad=2.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)

    # Per-grid CSV and a markdown summary
    pd.concat({n: f for n, f in zip(names, frames)}, names=["run"]).to_csv(args.out.with_suffix(".csv"))
    lines = [f"# {args.title or 'Legacy vs symbolic runs'}", "", "| metric | " + " | ".join(names) + " |",
             "|---|" + "---|" * len(names)]

    def row(metric, values):
        lines.append(f"| {metric} | " + " | ".join(values) + " |")

    row("solved (all requirements; symbolic worst case)", [f"{int((f.met == len(reqs)).sum())}/{len(f)}" for f in frames])
    row("requirements met, mean", [f"{f.met.mean():.2f}" for f in frames])
    row("shortfall, mean", [f"{f.shortfall.mean():.3f}" for f in frames])
    row("output tokens, mean", [f"{f.output_tokens.mean():.0f}" for f in frames])
    row("wall time (s), mean", [f"{f.time_s.mean():.0f}" for f in frames])
    row("best-case requirements met, mean", ["n/a"] + [f"{f.met_best.mean():.2f}" for f in frames[1:]])
    for m in MODES:
        row(f"rounds in {m} (improved)", ["n/a"] + [
            f"{(rounds[l]['mode'] == m).sum()} ({int(rounds[l][rounds[l]['mode'] == m].improved.sum())})"
            for l, _, _ in runs])
    row("uncovered situations, mean % over rounds", ["n/a"] + [f"{100 * rounds[l].uncovered_frac.mean():.1f}%" for l, _, _ in runs])
    row("rounds with any uncovered situation", ["n/a"] + [
        f"{int((rounds[l].uncovered_situations > 0).sum())}/{len(rounds[l])}" for l, _, _ in runs])
    row("rules per round, mean", ["n/a"] + [f"{rounds[l].num_rules.mean():.1f}" for l, _, _ in runs])
    row("invalid answers, total", ["n/a"] + [f"{int(rounds[l].invalid_answers.sum())}" for l, _, _ in runs])
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(args.out)


if __name__ == "__main__":
    main()
