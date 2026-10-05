"""Compare legacy against one or more symbolic runs (e.g. prompt ablations), by grid size.

Usage (from PRISM-Guided-Learning/):
    python viz/plot_runs.py configs/plot/catchall_ablation.yaml
Settings: the plot config (PlotConfig below). Model and round budget come from the runs.

Writes the figure, a CSV of per-grid numbers, and a markdown table (same stem).
Symbolic probabilities are worst case; black ticks mark the best case.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from loaders import (add_summary_metrics, load_legacy, load_symbolic, plot_config,  # noqa: E402
                     requirement_names, short_model)
from results_io import SYMBOLIC_RESULTS, met_and_shortfall, requirements_by_sample, run_facts  # noqa: E402
from theme import GRID, INK, INK_2, SURFACE, style  # noqa: E402,F401

# Reference categorical palette, slots 1-3 (validated all-pairs in light and dark)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MODES = ["first", "retry", "refine", "extend"]  # "retry" = blind retry (initial prompt after round 1)


@dataclass
class PlotConfig:
    """A configs/plot/*.yaml file for this script (configs/plot/catchall_ablation.yaml documents each key)."""
    script: str
    legacy: str
    symbolic: Dict[str, str]
    title: Optional[str]
    out: str


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
    df = pd.read_parquet(run_dir / SYMBOLIC_RESULTS).reset_index().sort_values(["sample_id", "iteration"])
    reqs = [c[len("prob_worst_"):] for c in df.columns if c.startswith("prob_worst_")]
    by_sample = requirements_by_sample(run_dir)

    def key(row):   # the planner's keep-best score: failures, then shortfalls (worst, best)
        worst = met_and_shortfall(by_sample[row.sample_id], {r: row[f"prob_worst_{r}"] for r in reqs})
        best = met_and_shortfall(by_sample[row.sample_id], {r: row[f"prob_best_{r}"] for r in reqs})
        return len(reqs) - worst[0], len(reqs) - best[0], worst[1], best[1]
    df["key"] = [key(row) for _, row in df.iterrows()]
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
    cfg = plot_config(PlotConfig, "plot_runs")
    if len(cfg.symbolic) > 2:
        raise SystemExit("at most two symbolic runs (three series stay distinguishable)")
    legacy_dir, out = Path(cfg.legacy), Path(cfg.out)
    facts = [run_facts(Path(path)) for path in cfg.symbolic.values()]

    legacy = load_legacy(legacy_dir)
    runs = []
    for label, path in cfg.symbolic.items():
        runs.append((label, Path(path), load_symbolic(Path(path), legacy["size"].to_dict())))
    common = sorted(set(legacy.index).intersection(*[set(r[2].index) for r in runs]))
    legacy = add_summary_metrics(legacy.loc[common], legacy_dir)
    runs = [(label, path, add_summary_metrics(df.loc[common], path)) for label, path, df in runs]
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
    style(ax["time"], f"Wall time ({max(f.max_rounds for f in facts)} rounds), mean per grid", "minutes")

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
    models = " / ".join(dict.fromkeys(short_model(f.model) for f in facts))
    fig.suptitle(cfg.title or f"Legacy vs symbolic runs: {models}, {len(common)} gridworlds, grouped by grid size",
                 x=0.012, ha="left", y=0.995, fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3, w_pad=2.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)

    # Per-grid CSV and a markdown summary
    pd.concat({n: f for n, f in zip(names, frames)}, names=["run"]).to_csv(out.with_suffix(".csv"))
    lines = [f"# {cfg.title or 'Legacy vs symbolic runs'}", "", "| metric | " + " | ".join(names) + " |",
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
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
