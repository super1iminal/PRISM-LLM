"""One page with every ablation result so far: tables, paired tests and figures.

Reads every finished run under out/results/ablations/<condition>/seed_<k>/ and rewrites
out/results/ablations/summary/ (SUMMARY.md + PNGs). Safe to rerun at any time; unfinished runs
are listed as pending.

Usage (from PRISM-Guided-Learning/): python viz/ablation_summary.py [--root out/results/ablations]
"""
import argparse
import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from loaders import add_summary_metrics, load_legacy, load_symbolic  # noqa: E402
from plot_budget import legacy_curve, symbolic_curve  # noqa: E402
from plot_comparison import INK, INK_2, GRID, SURFACE, style  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = "grid_20_balanced.csv"
SYMBOLIC_FILE, LEGACY_FILE = "SYMBOLIC_results.parquet", "LEGACY_FEEDBACK_SIMPLIFIED_results.parquet"

# condition -> (family, reference for paired tests, what changes)
CONDITIONS = {
    "B2": ("Baselines", None, "Symbolic defaults"),
    "R1": ("Retry sweep", "B2", "Never restart"),
    "R2": ("Retry sweep", "B2", "Restart after 1 stall"),
    "R3": ("Retry sweep", "B2", "Restart every 3rd round"),
    "R4": ("Retry sweep", "B2", "Restart when gain < 0.05"),
    "R5": ("Retry sweep", "B2", "Always restart (no feedback)"),
    "S1": ("Feedback content", "B2", "Results table only"),
    "S4": ("Feedback content", "B2", "No examples in the prompt"),
    "S5": ("Blame signal", "B2", "Blame by one-step regret"),
    "V1": ("Blame signal", "B2", "Blame on random rules/states"),
    "V2": ("Blame signal", "B2", "No blame section"),
    "B1": ("Baselines", "B2", "Legacy per-state"),
    "L1": ("Legacy variants", "B1", "Legacy + restart after 2 stalls"),
    "L2": ("Legacy variants", "B1", "Legacy without worked examples"),
}
FAMILY_COLORS = {"Baselines": "#2a78d6", "Retry sweep": "#eb6834", "Feedback content": "#1baf7a",
                 "Blame signal": "#eda100", "Legacy variants": "#4a3aa7"}
BUDGET_GROUPS = [("Retry sweep", ["B2", "R1", "R2", "R3", "R4", "R5", "B1"]),
                 ("Feedback content and blame", ["B2", "S1", "S4", "S5", "V1", "V2"]),
                 ("Legacy", ["B1", "L1", "L2", "B2"])]
LINE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def solvable_ids() -> set:
    path = ROOT / "out" / "results" / "ceilings" / f"gridworld_{Path(DATASET).stem}.csv"
    if not path.exists():
        return set(range(20))
    df = pd.read_csv(path)
    return {int(i) for i, ok in zip(df["sample_id"], df["jointly_feasible"]) if str(ok) == "True"}


def kept_coverage(run_dir: Path) -> pd.Series:
    """Uncovered share (%) of the kept policy, per sample (symbolic only)."""
    df = pd.read_parquet(run_dir / SYMBOLIC_FILE).reset_index().sort_values(["sample_id", "iteration"])
    out = {}
    for sid, g in df.groupby("sample_id"):
        kept = g[g.improved.astype(bool)]
        row = (kept if len(kept) else g).iloc[-1]
        out[sid] = 100 * row.uncovered_situations / max(row.reachable_situations, 1)
    return pd.Series(out)


def mode_stats(run_dir: Path) -> dict:
    df = pd.read_parquet(run_dir / SYMBOLIC_FILE).reset_index()
    df = df[df.iteration > 1]
    return {f"{m}_rounds": int((df["mode"] == m).sum()) for m in ("refine", "extend", "initial", "table")} | {
        f"{m}_improved": float(df[df["mode"] == m].improved.mean()) if (df["mode"] == m).any() else np.nan
        for m in ("refine", "extend", "initial", "table")}


def load_run(condition: str, run_dir: Path, solvable: set) -> pd.DataFrame:
    legacy = (run_dir / LEGACY_FILE).exists()
    df = add_summary_metrics(load_legacy(run_dir, DATASET) if legacy else load_symbolic(run_dir))
    if legacy:
        df["met_best"], df["uncovered_pct"] = df["met"], 0.0
    else:
        df["uncovered_pct"] = kept_coverage(run_dir)
    df["solved"] = df["met"] == len([c for c in df.columns if c.startswith("p_") and not c.startswith("p_best_")])
    df["solvable"] = [sid in solvable for sid in df.index]
    df["condition"], df["seed"], df["approach"] = condition, run_dir.name, "legacy" if legacy else "symbolic"
    return df.reset_index()


def finished_runs(root: Path):
    runs, pending = [], []
    for condition in CONDITIONS:
        for run_dir in sorted((root / condition).glob("seed_*")):
            if (run_dir / SYMBOLIC_FILE).exists() or (run_dir / LEGACY_FILE).exists():
                runs.append((condition, run_dir))
            else:
                pending.append(f"{condition}/{run_dir.name}")
    return runs, pending


def paired(data: pd.DataFrame, condition: str, reference: str, col: str = "met", n_perm: int = 20000):
    """Mean per-grid difference (seeds averaged), 95% bootstrap CI and sign-flip permutation p-value."""
    a = data[data.condition == condition].groupby("sample_id")[col].mean()
    b = data[data.condition == reference].groupby("sample_id")[col].mean()
    d = (a - b).dropna().to_numpy()
    if len(d) < 3:
        return None
    rng = np.random.default_rng(0)
    boot = rng.choice(d, size=(n_perm, len(d))).mean(axis=1)
    flips = rng.choice([-1, 1], size=(n_perm, len(d)))
    p = float(((np.abs((flips * d).mean(axis=1)) >= abs(d.mean()) - 1e-12).sum() + 1) / (n_perm + 1))
    return {"diff": float(d.mean()), "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975)),
            "p": p, "n": len(d)}


def condition_table(data: pd.DataFrame, extra: dict) -> pd.DataFrame:
    rows = []
    for condition in [c for c in CONDITIONS if c in set(data.condition)]:
        d = data[data.condition == condition]
        per_seed = d.groupby("seed")
        family, reference, what = CONDITIONS[condition]
        test = paired(data, condition, reference) if reference in set(data.condition) else None
        rows.append({
            "condition": condition, "family": family, "change": what, "seeds": d.seed.nunique(),
            "solved": per_seed.apply(lambda g: int((g.solved & g.solvable).sum())).mean(),
            "met": d.met.mean(), "met_seed_min": per_seed.met.mean().min(), "met_seed_max": per_seed.met.mean().max(),
            "met_best": d.met_best.mean(), "shortfall": d.shortfall.mean(), "uncovered": d.uncovered_pct.mean(),
            "tokens_k": d.output_tokens.mean() / 1000, "minutes": d.time_s.mean() / 60,
            "vs": reference if test else "", "diff": test["diff"] if test else np.nan,
            "ci": f"[{test['lo']:+.2f}, {test['hi']:+.2f}]" if test else "", "p": test["p"] if test else np.nan,
            **extra.get(condition, {}),
        })
    return pd.DataFrame(rows)


def _bar_panel(ax, table, data, col, title, ylabel, fmt):
    x = np.arange(len(table))
    colors = [FAMILY_COLORS[f] for f in table.family]
    ax.bar(x, table[col], 0.65, color=colors)
    for i, condition in enumerate(table.condition):
        seeds = data[data.condition == condition].groupby("seed")
        values = seeds.apply(lambda g: int((g.solved & g.solvable).sum())) if col == "solved" else seeds[
            {"met": "met", "shortfall": "shortfall", "uncovered": "uncovered_pct"}[col]].mean()
        ax.scatter(np.full(len(values), i), values, color=INK, s=14, zorder=3)
        ax.annotate(fmt.format(table[col].iloc[i]), (i, max(table[col].iloc[i], values.max())), ha="center",
                    va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=7.5, color=INK_2)
    for ref, ls in (("B2", "--"), ("B1", ":")):
        if ref in set(table.condition):
            ax.axhline(table[table.condition == ref][col].iloc[0], color=INK_2, lw=0.9, ls=ls, zorder=0)
    ax.set_xticks(x, table.condition)
    style(ax, title, ylabel)


def plot_conditions(table, data, out):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), facecolor=SURFACE)
    n_solvable = len(solvable_ids())
    _bar_panel(axes[0, 0], table, data, "met", "Requirements met (of 9), worst case", "mean per grid", "{:.2f}")
    _bar_panel(axes[0, 1], table, data, "shortfall", "Shortfall below thresholds (lower is better)", "mean per grid",
               "{:.2f}")
    _bar_panel(axes[1, 0], table, data, "solved", f"Grids solved (of {n_solvable} solvable)", "mean over seeds",
               "{:.1f}")
    ax = axes[1, 1]
    t = table[table.vs != ""].reset_index(drop=True)
    y = np.arange(len(t))[::-1]
    for yi, (_, r) in zip(y, t.iterrows()):
        color = FAMILY_COLORS[r.family]
        ax.plot([float(r.ci.strip("[]").split(",")[0]), float(r.ci.strip("[]").split(",")[1])], [yi, yi], color=color, lw=2)
        ax.scatter([r["diff"]], [yi], color=color, s=40, zorder=3)
        ax.annotate(f"p={r.p:.2f}", (r["diff"], yi), xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=7.5, color=INK_2)
    ax.axvline(0, color=INK_2, lw=0.9)
    ax.set_yticks(y, [f"{c} vs {v}" for c, v in zip(t.condition, t.vs)])
    style(ax, "Paired difference in requirements met (per grid)", "")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("difference vs reference, 95% CI", color=INK_2, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    fig.legend(handles + [plt.Line2D([], [], marker="o", ls="", color=INK, ms=4)],
               list(FAMILY_COLORS) + ["one seed"], loc="upper center", ncol=6, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("Ablations on 20 gridworlds (qwen3 14B, 5 rounds, obstacle visible), bars = mean over seeds",
                 x=0.01, ha="left", y=0.99, fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=3)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_mechanics(table, out):
    t = table[~table.condition.isin(["B1", "L1", "L2"])].reset_index(drop=True)
    panels = [("uncovered", "Uncovered situations, kept policy", "% of reachable", "{:.0f}%"),
              ("refine_improved", "Refine rounds that improved the policy", "share", "{:.0%}"),
              ("tokens_k", "LLM output tokens per grid", "thousand", "{:.1f}k"),
              ("minutes", "Wall time per grid", "minutes", "{:.1f}")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), facecolor=SURFACE)
    for ax, (col, title, ylabel, fmt) in zip(axes, panels):
        x = np.arange(len(t))
        values = t[col].fillna(0) if col in t else np.zeros(len(t))
        ax.bar(x, values, 0.65, color=[FAMILY_COLORS[f] for f in t.family])
        for i, v in enumerate(values):
            ax.annotate(fmt.format(v), (i, v), ha="center", va="bottom", xytext=(0, 2), textcoords="offset points",
                        fontsize=7, color=INK_2)
        ax.set_xticks(x, t.condition)
        style(ax, title, ylabel)
    fig.suptitle("Symbolic loop mechanics by condition", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_budget(runs, out):
    curves = []
    for condition, run_dir in runs:
        legacy = (run_dir / LEGACY_FILE).exists()
        rows = legacy_curve(run_dir, 5, DATASET) if legacy else symbolic_curve(run_dir, 5)
        curves.append(pd.DataFrame(rows).assign(condition=condition, seed=run_dir.name))
    data = pd.concat(curves)
    summary = data.groupby(["condition", "k"]).met.mean().reset_index()
    fig, axes = plt.subplots(1, len(BUDGET_GROUPS), figsize=(16, 4.4), facecolor=SURFACE, sharey=True)
    for ax, (title, members) in zip(axes, BUDGET_GROUPS):
        for i, condition in enumerate([m for m in members if m in set(summary.condition)]):
            d = summary[summary.condition == condition]
            ls = "--" if condition in ("B2", "B1") and members[0] != condition else "-"
            ax.plot(d.k, d.met, color=LINE_COLORS[i % len(LINE_COLORS)], lw=2, marker="o", ms=5, ls=ls,
                    label=f"{condition} {CONDITIONS[condition][2].lower()}")
            ax.annotate(f"{d.met.iloc[-1]:.2f}", (d.k.iloc[-1], d.met.iloc[-1]), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=7.5, color=INK_2)
        style(ax, title, "requirements met (mean per grid)")
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("rounds budget k", color=INK_2, fontsize=9)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("Requirements met by the kept policy after k rounds (seeds pooled)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return summary.pivot(index="condition", columns="k", values="met")


def uuv_table(root: Path) -> str:
    rows = []
    for run_dir in sorted((root / "U1").glob("seed_*")):
        path = run_dir / SYMBOLIC_FILE
        if not path.exists():
            continue
        df = pd.read_parquet(path).reset_index()
        final = df[df.is_final]
        reqs = [c[len("final_worst_"):] for c in final.columns if c.startswith("final_worst_")]
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        for _, f in final.iterrows():
            values = ", ".join(f"{r} {f[f'final_worst_{r}']:.3f}" for r in reqs)
            rows.append(f"| {run_dir.name} | {['North Sea', 'Caribbean'][int(f.sample_id)]} | {bool(f.success)} | "
                        f"{int(df[df.sample_id == f.sample_id].iteration.max())} | {int(f.final_num_rules)} | {values} |")
    if not rows:
        return ""
    return ("| seed | scenario | solved (worst case) | rounds | rules | final worst-case values |\n"
            "|---|---|---|---|---|---|\n" + "\n".join(rows))


def write_markdown(out_dir, table, budget, pending, uuv):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# Ablation results (auto-generated {now})", "",
             "Gridworld, 20 grids (19 solvable), qwen3 14B, 5 rounds, obstacle phase visible to rules and legacy. "
             "Symbolic numbers are worst case. Paired tests compare per-grid means (seeds averaged) against the "
             "reference with a sign-flip permutation test; with 20 grids and 2 seeds, treat p > 0.05 as noise.", "",
             "Regenerate: `python viz/ablation_summary.py`.", ""]
    if pending:
        lines += ["**Pending runs:** " + ", ".join(pending), ""]
    lines += ["## Overview", "", "![conditions](conditions.png)", "", "## Budget curves", "", "![budget](budget.png)",
              "", "## Loop mechanics (symbolic)", "", "![mechanics](mechanics.png)", "", "## Table", "",
              "| cond | change | seeds | solved | req. met (seed range) | best case | shortfall | uncovered % | "
              "vs | Δ met [95% CI] | p | refine improved | out tokens | min/grid |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        refine = f"{r.refine_improved:.0%}" if "refine_improved" in r and pd.notna(r.refine_improved) else ""
        diff = f"{r['diff']:+.2f} {r.ci}" if r.vs else ""
        lines.append(f"| {r.condition} | {r.change} | {r.seeds} | {r.solved:.1f} | {r.met:.2f} "
                     f"({r.met_seed_min:.2f} to {r.met_seed_max:.2f}) | {r.met_best:.2f} | {r.shortfall:.2f} | "
                     f"{r.uncovered:.0f} | {r.vs} | {diff} | {'' if pd.isna(r.p) else f'{r.p:.3f}'} | {refine} | "
                     f"{r.tokens_k:.1f}k | {r.minutes:.1f} |")
    ks = list(budget.columns)
    lines += ["", "## Requirements met after k rounds", "", "| cond | " + " | ".join(f"k={k}" for k in ks) + " |",
              "|---|" + "---|" * len(ks)]
    lines += [f"| {c} | " + " | ".join(f"{budget.loc[c, k]:.2f}" for k in ks) + " |"
              for c in CONDITIONS if c in budget.index]
    lines.append("")
    if uuv:
        lines += ["## UUV (U1: symbolic defaults on the paper's two scenarios, with the energy budget)", "", uuv, ""]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "out" / "results" / "ablations")
    args = parser.parse_args()
    out_dir = args.root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    solvable = solvable_ids()
    runs, pending = finished_runs(args.root)
    data = pd.concat([load_run(c, d, solvable) for c, d in runs], ignore_index=True)
    extra = {}
    for condition, run_dir in runs:
        if (run_dir / SYMBOLIC_FILE).exists():
            extra.setdefault(condition, []).append(mode_stats(run_dir))
    extra = {c: pd.DataFrame(v).mean(numeric_only=True).to_dict() for c, v in extra.items()}
    table = condition_table(data, extra)
    table.to_csv(out_dir / "conditions.csv", index=False)
    data.to_csv(out_dir / "per_grid.csv", index=False)

    plot_conditions(table, data, out_dir / "conditions.png")
    plot_mechanics(table, out_dir / "mechanics.png")
    budget = plot_budget(runs, out_dir / "budget.png")
    write_markdown(out_dir, table, budget, pending, uuv_table(args.root))
    print(out_dir / "SUMMARY.md")


if __name__ == "__main__":
    main()
