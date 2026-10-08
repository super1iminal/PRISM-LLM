"""One page with the results of a set of conditions: tables, paired tests and figures.

Reads every finished run under <root>/<condition>/seed_<k>/ and rewrites the `out` directory (SUMMARY.md +
PNGs). Safe to rerun at any time; unfinished runs are listed as pending. configs/plot/ablation_summary.yaml
covers the ablations; other configs compare other sets of conditions (e.g. models).

Usage (from PRISM-Guided-Learning/): python viz/ablation_summary.py configs/plot/ablation_summary.yaml
Settings: the plot config (PlotConfig below): which conditions, their families and references, the
budget groups, the cost weights, the planned comparisons, the UUV section and the text. Grid counts, model
and round budgets come from the runs.
"""
import datetime
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from loaders import add_summary_metrics, load_legacy, load_symbolic, short_model  # noqa: E402
from config import plot_config  # noqa: E402
from core.domain import load_domain  # noqa: E402
from plot_budget import at_budget, legacy_curve, symbolic_curve, with_cost  # noqa: E402
from results_io import LEGACY_RESULTS as LEGACY_FILE, SYMBOLIC_RESULTS as SYMBOLIC_FILE, run_facts  # noqa: E402
from settings import RESULTS_PATH  # noqa: E402
from theme import FAMILY_COLORS, INK, INK_2, GRID, SURFACE, style  # noqa: E402

LINE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


@dataclass
class PlotConfig:
    """configs/plot/ablation_summary.yaml documents each key."""
    script: str
    root: str
    out: str
    heading: str
    conditions: Dict[str, Dict[str, Optional[str]]]
    reference_lines: Dict[str, str]
    budget_groups: Dict[str, List[str]]
    cost: Optional[Dict[str, Any]]
    planned: List[Dict[str, str]]
    uuv: Optional[Dict[str, str]]
    title: str
    intro: str


@dataclass
class Batch:
    """What the finished runs have in common, for the text: filled into `title` and `intro`."""
    domain: str
    dataset: str
    grids: int
    solvable: Set[int]
    model: str
    rounds: int        # the most common round budget
    seeds: int         # the most seeds any condition has

    def text(self, template: str) -> str:
        return template.format(grids=self.grids, solvable=len(self.solvable), model=self.model, rounds=self.rounds,
                               seeds=self.seeds)


def describe(runs) -> Batch:
    facts = [run_facts(run_dir) for _, run_dir in runs]
    datasets = {(f.domain, f.dataset) for f in facts}
    if len(datasets) != 1:
        raise SystemExit(f"the runs use different domains or datasets: {sorted(datasets)}")
    (domain, dataset), = datasets
    grids = len(load_domain(domain).load_instances(dataset))
    path = RESULTS_PATH / "ceilings" / f"{domain}_{Path(dataset).stem}.csv"
    if path.exists():
        df = pd.read_csv(path)
        solvable = {int(i) for i, ok in zip(df["sample_id"], df["jointly_feasible"]) if str(ok) == "True"}
    else:
        solvable = set(range(grids))
    return Batch(domain, dataset, grids, solvable, " / ".join(dict.fromkeys(short_model(f.model) for f in facts)),
                 Counter(f.max_rounds for f in facts).most_common(1)[0][0],
                 max(Counter(condition for condition, _ in runs).values()))


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
    feedback = df[df["mode"].isin(["refine", "extend", "table"])]
    return {"feedback_improved": float(feedback.improved.mean()) if len(feedback) else np.nan} | {
        f"{m}_rounds": int((df["mode"] == m).sum()) for m in ("refine", "extend", "initial", "table")} | {
        f"{m}_improved": float(df[df["mode"] == m].improved.mean()) if (df["mode"] == m).any() else np.nan
        for m in ("refine", "extend", "initial", "table")}


def load_run(condition: str, run_dir: Path, solvable: set) -> pd.DataFrame:
    legacy = (run_dir / LEGACY_FILE).exists()
    df = add_summary_metrics(load_legacy(run_dir) if legacy else load_symbolic(run_dir), run_dir)
    raw = pd.read_parquet(run_dir / (LEGACY_FILE if legacy else SYMBOLIC_FILE)).reset_index()
    per = raw.groupby("sample_id")
    df["input_tokens"] = per.llm_prompt_tokens.first()
    df["prism_s"] = (per.prism_time.sum() + (per.joint_time.sum() if "joint_time" in raw else 0)
                     if "prism_time" in raw else np.nan)
    if legacy:
        df["met_best"], df["uncovered_pct"] = df["met"], 0.0
    else:
        df["uncovered_pct"] = kept_coverage(run_dir)
    df["solved"] = df["met"] == len([c for c in df.columns if c.startswith("p_") and not c.startswith("p_best_")])
    df["solvable"] = [sid in solvable for sid in df.index]
    df["condition"], df["seed"], df["approach"] = condition, run_dir.name, "legacy" if legacy else "symbolic"
    return df.reset_index()


def finished_runs(root: Path, conditions):
    runs, pending = [], []
    for condition in conditions:
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


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-adjusted p-values (step-down; controls the chance of any false positive among the comparisons)."""
    adjusted, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def planned_table(curves: pd.DataFrame, cfg: PlotConfig) -> pd.DataFrame:
    """The planned comparisons (cfg.planned): paired tests on requirements met per grid, Holm-corrected together.
    `at: rounds:N` compares the kept policies after N rounds. `at: cost` compares the condition's whole run with
    the reference at the same spend: on each grid, the reference gets the condition's mean spend there."""
    rows = []
    for c in cfg.planned:
        cond, ref, at = c["condition"], c["reference"], c["at"]
        a, b = curves[curves.condition == cond], curves[curves.condition == ref]
        if not len(a) or not len(b):
            continue
        if at.startswith("rounds:"):
            n = int(at.split(":")[1])
            if min(a.k.max(), b.k.max()) < n:
                raise SystemExit(f"planned comparison {cond} vs {ref} at {n} rounds: a run has fewer rounds")
            values, label = pd.concat([a[a.k == n], b[b.k == n]]), f"{n} rounds"
        elif at == "cost":
            if not cfg.cost:
                raise SystemExit("a planned comparison at equal cost needs `cost`")
            final = a.loc[a.groupby(["seed", "sample_id"]).k.idxmax()]
            budget = final.groupby("sample_id").cost.mean()
            values, label = pd.concat([final, at_budget(b, budget, ("seed", "sample_id"))]), "equal cost"
        else:
            raise SystemExit(f"planned comparison {cond} vs {ref}: unknown `at` {at!r} (rounds:N or cost)")
        test = paired(values, cond, ref)
        if test:
            rows.append({"condition": cond, "reference": ref, "at": label, **test})
    table = pd.DataFrame(rows)
    if len(table):
        table["p_holm"] = holm(table.p.to_numpy())
    return table


def condition_table(data: pd.DataFrame, extra: dict, conditions) -> pd.DataFrame:
    rows = []
    for condition in [c for c in conditions if c in set(data.condition)]:
        d = data[data.condition == condition]
        per_seed = d.groupby("seed")
        meta = conditions[condition]
        family, reference, what = meta["family"], meta["reference"], meta["change"]
        test = paired(data, condition, reference) if reference in set(data.condition) else None
        rows.append({
            "condition": condition, "family": family, "change": what, "seeds": d.seed.nunique(),
            "solved": per_seed.apply(lambda g: int((g.solved & g.solvable).sum())).mean(),
            "met": d.met.mean(), "met_seed_min": per_seed.met.mean().min(), "met_seed_max": per_seed.met.mean().max(),
            "met_best": d.met_best.mean(), "shortfall": d.shortfall.mean(), "uncovered": d.uncovered_pct.mean(),
            "met_median": d.met.median(), "met_q1": d.met.quantile(0.25), "met_q3": d.met.quantile(0.75),
            "short_median": d.shortfall.median(), "short_q1": d.shortfall.quantile(0.25),
            "short_q3": d.shortfall.quantile(0.75), "input_k": d.input_tokens.mean() / 1000,
            "prism_s": d.prism_s.mean(),
            "tokens_k": d.output_tokens.mean() / 1000, "minutes": d.time_s.mean() / 60,
            "vs": reference if test else "", "diff": test["diff"] if test else np.nan,
            "ci": f"[{test['lo']:+.2f}, {test['hi']:+.2f}]" if test else "", "p": test["p"] if test else np.nan,
            **extra.get(condition, {}),
        })
    return pd.DataFrame(rows)


def _bar_panel(ax, table, data, col, title, ylabel, fmt, reference_lines):
    x = np.arange(len(table))
    colors = [FAMILY_COLORS[f] for f in table.family]
    ax.bar(x, table[col], 0.65, color=colors)
    for i, condition in enumerate(table.condition):
        seeds = data[data.condition == condition].groupby("seed")
        values = seeds.apply(lambda g: int((g.solved & g.solvable).sum())) if col == "solved" else seeds[
            {"met": "met", "met_best": "met_best", "shortfall": "shortfall", "uncovered": "uncovered_pct"}[col]].mean()
        ax.scatter(np.full(len(values), i), values, color=INK, s=14, zorder=3)
        ax.annotate(fmt.format(table[col].iloc[i]), (i, max(table[col].iloc[i], values.max())), ha="center",
                    va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=7.5, color=INK_2)
    for ref, ls in reference_lines.items():
        if ref in set(table.condition):
            ax.axhline(table[table.condition == ref][col].iloc[0], color=INK_2, lw=0.9, ls=ls, zorder=0)
    ax.set_xticks(x, table.condition)
    style(ax, title, ylabel)


def _tests(table, planned, conditions) -> list:
    """The paired tests to draw: the planned comparisons when there are any, else each condition vs its reference.
    (label, diff, lo, hi, p text, colour) each."""
    color = {c: FAMILY_COLORS[m["family"]] for c, m in conditions.items()}
    if len(planned):
        return [(f"{r.condition} vs {r.reference}, {r.at}", r["diff"], r.lo, r.hi, f"p={r.p:.3f}, Holm {r.p_holm:.3f}",
                 color[r.condition]) for _, r in planned.iterrows()]
    return [(f"{r.condition} vs {r.vs}", r["diff"], *[float(v) for v in r.ci.strip("[]").split(",")], f"p={r.p:.2f}",
             color[r.condition]) for _, r in table[table.vs != ""].iterrows()]


def plot_conditions(table, data, out, title, num_reqs, reference_lines, planned, conditions):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), facecolor=SURFACE)
    _bar_panel(axes[0, 0], table, data, "met", f"Requirements met (of {num_reqs}), worst case", "mean per grid",
               "{:.2f}", reference_lines)
    _bar_panel(axes[0, 1], table, data, "shortfall", "Shortfall below thresholds (lower is better)", "mean per grid",
               "{:.2f}", reference_lines)
    _bar_panel(axes[1, 0], table, data, "met_best", "Requirements met, best case (uncovered states choose well)",
               "mean per grid", "{:.2f}", reference_lines)
    ax = axes[1, 1]
    tests = _tests(table, planned, conditions)
    y = np.arange(len(tests))[::-1]
    for yi, (_, diff, lo, hi, ptext, color) in zip(y, tests):
        ax.plot([lo, hi], [yi, yi], color=color, lw=2)
        ax.scatter([diff], [yi], color=color, s=40, zorder=3)
        ax.annotate(ptext, (diff, yi), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.5,
                    color=INK_2)
    ax.axvline(0, color=INK_2, lw=0.9)
    ax.set_yticks(y, [label for label, *_ in tests])
    style(ax, "Planned comparisons: paired difference in requirements met (per grid)" if len(planned) else
          "Paired difference in requirements met (per grid)", "")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("difference vs reference, 95% CI", color=INK_2, fontsize=9)
    families = [f for f in FAMILY_COLORS if f in set(table.family)]   # only those with results
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in families]
    fig.legend(handles + [plt.Line2D([], [], marker="o", ls="", color=INK, ms=4)],
               families + ["one sample (seed)"], loc="upper center", ncol=6, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(title, x=0.01, ha="left", y=0.99, fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=3)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_mechanics(table, data, out):
    legacy = set(data[data.approach == "legacy"].condition)
    t = table[~table.condition.isin(legacy)].reset_index(drop=True)
    panels = [("uncovered", "Rule coverage: uncovered situations, kept policy", "% of reachable", "{:.0f}%"),
              ("feedback_improved", "Feedback rounds that improved the policy", "share", "{:.0%}")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8), facecolor=SURFACE)
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


def plot_costs(table, data, out, num_reqs):
    """Per-grid cost and outcome distributions: median bars with interquartile whiskers."""
    t = table.reset_index(drop=True)
    panels = [("met", f"Requirements met (of {num_reqs}), worst case", "per grid", 1, "{:.1f}"),
              ("shortfall", "Shortfall below thresholds", "per grid", 1, "{:.2f}"),
              ("input_tokens", "LLM input tokens", "thousand per grid", 1e-3, "{:.1f}k"),
              ("output_tokens", "LLM output tokens", "thousand per grid", 1e-3, "{:.1f}k"),
              ("prism_s", "PRISM time", "seconds per grid", 1, "{:.0f}"),
              ("time_s", "Wall time", "minutes per grid", 1 / 60, "{:.1f}")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), facecolor=SURFACE)
    for ax, (col, title, ylabel, k, fmt) in zip(axes.flat, panels):
        x = np.arange(len(t))
        for i, condition in enumerate(t.condition):
            v = data[data.condition == condition][col].dropna() * k
            if not len(v):
                continue
            q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
            ax.bar(i, med, 0.65, color=FAMILY_COLORS[t.family[i]])
            ax.errorbar(i, med, yerr=[[med - q1], [q3 - med]], color=INK, capsize=4, lw=1)
            ax.annotate(fmt.format(med), (i, q3), ha="center", va="bottom", xytext=(0, 2),
                        textcoords="offset points", fontsize=7, color=INK_2)
        ax.set_xticks(x, t.condition)
        style(ax, title, ylabel)
    fig.suptitle("Per grid, all seeds pooled: bars = median, whiskers = interquartile range", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.5)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def budget_curves(runs, cfg: PlotConfig) -> pd.DataFrame:
    """Per run, sample and round budget k: the kept policy's metrics and the spend through round k (with `cost`)."""
    curves = []
    for condition, run_dir in runs:
        legacy = (run_dir / LEGACY_FILE).exists()
        rounds = run_facts(run_dir).max_rounds   # curves stop at the run's own budget
        rows = legacy_curve(run_dir, rounds) if legacy else symbolic_curve(run_dir, rounds)
        curves.append(pd.DataFrame(rows).assign(condition=condition, seed=run_dir.name))
    curves = pd.concat(curves, ignore_index=True)
    return with_cost(curves, cfg.cost) if cfg.cost else curves


def plot_budget(curves, out, cfg: PlotConfig):
    summary = curves.groupby(["condition", "k"]).agg(
        met=("met", "mean"), **({"cost": ("cost", "mean")} if cfg.cost else {})).reset_index()
    groups = list(cfg.budget_groups.items())
    xs = [("k", "rounds budget k")] + ([("cost", f"mean spend through round k ({cfg.cost['label']})")]
                                       if cfg.cost else [])
    fig, axes = plt.subplots(len(xs), len(groups), figsize=(max(7.0, 16 * len(groups) / 3), 4.4 * len(xs)),
                             facecolor=SURFACE, sharey=True, squeeze=False)
    for row, (x, xlabel) in zip(axes, xs):
        for ax, (title, members) in zip(row, groups):
            for i, condition in enumerate([m for m in members if m in set(summary.condition)]):
                d = summary[summary.condition == condition].dropna(subset=[x])   # legacy has no spend per round
                if not len(d):
                    continue
                ls = "--" if condition in cfg.reference_lines and members[0] != condition else "-"
                ax.plot(d[x], d.met, color=LINE_COLORS[i % len(LINE_COLORS)], lw=2, marker="o", ms=5, ls=ls,
                        label=f"{condition}: {cfg.conditions[condition]['change']}")
                ax.annotate(f"{d.met.iloc[-1]:.2f}", (d[x].iloc[-1], d.met.iloc[-1]), xytext=(5, 0),
                            textcoords="offset points", va="center", fontsize=7.5, color=INK_2)
            style(ax, title if x == "k" else f"{title}, by spend", "requirements met (mean per grid)")
            if x == "k":
                ax.set_xticks(range(1, int(summary.k.max()) + 1))
                ax.legend(frameon=False, fontsize=7.5, loc="best")
            ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    fig.suptitle("Requirements met by the kept policy after k rounds (seeds pooled)" +
                 ("; below, each k at the mean spend through round k" if cfg.cost else ""), x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if len(xs) == 1 else 0.95))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return summary.pivot(index="condition", columns="k", values="met")


def _uuv_runs(root: Path, condition: str) -> List[Path]:
    return [p for p in sorted((root / condition).glob("seed_*")) if (p / SYMBOLIC_FILE).exists()]


def _instances(run_dir: Path):
    facts = run_facts(run_dir)
    domain = load_domain(facts.domain)
    return domain, domain.load_instances(facts.dataset)


def _scenario(instance) -> str:
    return instance.data.get("name", instance.id).replace("_", " ").title()


def plot_uuv(root: Path, condition: str, out: Path) -> bool:
    """Per round, how far each UUV requirement is from its threshold (worst case), every scenario."""
    runs = _uuv_runs(root, condition)
    if not runs:
        return False
    domain, instances = _instances(runs[0])
    fig, axes = plt.subplots(1, len(instances), figsize=(14, 4.4), facecolor=SURFACE, sharey=True)
    for ax, inst in zip(axes, instances):
        reqs = domain.spec(inst).requirements
        for j, run_dir in enumerate(runs):
            df = pd.read_parquet(run_dir / SYMBOLIC_FILE).reset_index()
            g = df[df.sample_id == int(inst.id)].sort_values("iteration")
            for i, r in enumerate(reqs):
                margin = [(v - r.threshold) * (1 if r.maximize else -1) / abs(r.threshold) * 100
                          for v in g[f"prob_worst_{r.name}"]]
                ax.plot(g.iteration, margin, color=LINE_COLORS[i], lw=2, marker="o", ms=5,
                        ls="-" if j == 0 else "--", label=f"{r.name} ({run_dir.name})")
        ax.axhline(0, color=INK, lw=1)
        style(ax, _scenario(inst), "margin to threshold, % (above 0 = met)")
        ax.set_xticks(range(1, run_facts(runs[0]).max_rounds + 1))
        ax.set_xlabel("round", color=INK_2, fontsize=9)
    axes[-1].legend(frameon=False, fontsize=7.5, loc="lower right")
    seeds = ", ".join(f"{'solid' if j == 0 else 'dashed'} {r.name.replace('_', ' ')}" for j, r in enumerate(runs[:2]))
    fig.suptitle(f"UUV ({condition}): worst-case margin of each requirement per round ({seeds})",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def uuv_table(root: Path, condition: str) -> str:
    rows = []
    for run_dir in _uuv_runs(root, condition):
        _, instances = _instances(run_dir)
        df = pd.read_parquet(run_dir / SYMBOLIC_FILE).reset_index()
        final = df[df.is_final]
        reqs = [c[len("final_worst_"):] for c in final.columns if c.startswith("final_worst_")]
        for _, f in final.iterrows():
            values = ", ".join(f"{r} {f[f'final_worst_{r}']:.3f}" for r in reqs)
            rows.append(f"| {run_dir.name} | {_scenario(instances[int(f.sample_id)])} | {bool(f.success)} | "
                        f"{int(df[df.sample_id == f.sample_id].iteration.max())} | {int(f.final_num_rules)} | "
                        f"{f.llm_prompt_tokens / 1000:.1f}k / {f.llm_output_tokens / 1000:.1f}k | {values} |")
    if not rows:
        return ""
    return ("| seed | scenario | solved (worst case) | rounds | rules | tokens in / out | final worst-case values |\n"
            "|---|---|---|---|---|---|---|\n" + "\n".join(rows))


def write_markdown(out_dir, table, budget, planned, pending, uuv, cfg: PlotConfig, batch: Batch, config_path: str):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# {cfg.heading} (auto-generated {now})", "", batch.text(cfg.intro), "",
             f"Regenerate: `python viz/ablation_summary.py {config_path}`.", ""]
    if pending:
        lines += ["**Pending runs:** " + ", ".join(pending), ""]
    if len(planned):
        lines += ["## Planned comparisons", "",
                  "Fixed before the runs. Requirements met by the kept policy (worst case, the loop's values), per grid "
                  "averaged over runs; paired sign-flip test over the grids. p (Holm) corrects for all "
                  f"{len(planned)} comparisons together; below 0.05 counts. \"equal cost\": the condition's whole run "
                  "against the reference given, on each grid, the condition's mean spend there"
                  + (f" ({cfg.cost['label']})." if cfg.cost else "."), "",
                  "| comparison | at | grids | Δ met [95% CI] | p | p (Holm) |", "|---|---|---|---|---|---|"]
        lines += [f"| {r.condition} vs {r.reference} | {r.at} | {r.n} | {r['diff']:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] | "
                  f"{r.p:.3f} | {r.p_holm:.3f} |" for _, r in planned.iterrows()]
        lines.append("")
    lines += ["## Overview", "", "![conditions](conditions.png)", "",
              "## Distributions and costs (medians, interquartile range)", "", "![costs](costs.png)", "",
              "## Budget curves", "", "![budget](budget.png)", "",
              "## Loop mechanics (symbolic)", "", "![mechanics](mechanics.png)", "",
              "## Outcomes (per grid, seeds pooled)", "",
              "| cond | change | seeds | solved (of solvable) | req. met: mean (seed range) | median [IQR] | best case | "
              "shortfall: median [IQR] | uncovered % | vs | Δ met [95% CI] | p | feedback rounds improved |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        improved = f"{r.feedback_improved:.0%}" if "feedback_improved" in r and pd.notna(r.feedback_improved) else ""
        diff = f"{r['diff']:+.2f} {r.ci}" if r.vs else ""
        lines.append(f"| {r.condition} | {r.change} | {r.seeds} | {r.solved:g}/{len(batch.solvable)} | "
                     f"{r.met:.2f} ({r.met_seed_min:.2f} to "
                     f"{r.met_seed_max:.2f}) | {r.met_median:.0f} [{r.met_q1:.0f} to {r.met_q3:.0f}] | {r.met_best:.2f} | "
                     f"{r.short_median:.2f} [{r.short_q1:.2f} to {r.short_q3:.2f}] | {r.uncovered:.0f} | {r.vs} | "
                     f"{diff} | {'' if pd.isna(r.p) else f'{r.p:.3f}'} | {improved} |")
    def spend(r):
        if not cfg.cost:
            return ""
        v = 1000 * (r.input_k * cfg.cost["input"] + r.tokens_k * cfg.cost["output"])
        return f" {v:,.0f} |" if v >= 100 else f" {v:.3g} |"
    lines += ["", "## Costs (mean per grid)", "",
              "| cond | input tokens | output tokens | PRISM time (s) | wall time (min) |" +
              (f" spend ({cfg.cost['label']}) |" if cfg.cost else ""), "|---|---|---|---|---|" + ("---|" if cfg.cost else "")]
    lines += [f"| {r.condition} | {r.input_k:.1f}k | {r.tokens_k:.1f}k | {r.prism_s:.0f} | {r.minutes:.1f} |" + spend(r)
              for _, r in table.iterrows()]
    ks = list(budget.columns)
    lines += ["", "## Requirements met after k rounds", "", "| cond | " + " | ".join(f"k={k}" for k in ks) + " |",
              "|---|" + "---|" * len(ks)]
    lines += [f"| {c} | " + " | ".join("" if pd.isna(budget.loc[c, k]) else f"{budget.loc[c, k]:.2f}" for k in ks) + " |"
              for c in cfg.conditions if c in budget.index]
    lines.append("")
    if uuv:
        lines += [f"## UUV ({cfg.uuv['condition']}: {cfg.uuv['change']})", "", "![uuv](uuv.png)", "", uuv, ""]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cfg = plot_config(PlotConfig, "ablation_summary", argv)
    root = Path(cfg.root)
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs, pending = finished_runs(root, cfg.conditions)
    batch = describe(runs)
    data = pd.concat([load_run(c, d, batch.solvable) for c, d in runs], ignore_index=True)
    num_reqs = len([c for c in data.columns if c.startswith("p_") and not c.startswith("p_best_")])
    extra = {}
    for condition, run_dir in runs:
        if (run_dir / SYMBOLIC_FILE).exists():
            extra.setdefault(condition, []).append(mode_stats(run_dir))
    extra = {c: pd.DataFrame(v).mean(numeric_only=True).to_dict() for c, v in extra.items()}
    table = condition_table(data, extra, cfg.conditions)
    table.to_csv(out_dir / "conditions.csv", index=False)
    data.to_csv(out_dir / "per_grid.csv", index=False)

    curves = budget_curves(runs, cfg)
    planned = planned_table(curves, cfg)
    if len(planned):
        planned.to_csv(out_dir / "planned.csv", index=False)

    plot_conditions(table, data, out_dir / "conditions.png", batch.text(cfg.title), num_reqs, cfg.reference_lines,
                    planned, cfg.conditions)
    plot_mechanics(table, data, out_dir / "mechanics.png")
    plot_costs(table, data, out_dir / "costs.png", num_reqs)
    budget = plot_budget(curves, out_dir / "budget.png", cfg)
    uuv = ""
    if cfg.uuv:
        plot_uuv(root, cfg.uuv["condition"], out_dir / "uuv.png")
        uuv = uuv_table(root, cfg.uuv["condition"])
    write_markdown(out_dir, table, budget, planned, pending, uuv, cfg, batch, Path(argv[0]).as_posix())
    print(out_dir / "SUMMARY.md")


if __name__ == "__main__":
    main()
