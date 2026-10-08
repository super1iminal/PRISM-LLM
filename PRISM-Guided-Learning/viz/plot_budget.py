"""Success vs budget (ablation F1): what each run would have achieved with only k rounds.

Nothing in rounds 1..k depends on later rounds, so the result with budget k is the policy the
keep-best rule holds after round k. No extra runs are needed.

With `cost` set, the x axis is spend instead: each round budget k is placed at the mean cumulative cost
through round k (runs that stopped early add nothing after their last round), so a condition whose rounds
are cheaper reaches further left. Symbolic runs only: legacy runs don't log tokens per round.

Usage (from PRISM-Guided-Learning/):
    python viz/plot_budget.py configs/plot/budget_grid20.yaml
Settings: the plot config (PlotConfig below); a run can be a condition dir, whose seed_* runs are pooled.
Writes the figure and a CSV (same stem).
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from loaders import load_domain  # noqa: E402
from config import plot_config  # noqa: E402
from results_io import (LEGACY_RESULTS, SYMBOLIC_RESULTS, legacy_kept, met_and_shortfall,  # noqa: E402
                        requirements_by_sample, run_facts)
from theme import GRID, INK, INK_2, SURFACE  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


@dataclass
class PlotConfig:
    """A configs/plot/*.yaml file for this script (configs/plot/budget_grid20.yaml documents each key)."""
    script: str
    runs: Dict[str, str]
    max_k: Optional[int]
    cost: Optional[Dict[str, Any]]
    out: str


def run_dirs(path: Path):
    """A run directory, or every seed_* run under a condition directory."""
    seeds = sorted(p for p in path.glob("seed_*") if p.is_dir())
    return [(p.name, p) for p in seeds] or [("", path)]


def _metrics(probs: dict, requirements) -> dict:
    met, shortfall = met_and_shortfall(requirements, probs)
    return {"met": met, "success": met == len(probs),
            "shortfall": shortfall}


def symbolic_curve(run_dir: Path, max_k: int) -> list:
    df = pd.read_parquet(run_dir / SYMBOLIC_RESULTS).reset_index().sort_values(["sample_id", "iteration"])
    reqs = [c[len("prob_worst_"):] for c in df.columns if c.startswith("prob_worst_")]
    by_sample = requirements_by_sample(run_dir)
    rows = []
    for sid, g in df.groupby("sample_id"):
        kept, key, spent_in, spent_out = None, None, 0.0, 0.0
        for k in range(1, max_k + 1):
            r = g[g.iteration == k]
            if len(r):   # a round that ran (runs stop early on success)
                r = r.iloc[0]
                spent_in += r.get("iter_llm_prompt_tokens", np.nan)
                spent_out += r.get("iter_llm_output_tokens", np.nan)
                worst = {q: r[f"prob_worst_{q}"] for q in reqs}
                best = {q: r[f"prob_best_{q}"] for q in reqs}
                score = (sum(not _metrics({q: worst[q]}, by_sample[sid])["success"] for q in reqs),
                         sum(not _metrics({q: best[q]}, by_sample[sid])["success"] for q in reqs),
                         _metrics(worst, by_sample[sid])["shortfall"], _metrics(best, by_sample[sid])["shortfall"])
                if key is None or score < key:
                    kept, key = worst, score
            rows.append({"sample_id": sid, "k": k, **_metrics(kept, by_sample[sid]),
                         "input_tokens": spent_in, "output_tokens": spent_out})
    return rows


def legacy_curve(run_dir: Path, max_k: int) -> list:
    instances = load_domain("gridworld").load_instances(run_facts(run_dir).dataset)
    by_sample = requirements_by_sample(run_dir)
    rows = []
    for path in sorted((run_dir / "outputs").glob("sample_*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        probs = rec.get("iteration_prism_probs", [])
        for k in range(1, max_k + 1):
            prefix = {"iteration_prism_probs": probs[:k]}   # no final_prism_probs: replay keep-best on k rounds
            rows.append({"sample_id": rec["sample_id"], "k": k, **_metrics(legacy_kept(prefix, instances[rec["sample_id"]]), by_sample[rec["sample_id"]]),
                         "input_tokens": np.nan, "output_tokens": np.nan})   # legacy logs only per-run totals
    return rows


def with_cost(curve: pd.DataFrame, cost: Dict[str, Any]) -> pd.DataFrame:
    """The curve with a `cost` column: spend through round k, input tokens × cost['input'] + output × cost['output']."""
    return curve.reset_index(drop=True).assign(
        cost=curve.input_tokens.to_numpy() * cost["input"] + curve.output_tokens.to_numpy() * cost["output"])


def at_budget(curve: pd.DataFrame, budget, keys: Sequence[str]) -> pd.DataFrame:
    """Per `keys` group (a run's sample), the row of the last round whose cumulative cost fits `budget` (a number,
    or a Series by sample_id). Every group keeps at least its first round, whose prompt is the same in every
    condition. A run that used all its rounds below the budget keeps its last round."""
    limit = curve.sample_id.map(budget) if isinstance(budget, pd.Series) else budget
    fits = curve[(curve.cost <= limit) | (curve.k == 1)]
    return fits.loc[fits.groupby(list(keys)).k.idxmax()]


def main():
    cfg = plot_config(PlotConfig, "plot_budget")
    out = Path(cfg.out)
    runs = [(label, seed, run_dir) for label, path in cfg.runs.items() for seed, run_dir in run_dirs(Path(path))]
    max_k = cfg.max_k or max(run_facts(run_dir).max_rounds for _, _, run_dir in runs)

    frames = []
    for label, seed, run_dir in runs:
        legacy = (run_dir / LEGACY_RESULTS).exists()
        rows = legacy_curve(run_dir, max_k) if legacy else symbolic_curve(run_dir, max_k)
        frames.append(pd.DataFrame(rows).assign(run=label, seed=seed))
    data = pd.concat(frames)
    x = "k"
    if cfg.cost:
        data, x = with_cost(data, cfg.cost).dropna(subset=["cost"]), "cost"   # drops legacy runs (no per-round tokens)
    summary = data.groupby(["run", "k"]).agg(success=("success", "mean"), met=("met", "mean"),
                                             shortfall=("shortfall", "mean"), n=("success", "size"),
                                             **({"cost": ("cost", "mean")} if cfg.cost else {})).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=SURFACE)
    panels = [("success", "Solved (worst case for symbolic)", "share of instances", "{:.0%}"),
              ("met", "Requirements met", "mean per instance", "{:.2f}"),
              ("shortfall", "Shortfall below thresholds (lower is better)", "mean per instance", "{:.2f}")]
    labels = [label for label in cfg.runs if label in set(summary.run)]
    for ax, (col, title, ylabel, fmt) in zip(axes, panels):
        for i, label in enumerate(labels):
            d = summary[summary.run == label]
            ax.plot(d[x], d[col], color=COLORS[i % len(COLORS)], lw=2, marker="o", ms=6, label=label)
            last = d.iloc[-1]
            ax.annotate(fmt.format(last[col]), (last[x], last[col]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_xlabel(f"mean spend after k rounds ({cfg.cost['label']})" if cfg.cost else "rounds budget k",
                      color=INK_2, fontsize=9)
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
        if not cfg.cost:
            ax.set_xticks(range(1, max_k + 1))
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_2, labelsize=9)
    axes[0].set_ylim(0, max(0.1, summary.success.max() * 1.15))   # a share: never below 0, readable when all 0
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Result vs spend: kept policy after k rounds, at the mean spend through round k" if cfg.cost else
                 "Result vs rounds budget (keep-best policy after k rounds)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    summary.to_csv(out.with_suffix(".csv"), index=False)
    print(summary.to_string(index=False))
    print(out)


if __name__ == "__main__":
    main()
