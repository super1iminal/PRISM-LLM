"""Success vs budget (ablation F1): what each run would have achieved with only k rounds.

Nothing in rounds 1..k depends on later rounds, so the result with budget k is the policy the
keep-best rule holds after round k. No extra runs are needed.

Usage (from PRISM-Guided-Learning/):
    python viz/plot_budget.py "Legacy=out/results/legacy_grid20" "Symbolic=out/results/symbolic_grid20_capped"
    python viz/plot_budget.py "B2=out/results/ablations/B2" "R5=out/results/ablations/R5"   # pools seed_* dirs
Writes viz/figures/budget.png (+ .csv) unless --out is given.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from loaders import load_domain  # noqa: E402
from results_io import (LEGACY_RESULTS, SYMBOLIC_RESULTS, legacy_kept, met_and_shortfall,  # noqa: E402
                        requirements_by_sample, run_facts)
from theme import GRID, INK, INK_2, SURFACE  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


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
        kept, key = None, None
        for k in range(1, max_k + 1):
            r = g[g.iteration == k]
            if len(r):   # a round that ran (runs stop early on success)
                r = r.iloc[0]
                worst = {q: r[f"prob_worst_{q}"] for q in reqs}
                best = {q: r[f"prob_best_{q}"] for q in reqs}
                score = (sum(not _metrics({q: worst[q]}, by_sample[sid])["success"] for q in reqs),
                         sum(not _metrics({q: best[q]}, by_sample[sid])["success"] for q in reqs),
                         _metrics(worst, by_sample[sid])["shortfall"], _metrics(best, by_sample[sid])["shortfall"])
                if key is None or score < key:
                    kept, key = worst, score
            rows.append({"sample_id": sid, "k": k, **_metrics(kept, by_sample[sid])})
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
            rows.append({"sample_id": rec["sample_id"], "k": k, **_metrics(legacy_kept(prefix, instances[rec["sample_id"]]), by_sample[rec["sample_id"]])})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="label=path (a run dir, or a condition dir with seed_* runs)")
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("viz/figures/budget.png"))
    args = parser.parse_args()

    frames = []
    for spec in args.runs:
        label, _, path = spec.partition("=")
        for seed, run_dir in run_dirs(Path(path)):
            legacy = (run_dir / LEGACY_RESULTS).exists()
            rows = legacy_curve(run_dir, args.max_k) if legacy else symbolic_curve(run_dir, args.max_k)
            frames.append(pd.DataFrame(rows).assign(run=label, seed=seed))
    data = pd.concat(frames)
    summary = data.groupby(["run", "k"]).agg(success=("success", "mean"), met=("met", "mean"),
                                             shortfall=("shortfall", "mean"), n=("success", "size")).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=SURFACE)
    panels = [("success", "Solved (worst case for symbolic)", "share of instances", "{:.0%}"),
              ("met", "Requirements met", "mean per instance", "{:.2f}"),
              ("shortfall", "Shortfall below thresholds (lower is better)", "mean per instance", "{:.2f}")]
    labels = list(dict.fromkeys(s.partition("=")[0] for s in args.runs))
    for ax, (col, title, ylabel, fmt) in zip(axes, panels):
        for i, label in enumerate(labels):
            d = summary[summary.run == label]
            ax.plot(d.k, d[col], color=COLORS[i % len(COLORS)], lw=2, marker="o", ms=6, label=label)
            last = d.iloc[-1]
            ax.annotate(fmt.format(last[col]), (last.k, last[col]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_xlabel("rounds budget k", color=INK_2, fontsize=9)
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
        ax.set_xticks(range(1, args.max_k + 1))
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_2, labelsize=9)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Result vs rounds budget (keep-best policy after k rounds)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    summary.to_csv(args.out.with_suffix(".csv"), index=False)
    print(summary.to_string(index=False))
    print(args.out)


if __name__ == "__main__":
    main()
