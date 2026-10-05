"""End-to-end comparison of the legacy (per-state) and symbolic approaches.

Usage: python src/compare.py configs/plot/compare_grid20.yaml [--set key=value ...]
Settings: the plot config (PlotConfig below); the dataset comes from the runs.

Success means every requirement meets its threshold: for legacy, on the full per-state policy;
for symbolic, in the worst case over all completions of the (possibly partial) rule list.
With per-instance ceilings (src/ceilings.py, picked up automatically), the report also gives
solved-of-solvable and the shortfall below what is actually achievable (min(threshold, optimum)).
"""
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import plot_config
from core.domain import Requirement, load_domain
from results_io import (LEGACY_RESULTS, SYMBOLIC_RESULTS, legacy_kept, met_and_shortfall, requirements_by_sample,
                        run_facts)
from settings import RESULTS_PATH

PREFIXES = ("legacy_", "symbolic_worst_", "symbolic_best_")   # probability columns that get metrics


@dataclass
class PlotConfig:
    """configs/plot/compare_grid20.yaml documents each key."""
    script: str
    legacy: str
    symbolic: str
    ceilings: Optional[str]
    out: str


def main():
    cfg = plot_config(PlotConfig, "compare")
    legacy_dir, symbolic_dir, out_dir = Path(cfg.legacy), Path(cfg.symbolic), Path(cfg.out)
    dataset = run_facts(symbolic_dir).dataset
    if run_facts(legacy_dir).dataset != dataset:
        raise SystemExit(f"the runs used different datasets ({run_facts(legacy_dir).dataset} vs {dataset})")
    ceilings_path = Path(cfg.ceilings) if cfg.ceilings else RESULTS_PATH / "ceilings" / f"gridworld_{Path(dataset).stem}.csv"
    os.makedirs(out_dir, exist_ok=True)

    ceilings = pd.read_csv(ceilings_path, index_col="sample_id") if ceilings_path.exists() else None
    per_sample, requirements = per_sample_table(legacy_dir, symbolic_dir, ceilings)
    per_sample.to_csv(out_dir / "per_sample.csv")
    report = render_report(per_sample, requirements, legacy_dir, symbolic_dir)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


# ---------------------------------------------------------------- per-sample table

def per_sample_table(legacy_dir: Path, symbolic_dir: Path,
                     ceilings: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    """One row per sample of either run: outcomes, costs and metrics of both. Also returns the
    requirement names (sorted)."""
    instances = load_domain("gridworld").load_instances(run_facts(symbolic_dir).dataset)
    leg_all = pd.read_parquet(legacy_dir / LEGACY_RESULTS).reset_index()
    sym_all = pd.read_parquet(symbolic_dir / SYMBOLIC_RESULTS).reset_index()
    leg = leg_all[leg_all.is_final].set_index("sample_id")
    sym = sym_all[sym_all.is_final].set_index("sample_id")

    # The legacy parquet's final row is the last iteration; its outcome is the kept-best policy.
    leg_final_probs = {}
    for path in sorted((legacy_dir / "outputs").glob("sample_*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        leg_final_probs[rec["sample_id"]] = legacy_kept(rec, instances[rec["sample_id"]])

    requirements = sorted({c[len("final_worst_"):] for c in sym.columns if c.startswith("final_worst_")})
    by_sample = requirements_by_sample(symbolic_dir)
    rows = []
    for sid in sorted(set(leg.index) | set(sym.index)):
        row = {"sample_id": sid}
        if sid in leg.index:
            row.update(_legacy_columns(leg.loc[sid], leg_all[leg_all.sample_id == sid], leg_final_probs.get(sid, {})))
        if sid in sym.index:
            row.update(_symbolic_columns(sym.loc[sid], sym_all[sym_all.sample_id == sid], requirements))
        if ceilings is not None and sid in ceilings.index:
            row["solvable"] = bool(ceilings.loc[sid, "jointly_feasible"] == True)  # noqa: E712
        optimum = ceilings.loc[sid] if ceilings is not None and sid in ceilings.index else None
        row.update(_metric_columns(row, requirements, by_sample[sid], optimum))
        rows.append(row)
    return pd.DataFrame(rows).set_index("sample_id"), requirements


def _legacy_columns(final: pd.Series, rounds: pd.DataFrame, kept_probs: Dict[str, float]) -> Dict:
    return {"size": final["size"], "legacy_success": bool(final["success"]),
            "legacy_iterations": int(rounds.iteration.max()),
            "legacy_time": final["total_time"], "legacy_llm_calls": final["llm_calls"],
            "legacy_output_tokens": final["llm_output_tokens"], "legacy_prompt_tokens": final["llm_prompt_tokens"],
            "legacy_llm_time": rounds.llm_time.sum(), "legacy_prism_time": rounds.prism_time.sum(),
            **{f"legacy_{k}": p for k, p in kept_probs.items()}}


def _symbolic_columns(final: pd.Series, rounds: pd.DataFrame, requirements: List[str]) -> Dict:
    return {"symbolic_success": bool(final["success"]), "symbolic_best_case_success": bool(final["best_case_success"]),
            "symbolic_iterations": int(rounds.iteration.max()),
            "symbolic_time": final["total_time"], "symbolic_llm_calls": final["llm_calls"],
            "symbolic_output_tokens": final["llm_output_tokens"], "symbolic_prompt_tokens": final["llm_prompt_tokens"],
            "symbolic_llm_time": rounds.llm_time.sum(), "symbolic_prism_time": rounds.prism_time.sum(),
            "symbolic_rules": final["final_num_rules"],
            "symbolic_uncovered_pct": 100 * (final["uncovered_situations"] / final["reachable_situations"]
                                             if final["reachable_situations"] else 0.0),
            "symbolic_invalid_answers": rounds.invalid_answers.sum(),
            **{f"symbolic_worst_{k}": final.get(f"final_worst_{k}") for k in requirements},
            **{f"symbolic_best_{k}": final.get(f"final_best_{k}") for k in requirements},
            **{f"optimum_{k}": final.get(f"optimum_{k}") for k in requirements}}


def _metric_columns(row: Dict, requirements: List[str], reqs: List[Requirement],
                    optimum: Optional[pd.Series]) -> Dict:
    """Requirements met and shortfall for each set of probabilities in `row`; with the sample's
    ceilings (`optimum`), also the shortfall below what is achievable."""
    out = {}
    for prefix in PREFIXES:
        probs = {k: row[prefix + k] for k in requirements if row.get(prefix + k) is not None}
        if not probs:
            continue
        out[prefix + "met"], out[prefix + "shortfall"] = met_and_shortfall(reqs, probs)
        if optimum is not None:
            # Achievable: the threshold, or the optimum where the threshold is out of reach
            achievable = [replace(r, threshold=(min if r.maximize else max)(r.threshold, optimum[f"optimum_{r.name}"]))
                          for r in reqs if r.name in probs]
            out[prefix + "achievable_shortfall"] = met_and_shortfall(achievable, probs)[1]
    return out


# ---------------------------------------------------------------- report

def render_report(per_sample: pd.DataFrame, requirements: List[str], legacy_dir: Path, symbolic_dir: Path) -> str:
    stats = _Stats(per_sample)
    lines = [
        "# Legacy vs symbolic: end-to-end comparison",
        "",
        f"- Legacy run: `{legacy_dir}`",
        f"- Symbolic run: `{symbolic_dir}`",
        f"- Samples: {len(per_sample)}",
        "",
        "Success = all requirements meet their thresholds (symbolic: in the **worst case** over all completions).",
        "",
    ]
    lines += _overview(stats, len(requirements)) + [""] + _medians(stats) + [""] + _per_requirement(stats, requirements)
    if "size" in per_sample:
        lines += [""] + _by_size(per_sample)
    return "\n".join(lines) + "\n"


class _Stats:
    """Formatted summaries of per-sample columns ("n/a" for a column the table lacks)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def rate(self, col):
        return f"{int(self.df[col].fillna(False).sum())}/{len(self.df)}" if col in self.df else "n/a"

    def mean(self, col, fmt="{:.1f}"):
        return fmt.format(self.df[col].mean()) if col in self.df else "n/a"

    def median(self, col, fmt="{:.1f}"):
        """Median [interquartile range] over samples."""
        if col not in self.df:
            return "n/a"
        q = self.df[col].quantile([0.25, 0.5, 0.75])
        return f"{fmt.format(q[0.5])} [{fmt.format(q[0.25])}–{fmt.format(q[0.75])}]"

    def solved_of_solvable(self, col):
        if "solvable" not in self.df or col not in self.df:
            return "n/a"
        solvable = self.df[self.df.solvable]
        return f"{int(solvable[col].fillna(False).sum())}/{len(solvable)}"

    def total(self, col):
        return int(self.df[col].sum()) if col in self.df else "n/a"


def _overview(s: _Stats, num_requirements: int) -> List[str]:
    return [
        "| metric | legacy (per-state) | symbolic (rules) |",
        "|---|---|---|",
        f"| success | {s.rate('legacy_success')} | {s.rate('symbolic_success')} |",
        f"| success, of jointly solvable instances | {s.solved_of_solvable('legacy_success')} | "
        f"{s.solved_of_solvable('symbolic_success')} |",
        f"| success in best case only | n/a | {s.rate('symbolic_best_case_success')} |",
        f"| shortfall below the achievable (mean) | {s.mean('legacy_achievable_shortfall', '{:.3f}')} | "
        f"{s.mean('symbolic_worst_achievable_shortfall', '{:.3f}')} |",
        f"| mean requirements met (of {num_requirements}) | {s.mean('legacy_met', '{:.2f}')} | "
        f"{s.mean('symbolic_worst_met', '{:.2f}')} (best case {s.mean('symbolic_best_met', '{:.2f}')}) |",
        f"| mean total shortfall below thresholds | {s.mean('legacy_shortfall', '{:.3f}')} | "
        f"{s.mean('symbolic_worst_shortfall', '{:.3f}')} (best case {s.mean('symbolic_best_shortfall', '{:.3f}')}) |",
        f"| mean iterations | {s.mean('legacy_iterations', '{:.2f}')} | {s.mean('symbolic_iterations', '{:.2f}')} |",
        f"| mean LLM calls | {s.mean('legacy_llm_calls')} | {s.mean('symbolic_llm_calls')} |",
        f"| mean output tokens | {s.mean('legacy_output_tokens', '{:.0f}')} | {s.mean('symbolic_output_tokens', '{:.0f}')} |",
        f"| mean prompt tokens | {s.mean('legacy_prompt_tokens', '{:.0f}')} | {s.mean('symbolic_prompt_tokens', '{:.0f}')} |",
        f"| mean LLM time (s) | {s.mean('legacy_llm_time')} | {s.mean('symbolic_llm_time')} |",
        f"| mean PRISM time (s) | {s.mean('legacy_prism_time')} | {s.mean('symbolic_prism_time')} |",
        f"| mean wall time per sample (s) | {s.mean('legacy_time')} | {s.mean('symbolic_time')} |",
        f"| mean final rules | n/a | {s.mean('symbolic_rules')} |",
        f"| uncovered situations, last round's policy (mean %) | n/a | {s.mean('symbolic_uncovered_pct')} |",
        f"| invalid LLM answers (total) | n/a | {s.total('symbolic_invalid_answers')} |",
    ]


def _medians(s: _Stats) -> List[str]:
    return [
        "## Medians [interquartile range] per sample",
        "",
        "| metric | legacy | symbolic |",
        "|---|---|---|",
        f"| requirements met | {s.median('legacy_met', '{:.0f}')} | {s.median('symbolic_worst_met', '{:.0f}')} |",
        f"| shortfall | {s.median('legacy_shortfall', '{:.2f}')} | {s.median('symbolic_worst_shortfall', '{:.2f}')} |",
        f"| output tokens | {s.median('legacy_output_tokens', '{:.0f}')} | {s.median('symbolic_output_tokens', '{:.0f}')} |",
        f"| prompt (input) tokens | {s.median('legacy_prompt_tokens', '{:.0f}')} | "
        f"{s.median('symbolic_prompt_tokens', '{:.0f}')} |",
        f"| PRISM time (s) | {s.median('legacy_prism_time')} | {s.median('symbolic_prism_time')} |",
        f"| wall time (s) | {s.median('legacy_time', '{:.0f}')} | {s.median('symbolic_time', '{:.0f}')} |",
    ]


def _per_requirement(s: _Stats, requirements: List[str]) -> List[str]:
    lines = ["## Mean final probability per requirement", "",
             "| requirement | legacy | symbolic worst | symbolic best | optimum (bare MDP) |", "|---|---|---|---|---|"]
    for k in requirements:
        lines.append(f"| `{k}` | {s.mean('legacy_' + k, '{:.3f}')} | {s.mean('symbolic_worst_' + k, '{:.3f}')} | "
                     f"{s.mean('symbolic_best_' + k, '{:.3f}')} | {s.mean('optimum_' + k, '{:.3f}')} |")
    return lines


def _by_size(per_sample: pd.DataFrame) -> List[str]:
    lines = ["## By grid size", "",
             "| size | legacy success | symbolic success | legacy req. met | symbolic req. met (worst) | "
             "legacy output tokens | symbolic output tokens |",
             "|---|---|---|---|---|---|---|"]
    for size, grp in per_sample.groupby("size"):
        sym_ok = int(grp.symbolic_success.fillna(False).sum()) if "symbolic_success" in grp else 0
        lines.append(f"| {int(size)} | {int(grp.legacy_success.fillna(False).sum())}/{len(grp)} | {sym_ok}/{len(grp)} | "
                     f"{grp.get('legacy_met', pd.Series(dtype=float)).mean():.2f} | "
                     f"{grp.get('symbolic_worst_met', pd.Series(dtype=float)).mean():.2f} | "
                     f"{grp.get('legacy_output_tokens', pd.Series(dtype=float)).mean():.0f} | "
                     f"{grp.get('symbolic_output_tokens', pd.Series(dtype=float)).mean():.0f} |")
    return lines


if __name__ == "__main__":
    main()
