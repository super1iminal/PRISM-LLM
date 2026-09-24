"""End-to-end comparison of the legacy (per-state) and symbolic approaches.

Usage: python src/compare.py --legacy out/results/legacy_grid20 --symbolic out/results/symbolic_grid20 [--out DIR]

Success means every requirement meets its threshold: for legacy, on the full per-state policy;
for symbolic, in the worst case over all completions of the (possibly partial) rule list.
"""
import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd

from core.domain import load_domain
from legacy.gridworld import GridWorld as LegacyGridWorld
from legacy.requirements import SimplifiedVerifier, get_threshold_for_key

_quiet = logging.getLogger("compare")
_quiet.addHandler(logging.NullHandler())
_quiet.propagate = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--symbolic", required=True)
    parser.add_argument("--data", default="grid_20_balanced.csv", help="Gridworld dataset both runs used")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    instances = load_domain("gridworld").load_instances(args.data)
    legacy_dir, symbolic_dir = Path(args.legacy), Path(args.symbolic)
    out_dir = Path(args.out) if args.out else symbolic_dir.parent / "comparison"
    os.makedirs(out_dir, exist_ok=True)

    leg_all = pd.read_parquet(legacy_dir / "LEGACY_FEEDBACK_SIMPLIFIED_results.parquet").reset_index()
    sym_all = pd.read_parquet(symbolic_dir / "SYMBOLIC_results.parquet").reset_index()
    leg = leg_all[leg_all.is_final].set_index("sample_id")
    sym = sym_all[sym_all.is_final].set_index("sample_id")

    # The legacy parquet's final row is the last iteration; its outcome is the kept-best policy.
    leg_final_probs = {}
    for path in sorted((legacy_dir / "outputs").glob("sample_*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        leg_final_probs[rec["sample_id"]] = _legacy_kept(rec, instances[rec["sample_id"]])

    requirements = sorted({c[len("final_worst_"):] for c in sym.columns if c.startswith("final_worst_")})
    samples = sorted(set(leg.index) | set(sym.index))

    rows = []
    for sid in samples:
        row = {"sample_id": sid}
        if sid in leg.index:
            l = leg.loc[sid]
            row.update({"size": l["size"], "legacy_success": bool(l["success"]),
                        "legacy_iterations": int(leg_all[leg_all.sample_id == sid].iteration.max()),
                        "legacy_time": l["total_time"], "legacy_llm_calls": l["llm_calls"],
                        "legacy_output_tokens": l["llm_output_tokens"], "legacy_prompt_tokens": l["llm_prompt_tokens"],
                        "legacy_llm_time": leg_all[leg_all.sample_id == sid].llm_time.sum(),
                        "legacy_prism_time": leg_all[leg_all.sample_id == sid].prism_time.sum()})
            row.update({f"legacy_{k}": p for k, p in leg_final_probs.get(sid, {}).items()})
        if sid in sym.index:
            s = sym.loc[sid]
            row.update({"symbolic_success": bool(s["success"]), "symbolic_best_case_success": bool(s["best_case_success"]),
                        "symbolic_iterations": int(sym_all[sym_all.sample_id == sid].iteration.max()),
                        "symbolic_time": s["total_time"], "symbolic_llm_calls": s["llm_calls"],
                        "symbolic_output_tokens": s["llm_output_tokens"], "symbolic_prompt_tokens": s["llm_prompt_tokens"],
                        "symbolic_llm_time": sym_all[sym_all.sample_id == sid].llm_time.sum(),
                        "symbolic_prism_time": sym_all[sym_all.sample_id == sid].prism_time.sum(),
                        "symbolic_rules": s["final_num_rules"],
                        "symbolic_invalid_answers": sym_all[sym_all.sample_id == sid].invalid_answers.sum()})
            row.update({f"symbolic_worst_{k}": s.get(f"final_worst_{k}") for k in requirements})
            row.update({f"symbolic_best_{k}": s.get(f"final_best_{k}") for k in requirements})
            row.update({f"optimum_{k}": s.get(f"optimum_{k}") for k in requirements})
        for prefix in ("legacy_", "symbolic_worst_", "symbolic_best_"):
            probs = {k: row[prefix + k] for k in requirements if row.get(prefix + k) is not None}
            if probs:
                row[prefix + "met"] = sum(p >= get_threshold_for_key(k) for k, p in probs.items())
                row[prefix + "shortfall"] = sum(max(0.0, get_threshold_for_key(k) - p) for k, p in probs.items())
        rows.append(row)
    per_sample = pd.DataFrame(rows).set_index("sample_id")
    per_sample.to_csv(out_dir / "per_sample.csv")

    n = len(samples)

    def rate(col):
        return f"{int(per_sample[col].fillna(False).sum())}/{n}" if col in per_sample else "n/a"

    def mean(col, fmt="{:.1f}"):
        return fmt.format(per_sample[col].mean()) if col in per_sample else "n/a"

    lines = [
        "# Legacy vs symbolic: end-to-end comparison",
        "",
        f"- Legacy run: `{legacy_dir}`",
        f"- Symbolic run: `{symbolic_dir}`",
        f"- Samples: {n}",
        "",
        "Success = all requirements meet their thresholds (symbolic: in the **worst case** over all completions).",
        "",
        "| metric | legacy (per-state) | symbolic (rules) |",
        "|---|---|---|",
        f"| success | {rate('legacy_success')} | {rate('symbolic_success')} |",
        f"| success in best case only | n/a | {rate('symbolic_best_case_success')} |",
        f"| mean requirements met (of {len(requirements)}) | {mean('legacy_met', '{:.2f}')} | "
        f"{mean('symbolic_worst_met', '{:.2f}')} (best case {mean('symbolic_best_met', '{:.2f}')}) |",
        f"| mean total shortfall below thresholds | {mean('legacy_shortfall', '{:.3f}')} | "
        f"{mean('symbolic_worst_shortfall', '{:.3f}')} (best case {mean('symbolic_best_shortfall', '{:.3f}')}) |",
        f"| mean iterations | {mean('legacy_iterations', '{:.2f}')} | {mean('symbolic_iterations', '{:.2f}')} |",
        f"| mean LLM calls | {mean('legacy_llm_calls')} | {mean('symbolic_llm_calls')} |",
        f"| mean output tokens | {mean('legacy_output_tokens', '{:.0f}')} | {mean('symbolic_output_tokens', '{:.0f}')} |",
        f"| mean prompt tokens | {mean('legacy_prompt_tokens', '{:.0f}')} | {mean('symbolic_prompt_tokens', '{:.0f}')} |",
        f"| mean LLM time (s) | {mean('legacy_llm_time')} | {mean('symbolic_llm_time')} |",
        f"| mean PRISM time (s) | {mean('legacy_prism_time')} | {mean('symbolic_prism_time')} |",
        f"| mean wall time per sample (s) | {mean('legacy_time')} | {mean('symbolic_time')} |",
        f"| mean final rules | n/a | {mean('symbolic_rules')} |",
        f"| invalid LLM answers (total) | n/a | {int(per_sample['symbolic_invalid_answers'].sum()) if 'symbolic_invalid_answers' in per_sample else 'n/a'} |",
        "",
        "## Mean final probability per requirement",
        "",
        "| requirement | legacy | symbolic worst | symbolic best | optimum (bare MDP) |",
        "|---|---|---|---|---|",
    ]
    for k in requirements:
        lines.append(f"| `{k}` | {mean('legacy_' + k, '{:.3f}')} | {mean('symbolic_worst_' + k, '{:.3f}')} | "
                     f"{mean('symbolic_best_' + k, '{:.3f}')} | {mean('optimum_' + k, '{:.3f}')} |")

    if "size" in per_sample:
        lines += ["", "## By grid size", "",
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

    report = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


def _legacy_kept(record, instance) -> dict:
    """Probabilities of the policy the legacy keep-best loop ended with.

    Newer legacy runs store them directly. Otherwise, replay the rule: fewest failed
    requirements, ties broken by the higher weighted score, earliest iteration first.
    """
    if record.get("final_prism_probs"):
        return record["final_prism_probs"]
    iterations = record.get("iteration_prism_probs", [])
    if not iterations:
        return {}
    d = instance.data
    verifier = SimplifiedVerifier(None, LegacyGridWorld(d["n"], d["goals"], d["static"], d["moving"]), _quiet)
    best, best_key = iterations[0], None
    for probs in iterations:
        mistakes = sum(1 for k, p in probs.items() if p < get_threshold_for_key(k))
        key = (mistakes, -verifier._calculate_score(list(probs.values())))
        if best_key is None or key < best_key:
            best, best_key = probs, key
    return best


if __name__ == "__main__":
    main()
