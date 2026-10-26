"""Report page of a sequential adaptation study (src/sequential.py): per mode and sequence, the decisions taken, the
episodes served (by a certified rule list, or by a strategy synthesized for the instance alone), generations and their
outcomes, and what each decision cost.

Usage (from artifact/): python src/sequential_report.py configs/sequential/<study>.yaml
Writes out/results/sequential/<study>/REPORT.md.
"""
import argparse
from pathlib import Path

import pandas as pd

from sequential import OUT_ROOT, load_study

PRICES = (0.10, 0.50)    # Haiku 5.5, $ per M prompt and output tokens (configs/experiments/rq1_comparison.yaml)


def md(df: pd.DataFrame, index: bool = False) -> str:
    """A markdown table (pandas' to_markdown needs tabulate, which the venv does not have)."""
    if index:
        df = df.reset_index()
    cells = lambda row: "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
    return "\n".join([cells(df.columns), "|" + "---|" * len(df.columns)] + [cells(r) for r in df.itertuples(index=False)])


def load(study_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(study_dir.glob("*/*/episodes.csv")):
        frames.append(pd.read_csv(path).assign(mode=path.parent.parent.name, sequence=path.parent.name))
    return pd.concat(frames, ignore_index=True)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, seq), g in df.groupby(["mode", "sequence"]):
        gen = g[g.get("new_outcome", pd.Series(index=g.index, dtype=object)).notna()] if "new_outcome" in g else g.iloc[0:0]
        tokens = (g.get("prompt_tokens", 0) * PRICES[0] + g.get("output_tokens", 0) * PRICES[1]) / 1e6
        rows.append({"mode": mode, "sequence": seq, "episodes": len(g),
                     "served": int((g.decision != "defer").sum()),
                     "retain": int((g.decision == "retain").sum()), "switch": int((g.decision == "switch").sum()),
                     "generate": int((g.decision == "generate").sum()),
                     "synthesize": int((g.decision == "synthesize").sum()), "defer": int((g.decision == "defer").sum()),
                     "generations": len(gen), "generations certified": int((gen.get("new_outcome") == "certified").sum())
                     if len(gen) else 0,
                     "library at end": int(g.library_size.iloc[-1]),
                     "LLM $": round(float(pd.Series(tokens).fillna(0).sum()), 3)})
    return pd.DataFrame(rows)


def costs(df: pd.DataFrame) -> pd.DataFrame:
    """Seconds per mode and decision: certification and the whole episode; then, over every generation run (the
    certified and the failed ones), its total and its loop's summed LLM and PRISM time."""
    out = df.groupby(["mode", "decision"]).agg(episodes=("episode", "size"), checks=("checks", "mean"),
                                               certify_s=("certify_seconds", "mean"), total_s=("seconds", "mean"))
    if "generation_seconds" in df:
        gen = df[df.generation_seconds.notna()]
        out.loc[("growing", "(generation runs)"), ["episodes", "total_s", "llm_s", "prism_s"]] = [
            len(gen), gen.generation_seconds.mean(), gen.loop_llm_seconds.mean(), gen.loop_prism_seconds.mean()]
    return out.round(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", type=Path)
    args = parser.parse_args(argv)
    study = load_study(args.study)
    study_dir = OUT_ROOT / study.name
    df = load(study_dir)
    checks = pd.concat([pd.read_csv(p).assign(mode=p.parent.parent.name, sequence=p.parent.name)
                        for p in sorted(study_dir.glob("*/*/checks.csv"))], ignore_index=True)
    lines = [f"# Sequential adaptation study `{study.name}`", "",
             f"Config: `configs/sequential/{args.study.name}`. Seed library: `{study.seed_run}` (rule lists with "
             f"`any` resolved). Generation: condition `{study.condition}` on `{study.dataset}`. Policy: retain the "
             "current rule list if certified, else switch to the certified library list with the largest smallest "
             "margin, else generate (mode `growing`) on the instance plus the closest list's training instances, "
             f"else {'synthesize a strategy for the instance alone and deploy it if certified (one requirement only: PRISM cannot export a strategy for several), else ' if study.fallback == 'synthesize' else ''}"
             "defer (runs from before the synthesis fallback have no `synthesize` episodes). "
             "Certification: interval iteration; within `prism.exact_epsilon` of a threshold, or from "
             "the Gauss-Seidel fallback, is inconclusive. Validation instances only.", "",
             "## Decisions per sequence", "", md(summary(df)), "",
             "## Certification outcomes", "",
             md(checks.groupby(["mode", "outcome"]).size().unstack(fill_value=0), index=True), "",
             "Solvers used: " + ", ".join(f"{k}: {v}" for k, v in checks.solver.value_counts().items()), "",
             "## Cost per decision (mean seconds)", "", md(costs(df), index=True), "",
             "## Episodes", ""]
    for (mode, seq), g in df.groupby(["mode", "sequence"]):
        cols = [c for c in ["episode", "instance", "family", "n", "decision", "deployed", "checks", "training",
                            "new_outcome", "synthesis", "library_size", "seconds"] if c in g]
        lines += [f"### {mode} {seq}", "", md(g[cols].round(1)), ""]
    (study_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(study_dir / "REPORT.md")


if __name__ == "__main__":
    main()
