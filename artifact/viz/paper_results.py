"""Tables and figures for the evaluation section, built only from existing runs (no LLM or PRISM calls).

RQ1 (the paper's RQ1, certified coverage) reads the core runs and baselines of configs/experiments/rq1_comparison.yaml;
RQ2 (adaptation over sequences) reads out/results/sequential/<study>/; RQ3 (ingredients of generation; the
repository's "rq2") reads the core runs, their no-repair picks and the pure-restart runs of the same loop.

Usage (from artifact/): python viz/paper_results.py [--study shift_v0]
Writes out/paper/tables/<label>.tex (a tabular body) and .csv (long format), and out/paper/figures/<label>.pdf.
"""
import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from core.domain import load_domain  # noqa: E402
from core.verifier import SOUND  # noqa: E402
from experiments.base import experiment, held_out  # noqa: E402
from experiments.comparison import Comparison  # noqa: E402
from results_io import sample_records  # noqa: E402
from transfer import RULES, csv_path, parse_ids  # noqa: E402

RESULTS = ROOT / "out" / "results"
BASELINES_DIR = RESULTS / "baselines"
OUT = ROOT / "out" / "paper"
RQ1_SPEC = ROOT / "configs" / "experiments" / "rq1_comparison.yaml"

# The paper's configuration names -> the RQ1 harness's baseline names, per benchmark where they differ
BASELINES = {"DT-raw": "1-2-3-Go! (reimplemented: trees over raw variables)",
             "DT-voc": "1-2-3-Go! with our features",
             "RL": "PPO with our features",
             "Hand": {"grids": "Hand-written template", "families": "Hand-written template",
                      "pacman": "Hand-written maze rules"}}
APPROACHES = ["Full", "DT-raw", "DT-voc", "RL", "Hand"]
RQ1_BENCHES = {"grids": "Random grids", "families": "Grid-world families", "pacman": "Pac-Man",
               "mer": "MER across $x$", "mer_n": "MER across $n$"}
EXTRA_BENCHES = {"mer_one": "MER across $x$, trained on $n=1$", "mer_n_one": "MER across $n$, trained on $n=1$"}
SIZE = {"grids": ("n", "grid side length", 11), "families": ("n", "grid side length", 8),
        "pacman": ("max_steps", "game length (moves)", 18), "mer_n": ("n", "number of requests $n$", 3)}

# One colour, marker and line style per approach in every figure; markers and dashes keep them apart in grayscale
STYLE = {"Full": dict(color="#1f5fae", marker="o", ls="-"),
         "DT-raw": dict(color="#d55e00", marker="^", ls="--"),
         "DT-voc": dict(color="#8c5a1e", marker="s", ls="-."),
         "RL": dict(color="#009e73", marker="D", ls=":"),
         "Hand": dict(color="#6a3d9a", marker="x", ls=(0, (5, 1, 1, 1, 1, 1))),
         "Restart": dict(color="#7f7f7f", marker="v", ls="--"),
         "Extended": dict(color="#1f5fae", marker="o", ls="-"),
         "Frozen": dict(color="#7f7f7f", marker="s", ls="--")}

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif"})


# ---------------------------------------------------------------- shared

def write_table(label: str, tex: List[str], rows: List[dict]) -> None:
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "tables" / f"{label}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "tables" / f"{label}.csv", index=False)
    print(f"wrote tables/{label}.tex, .csv")


def save_figure(fig, label: str) -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / f"{label}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{label}.pdf")


def pct(mean: float, sd: float, n: int, bold: bool = False) -> str:
    """Percent, with the standard deviation when there is more than one value."""
    if mean is None or pd.isna(mean):
        return "--"
    body = f"{100 * mean:.1f}" + (f" \\pm {100 * sd:.1f}" if n > 1 and not pd.isna(sd) else "")
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def count(n) -> int:
    return 0 if n is None or pd.isna(n) else int(n)


def stats(values) -> dict:
    v = pd.Series(values, dtype=float).dropna()
    return {"mean": v.mean() if len(v) else np.nan, "sd": v.std(ddof=1) if len(v) > 1 else np.nan, "n": len(v)}


def runs_of(name: str, root: str = "core") -> List[Path]:
    """The finished runs of a condition (a run still in progress has no results parquet yet)."""
    return sorted(p for p in (RESULTS / root / name).glob("seed_*")
                  if (p / "config.json").exists() and (p / "SYMBOLIC_results.parquet").exists())


# ---------------------------------------------------------------- RQ1: per (controller, held-out target) outcomes

CHECKS = "exact"   # exact: src/recertify.py's verdicts (interval iteration; where it stalls, RECHECKS' verdicts)
                   # (transfer_exact_*.csv, `recertify.py combine`); ii / lp: one method each; gs: the
                   # loop solver's


def checked(path: Optional[Path], kind: Optional[str] = None) -> Optional[Path]:
    """The CSV of `path`'s checks of `kind` (default CHECKS): transfer_<x>.csv itself (gs), its
    transfer_exact_<x>.csv (exact), transfer_lp_<x>.csv (lp), transfer_ii_<x>.csv (ii) or, for the sealed test pass, its
    transfer_sealed_<x>.csv (sealed; all by
    src/recertify.py)."""
    kind = kind or CHECKS
    if path is None or kind == "gs":
        return path
    return path.with_name(path.name.replace("transfer_", f"transfer_{kind}_", 1))


# The sealed test targets per benchmark (configs/transfer/default.yaml `sealed`), read only by the test pass (P2)
SEALED = {"grids": {"targets": "0-41", "dataset": "grid_test_large.csv"},
          "families": {"targets": "36-59,96-119,156-179",
                       "groups": {"rooms": "36-59", "crossing": "96-119", "adjacent": "156-179"}},
          "pacman": {"targets": "10-11"}, "mer": {"targets": "9-11"}, "mer_n": {"targets": "9-11"}}


def sealed_bench(bench):
    """`bench` with its sealed test targets (and dataset) in place of the validation ones."""
    return replace(bench, **SEALED[bench.name])


def baseline_csv(config: str, rules: str = "final", dataset: Optional[str] = None) -> Path:
    """A baseline's transfer CSV (Comparison._csv), named for `dataset` when it checks another dataset's targets."""
    path = Comparison._csv(config, rules)
    if dataset is None:
        return path
    suffix = "_open" if rules == "open" else ""
    return path.with_name(f"transfer_{Path(dataset).stem}{suffix}.csv")


# Every verdict the tables use (src/certs.py, Oct 21): one rule, one file. A check is found by the check file it
# came from (`source_file`, the path src/recertify.py or src/sequential.py wrote or would write) and its split.
CERTS = OUT / "certs.csv"
_certs: Optional[pd.DataFrame] = None


def certs() -> pd.DataFrame:
    global _certs
    if _certs is None:
        _certs = pd.read_csv(CERTS, dtype={"list_id": str, "run": str})
    return _certs


def certified_checks(path: Optional[Path], split: Optional[str] = None) -> pd.DataFrame:
    """certs.csv's rows of the check file `path` (any split, or `split`), with the columns of a transfer CSV:
    source, target, outcome, certified, margin, binding, solver, states, seconds."""
    if path is None:
        return pd.DataFrame()
    rel = Path(path).resolve().relative_to(ROOT).as_posix()
    c = certs()
    df = c[c.source_file == rel]
    if split is not None:
        df = df[df.split == split]
    worst = pd.DataFrame([json.loads(w) if isinstance(w, str) else {} for w in df.worst], index=df.index)
    df = pd.concat([df, worst.add_prefix("worst_")], axis=1)
    return df.rename(columns={"list_id": "source", "instance": "target", "verdict": "outcome"}).assign(
        certified=lambda d: d.outcome == "certified")


def outcomes(path: Path, bench, approach: str, run: str = "", sources: Optional[List[str]] = None,
             source_groups: Optional[Dict[str, str]] = None, kind: Optional[str] = None) -> pd.DataFrame:
    """One row per (rule list or controller, held-out target) of a check file (`path` as src/transfer.py names it,
    under CHECKS), from certs.csv, on the list's own group's targets where the benchmark has groups (a source checked
    on every group, "*", counts once per group). Columns certified, inconclusive, timeout (not certifiable within
    limits) and error_other."""
    path = checked(path, kind)
    df = certified_checks(path)
    df = df[df.split != "training"].copy() if len(df) else df
    if df.empty:
        if path is not None:
            print(f"  missing from certs.csv: {path.relative_to(ROOT)}")
        return pd.DataFrame()
    df["source"] = df.source.astype(str)
    if sources is not None:
        df = df[df.source.isin(sources)]
    df = df[df.target.isin(held_out(bench))].copy()
    df["inconclusive"] = df.outcome == "inconclusive"
    df["timeout"] = df.outcome == "timeout"
    df["error_other"] = df.outcome == "error"
    parts = []
    for source, part in df.groupby("source"):
        if bench.groups:
            group = (source_groups or bench.source_groups).get(source)
            for g in (list(bench.groups) if group == "*" else [group]):
                parts.append(part[part.target.isin(parse_ids(bench.groups[g]))].assign(group=g))
        else:
            parts.append(part.assign(group=""))
    out = pd.concat(parts) if parts else df.assign(group="")
    out["list"] = run + "/" + out.source + "@" + out.group
    return out.assign(approach=approach, bench=bench.name, run=run)


def rq1_outcomes(exp, sealed: bool = False) -> Dict[str, pd.DataFrame]:
    """{"frozen": ..., "permissive": ...}: per-target outcomes of every RQ1 controller. Full: frozen = the list with
    `any` resolved by the pick's order (determinized; Pac-Man: the narrowed lists as kept, already resolved),
    permissive = the list as kept, `any` open (none for Pac-Man). Baselines: their CSV, and the ties-open CSV as the
    permissive controller where one exists (otherwise the controller has one action per state and both coincide).
    `sealed`: the same controllers on the sealed test targets (transfer_sealed_*.csv)."""
    spec = exp.spec
    kind = "sealed" if sealed else None
    frozen, permissive = [], []
    baselines = {b["name"]: b for b in spec["baselines"]}
    for name, bench in spec["benchmarks"].items():
        if name not in RQ1_BENCHES and (sealed or name not in EXTRA_BENCHES):
            continue
        if sealed and name in SEALED_LEFT_OUT:
            continue
        cfg = load_config(bench.base)
        dataset = bench.dataset or cfg.domain.dataset
        other = None
        if sealed:
            bench = sealed_bench(bench)
            other = SEALED[name].get("dataset")
            dataset = other or dataset
        for run in runs_of(bench.base):
            fz = "final" if name == "pacman" else "determinized"
            frozen.append(outcomes(csv_path(run, dataset, fz), bench, "Full", run.name, kind=kind))
            if name != "pacman":
                permissive.append(outcomes(csv_path(run, dataset, "final"), bench, "Full", run.name, kind=kind))
        for approach in APPROACHES[1:]:
            base = BASELINES[approach]
            base = base.get(name) if isinstance(base, dict) else base
            b = baselines.get(base)
            if b is None or name not in b["config"]:
                continue
            src = Comparison._sources(b["sources"][name])
            config = b["config"][name]
            frozen.append(outcomes(baseline_csv(config, dataset=other), bench, approach, "", src["sources"],
                                   src["source_groups"], kind=kind))
            opened = baseline_csv(config, "open", other)
            permissive.append(outcomes(opened if not certified_checks(checked(opened, kind)).empty else baseline_csv(config, dataset=other),
                                       bench, approach, "", src["sources"], src["source_groups"], kind=kind))
            if approach == "RL" and name in RL_SEEDS:     # P8: more PPO seeds per training set on the grid worlds
                for seed in RL_SEEDS[name]:
                    extra = config.replace(".yaml", f"_seed{seed}.yaml")
                    for part in (frozen, permissive):     # one action per state: both columns coincide
                        part.append(outcomes(baseline_csv(extra, dataset=other), bench, approach, f"seed{seed}",
                                             src["sources"], src["source_groups"], kind=kind))
    return {"frozen": pd.concat(frozen, ignore_index=True), "permissive": pd.concat(permissive, ignore_index=True)}


# P8: PPO on the grid worlds with these seeds besides seed 0 (configs/baselines/ppo_features_*_seed<k>.yaml)
RL_SEEDS = {"grids": [1, 2], "families": [1, 2]}


def llm_outcomes(exp, sealed: bool = False) -> Dict[str, pd.DataFrame]:
    """As rq1_outcomes, for Full only, one approach per LLM: Haiku 5.5's core runs and the other LLMs' runs of the
    same core conditions (OTHER_LLMS; only the model differs)."""
    kind = "sealed" if sealed else None
    frozen, permissive = [], []
    for name, bench in exp.spec["benchmarks"].items():
        if name not in RQ1_BENCHES:
            continue
        cfg = load_config(bench.base)
        dataset = bench.dataset or cfg.domain.dataset
        if sealed:
            bench = sealed_bench(bench)
            dataset = SEALED[name].get("dataset") or dataset
        arms = {"Claude Haiku 5.5": runs_of(bench.base)}
        arms.update({llm: runs_of(bench.base + suffix, "ablations") for llm, suffix in OTHER_LLMS.items()})
        for llm, runs in arms.items():
            for run in runs:
                fz = "final" if name == "pacman" else "determinized"
                frozen.append(outcomes(csv_path(run, dataset, fz), bench, llm, run.name, kind=kind))
                if name != "pacman":
                    permissive.append(outcomes(csv_path(run, dataset, "final"), bench, llm, run.name, kind=kind))
    return {"frozen": pd.concat(frozen, ignore_index=True), "permissive": pd.concat(permissive, ignore_index=True)}


def per_list(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["bench", "approach", "list"]).agg(
        coverage=("certified", "mean"), certified=("certified", "sum"), targets=("target", "count"),
        timeout=("timeout", "mean"), error=("error_other", "mean"),
        inconclusive=("inconclusive", "mean")).reset_index()


def coverage_table(data: Dict[str, pd.DataFrame], benches, approaches) -> pd.DataFrame:
    """Per (benchmark, approach): mean, sd and n over controllers of their coverage, permissive and frozen, and the
    shares of the frozen checks that were inconclusive, not certifiable within limits (timeout) or errors."""
    lists = {k: per_list(v) for k, v in data.items()}
    rows = []
    for bench in benches:
        for approach in approaches:
            entry = {"bench": bench, "approach": approach}
            for kind, df in lists.items():
                g = df[(df.bench == bench) & (df.approach == approach)]
                if g.empty:
                    continue
                s = stats(g.coverage)
                entry.update({f"{kind}_mean": s["mean"], f"{kind}_sd": s["sd"], f"{kind}_n": s["n"],
                              f"{kind}_targets_per_list": g.targets.mean(),
                              f"{kind}_certified_per_list": g.certified.mean()})
                if kind == "frozen":
                    entry.update({"timeout_share": g.timeout.mean(), "error_share": g.error.mean(),
                                  "inconclusive_share": g.inconclusive.mean()})
            if len(entry) > 2:
                entry.update({"split": "validation", "checks": CHECKS})
                rows.append(entry)
    return pd.DataFrame(rows)


def share(value) -> str:
    return "--" if value is None or pd.isna(value) else f"${100 * value:.1f}$"


# Benchmarks the sealed table leaves out, and why. Pac-Man was until Oct 21: its rule-list
# models at lengths 50 and 100 build (0.8M and 14M states); the side analysis of the bare game did not.
SEALED_LEFT_OUT: Dict[str, str] = {}


def rq1_sealed_table(data: Dict[str, pd.DataFrame]) -> None:
    """T1 on the sealed test targets (P2), in the validation table's layout, with the number of controllers (n) per
    cell: a cell is only what has been checked so far, and a benchmark or controller with no sealed checks is left
    out."""
    benches = [b for b in RQ1_BENCHES if b not in SEALED_LEFT_OUT]
    table = coverage_table(data, benches, APPROACHES).assign(split="sealed test", checks="sealed")
    long = table.melt(id_vars=["bench", "approach", "split", "checks"], var_name="metric").dropna(subset=["value"])
    tex = [r"\toprule", r"Controller & Permissive (\%) & Frozen (\%) & Inconclusive (\%) & Timeout (\%) & $n$ \\"]
    for bench, title in RQ1_BENCHES.items():
        b = table[table.bench == bench]
        if b.empty:
            continue
        best = {k: b[f"{k}_mean"].max() for k in ("permissive", "frozen")}
        tex += [r"\midrule", rf"\multicolumn{{6}}{{l}}{{\emph{{{title}}}}} \\"]
        for _, r in b.iterrows():
            cells = [pct(r.get(f"{k}_mean"), r.get(f"{k}_sd"), count(r.get(f"{k}_n")),
                         bold=not pd.isna(r.get(f"{k}_mean")) and np.isclose(r[f"{k}_mean"], best[k]))
                     for k in ("permissive", "frozen")]
            tex.append(f"{r.approach} & {cells[0]} & {cells[1]} & {share(r.get('inconclusive_share'))} & "
                       f"{share(r.timeout_share)} & {count(r.get('frozen_n'))} \\\\")
    tex.append(r"\bottomrule")
    write_table("tab_rq1-coverage-sealed", tex, long.to_dict("records"))


def rq1_tables(data: Dict[str, pd.DataFrame], llms: Dict[str, pd.DataFrame]) -> None:
    table = coverage_table(data, list(RQ1_BENCHES) + list(EXTRA_BENCHES), APPROACHES)
    table["llm"] = np.where(table.approach == "Full", "anthropic/claude-haiku-5.5", "")
    long = table.melt(id_vars=["bench", "approach", "split", "llm", "checks"], var_name="metric").dropna(subset=["value"])
    tex = [r"\toprule", r"Controller & Permissive (\%) & Frozen (\%) & Inconclusive (\%) & Timeout (\%) \\"]
    for bench, title in RQ1_BENCHES.items():
        b = table[table.bench == bench]
        best = {k: b[f"{k}_mean"].max() for k in ("permissive", "frozen")}
        tex += [r"\midrule", rf"\multicolumn{{5}}{{l}}{{\emph{{{title}}}}} \\"]
        for _, r in b.iterrows():
            cells = []
            for k in ("permissive", "frozen"):
                cells.append(pct(r.get(f"{k}_mean"), r.get(f"{k}_sd"), count(r.get(f"{k}_n")),
                                 bold=not pd.isna(r.get(f"{k}_mean")) and np.isclose(r[f"{k}_mean"], best[k])))
            tex.append(f"{r.approach} & {cells[0]} & {cells[1]} & {share(r.get('inconclusive_share'))} & "
                       f"{share(r.timeout_share)} \\\\")
    tex.append(r"\bottomrule")
    write_table("tab_rq1-coverage", tex, long.to_dict("records"))

    llm_table(llms, "tab_rq1-coverage-llm")


def llm_table(llms: Dict[str, pd.DataFrame], label: str, split: str = "validation") -> None:
    """T1 per LLM (Full only): Haiku 5.5's core runs and the other LLMs' runs of the same conditions."""
    names = ["Claude Haiku 5.5"] + list(OTHER_LLMS)
    per_llm = coverage_table(llms, list(RQ1_BENCHES), names).assign(split=split)
    tex = [r"\toprule", r"LLM & Permissive (\%) & Frozen (\%) & Inconclusive (\%) & Timeout (\%) \\"]
    for bench, title in RQ1_BENCHES.items():
        tex += [r"\midrule", rf"\multicolumn{{5}}{{l}}{{\emph{{{title}}}}} \\"]
        for _, r in per_llm[per_llm.bench == bench].iterrows():
            tex.append(f"{r.approach} & {pct(r.get('permissive_mean'), r.get('permissive_sd'), count(r.get('permissive_n')))}"
                       f" & {pct(r.frozen_mean, r.frozen_sd, count(r.frozen_n))} & {share(r.get('inconclusive_share'))}"
                       f" & {share(r.timeout_share)} \\\\")
    tex.append(r"\bottomrule")
    long = per_llm.rename(columns={"approach": "llm"}).melt(id_vars=["bench", "llm", "split", "checks"],
                                                             var_name="metric").dropna(subset=["value"])
    write_table(label, tex, long.to_dict("records"))


def _size_controllers(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The controllers the size figure draws: Full's permissive lists (Pac-Man: its frozen lists, which have no
    permissive form) and each baseline's permissive controller (= frozen where it has one action per state)."""
    return pd.concat([data["permissive"], data["frozen"][(data["frozen"].bench == "pacman")
                                                         & (data["frozen"].approach == "Full")]])


def rq1_size_figure(data: Dict[str, pd.DataFrame], sealed: Optional[Dict[str, pd.DataFrame]] = None) -> None:
    """Certified coverage per held-out size (`_size_controllers`). With `sealed`, the sealed test targets are drawn
    too, in the same colours with hollow markers and dotted lines, and the sizes beyond the largest validation size
    are shaded."""
    splits = {"validation": _size_controllers(data)}
    if sealed is not None:
        splits["sealed test"] = _size_controllers(sealed)
    rows = []
    fig, axes = plt.subplots(1, len(SIZE), figsize=(7.16, 2.1), sharey=True)
    for ax, (bench, (column, xlabel, largest)) in zip(axes, SIZE.items()):
        biggest_val = None
        for split, df in splits.items():
            b = df[df.bench == bench]
            if b.empty:
                continue
            sizes = _sizes(bench, column, sealed=split != "validation")
            b = b.assign(size=b.target.map(sizes))
            if split == "validation":
                biggest_val = b["size"].max()
            span = max(b["size"].max(), largest) - min(b["size"].min(), largest)
            for k, approach in enumerate(APPROACHES):
                a = b[b.approach == approach]
                if a.empty:
                    continue
                by_list = a.groupby(["size", "list"]).certified.mean().reset_index()
                curve = by_list.groupby("size").certified.agg(["mean", "std", "count"]).reset_index()
                st = STYLE[approach]
                dodge = (k - 2) * 0.012 * span          # small horizontal offsets so coinciding lines stay visible
                test = split != "validation"
                ax.plot(curve["size"] + dodge, curve["mean"], color=st["color"], marker=st["marker"],
                        ls=":" if test else st["ls"], lw=1.0 if test else 1.2, ms=4.5,
                        mfc="none" if (approach != "Full" or test) else st["color"],
                        alpha=0.75 if test else 1.0, label=None if test else approach)
                rows += [{"bench": bench, "approach": approach, "size": s, "coverage_mean": m, "coverage_sd": sd,
                          "n_lists": c, "targets_per_size": int(a[a["size"] == s].target.nunique()),
                          "controller": "frozen" if (bench == "pacman" and approach == "Full") else "permissive",
                          "split": split}
                         for s, m, sd, c in curve.itertuples(index=False)]
        if sealed is not None and biggest_val is not None:
            right = max((r["size"] for r in rows if r["bench"] == bench), default=biggest_val)
            if right > biggest_val:
                ax.axvspan(biggest_val, right, color="#e4e3df", alpha=0.5, lw=0, zorder=0)
        ax.axvline(largest, color="#444444", ls=(0, (4, 3)), lw=0.8)
        if sealed is not None and bench == "mer_n":       # test sizes reach n = 100
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel(xlabel)
        ax.set_title(RQ1_BENCHES[bench], fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("certified coverage (fraction)")
    handles, labels = axes[1].get_legend_handles_labels()
    if sealed is not None:
        handles.append(matplotlib.lines.Line2D([], [], color="#444444", ls=":", marker="o", mfc="none", lw=1.0))
        labels.append("test (sealed)")
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    save_figure(fig, "fig_rq1-size")
    pd.DataFrame(rows).to_csv(OUT / "figures" / "fig_rq1-size.csv", index=False)


# Sealed test instances per benchmark for the solvability table: (dataset, ids), and the multi-objective query's
# results where src/ceilings.py (or a subset run of its query) wrote them
FEASIBILITY = {"grids": ("grid_test_large.csv", "0-41", "gridworld_grid_test_large.csv"),
               "families": ("families_v0.csv", "36-59,96-119,156-179", "gridworld_families_v0_test.csv"),
               "pacman": ("pacman_transfer.csv", "10-11", None),
               "mer": ("mer_transfer.csv", "9-11", None), "mer_n": ("mer_n_transfer.csv", "9-11", None)}


def solvability() -> None:
    """Setup: is every sealed test instance solvable? Two sources. (1) A witness: a controller of any method certified
    on the instance under the certification rule meets every threshold at once, which proves joint feasibility.
    (2) PRISM's multi-objective query on the bare MDP (src/ceilings.py), where it was run; its "false" is
    contradicted by a witness on some instances, so it is reported but not relied on. `tab_setup-feasibility.csv`:
    one row per instance; a summary per benchmark is printed."""
    rows = []
    for bench, (dataset, ids, query_csv) in FEASIBILITY.items():
        stem = Path(dataset).stem
        wanted = set(parse_ids(ids))
        witnesses: Dict[int, set] = {i: set() for i in wanted}
        c = certs()
        ok = c[(c.split == "test") & (c.verdict == "certified") & c.source_file.str.contains(f"transfer_sealed_{stem}")
               & c.instance.isin(wanted)]
        for t, f, s in zip(ok.instance, ok.source_file, ok.list_id):
            witnesses[int(t)].add(f"{f}:{s}")
        query = {}
        if query_csv and (RESULTS / "ceilings" / query_csv).exists():
            q = pd.read_csv(RESULTS / "ceilings" / query_csv)
            key = "instance" if "instance" in q else q.columns[0]
            query = dict(zip(q[key].astype(int), q.jointly_feasible))
        for i in sorted(wanted):
            rows.append({"bench": bench, "dataset": dataset, "instance": i, "witnesses": len(witnesses[i]),
                         "witnessed": bool(witnesses[i]), "joint_query": query.get(i), "split": "sealed test"})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "tab_setup-feasibility.csv", index=False)
    for bench, g in df.groupby("bench", sort=False):
        q = g.joint_query.dropna().astype(str)
        print(f"  solvability {bench}: {int(g.witnessed.sum())}/{len(g)} witnessed by a certified controller; "
              f"joint query: {q.value_counts().to_dict() if len(q) else 'not run'}")
    print("wrote tables/tab_setup-feasibility.csv")


def rq1_by_family(data: Dict[str, pd.DataFrame], sealed: Dict[str, pd.DataFrame]) -> None:
    """Families: certified coverage of every controller per family, validation and sealed test (frozen controllers,
    each list on its own family). `tab_rq1-by-family.csv`."""
    rows = []
    for split, d in (("validation", data["frozen"]), ("sealed test", sealed["frozen"])):
        f = d[d.bench == "families"]
        for (approach, group), g in f.groupby(["approach", "group"]):
            lists = g.groupby("list").certified.mean()
            rows.append({"approach": approach, "family": group, "split": split, "coverage_mean": lists.mean(),
                         "coverage_sd": lists.std(ddof=1) if len(lists) > 1 else np.nan, "n_lists": len(lists),
                         "targets_per_list": g.groupby("list").size().mean()})
    pd.DataFrame(rows).to_csv(OUT / "tables" / "tab_rq1-by-family.csv", index=False)
    print("wrote tables/tab_rq1-by-family.csv")


def _sizes(bench: str, column: str, sealed: bool = False) -> Dict[int, float]:
    exp_bench = EXP.spec["benchmarks"][bench]
    cfg = load_config(exp_bench.base)
    dataset = (SEALED[bench].get("dataset") if sealed else None) or exp_bench.dataset or cfg.domain.dataset
    data = pd.read_csv(ROOT / "domains" / cfg.domain.name / "data" / dataset)
    return data[column].to_dict()


def rq1_misses(data: Dict[str, pd.DataFrame]) -> None:
    """T1b: for each held-out target a frozen Full list does not certify, the smallest margin over the requirements
    (negative: how far the worst case falls short of the threshold)."""
    df = data["frozen"]
    df = df[(df.approach == "Full") & ~df.certified]
    rows = []
    for bench, part in df.groupby("bench"):
        b = EXP.spec["benchmarks"][bench]
        cfg = load_config(b.base)
        domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
        instances = {int(i.id): i for i in domain.load_instances(b.dataset or cfg.domain.dataset)}
        reqs = {t: domain.spec(instances[t]).requirements for t in part.target.unique()}
        for r in part.itertuples(index=False):
            margins = {q.name: q.margin(getattr(r, f"worst_{q.name}")) for q in reqs[r.target]
                       if not pd.isna(getattr(r, f"worst_{q.name}", np.nan))}
            worst = min(margins, key=margins.get) if margins else None
            rows.append({"bench": bench, "list": r.list, "target": r.target, "smallest_margin": margins.get(worst),
                         "requirement": worst, "timeout": r.timeout, "outcome": getattr(r, "outcome", None),
                         "split": "validation", "checks": CHECKS})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "tables" / "tab_rq1-misses.csv", index=False)
    print("wrote tables/tab_rq1-misses.csv")


# ---------------------------------------------------------------- RQ2: the sequential study

# P6: the sequential study from the other core families runs' seed libraries (configs/sequential/shift_v0_lib*.yaml)
LIBRARIES = {"core/port_fam_history run 2": "shift_v0_lib2", "core/port_fam_history run 3": "shift_v0_lib3"}


def sequence_checks(study: str, mode: str, seq: str) -> pd.DataFrame:
    """One sequence's certification checks from certs.csv, as its checks.csv lays them out: verdicts by the rule
    (outcome, margin), the worst case per requirement (worst_<name>), and the manager's own run (seconds,
    model_seconds, solver: what the study's certification took and used as it ran)."""
    c = certs()
    df = c[(c.run == f"{study}/{mode}/{seq}") & c.controller.str.startswith("Library")].copy()
    worst = pd.DataFrame([json.loads(w) if isinstance(w, str) else {} for w in df.worst], index=df.index)
    df = df.rename(columns={"list_id": "rule_list", "verdict": "outcome"}).assign(
        seconds=df.old_seconds, solver=df.old_solver, episode=df.episode.astype(int))
    return pd.concat([df, worst.add_prefix("worst_")], axis=1)


def sequential(study: str) -> Dict[str, pd.DataFrame]:
    episodes, checks = [], []
    for mode, setting in (("growing", "Extended"), ("frozen", "Frozen")):
        for seq in sorted((RESULTS / "sequential" / study / mode).glob("s*")):
            e = pd.read_csv(seq / "episodes.csv").assign(setting=setting, sequence=seq.name)
            c = sequence_checks(study, mode, seq.name).assign(setting=setting, sequence=seq.name)
            c = c.merge(e[["episode", "n", "family", "decision"]], on="episode", how="left")
            lib = json.loads((seq / "library.json").read_text(encoding="utf-8")) if (seq / "library.json").exists() \
                else None
            episodes.append(e.assign(final_library=len(lib) if lib is not None else np.nan))
            checks.append(c)
    return {"episodes": pd.concat(episodes, ignore_index=True), "checks": pd.concat(checks, ignore_index=True)}


def rq2_tables(seq: Dict[str, pd.DataFrame], expected: int) -> None:
    ep, ck = seq["episodes"], seq["checks"]
    decisions = ["retain", "switch", "generate", "defer"]
    rows, tex = [], [r"\toprule",
                     r"Library & Retain & Switch & Generate & Defer & Final library size \\", r"\midrule",
                     r"\multicolumn{6}{l}{\emph{Grid worlds (families and random grids)}} \\"]
    for setting in ("Extended", "Frozen"):
        s = ep[ep.setting == setting]
        for name, g in s.groupby("sequence"):
            rows += [{"setting": setting, "sequence": name, "decision": d, "episodes": int((g.decision == d).sum())}
                     for d in decisions]
            rows.append({"setting": setting, "sequence": name, "decision": "generation_attempted",
                         "episodes": int(g.get("generation_seconds", pd.Series(dtype=float)).notna().sum())})
            rows.append({"setting": setting, "sequence": name, "decision": "episodes_done", "episodes": len(g)})
            rows.append({"setting": setting, "sequence": name, "decision": "final_library_size",
                         "episodes": int(g.library_size.iloc[-1])})
        counts = [int((s.decision == d).sum()) for d in decisions]
        sizes = ", ".join(str(int(g.library_size.iloc[-1])) for _, g in s.groupby("sequence"))
        tex.append(f"{setting} & " + " & ".join(map(str, counts)) + f" & {sizes} \\\\")
    for label, study in LIBRARIES.items():      # P6: the same sequences from the other seed libraries
        if not (RESULTS / "sequential" / study).exists():
            continue
        other = sequential(study)["episodes"]
        tex += [r"\midrule", rf"\multicolumn{{6}}{{l}}{{\emph{{Seed library: {label}}}}} \\"]
        for setting in ("Extended", "Frozen"):
            s = other[other.setting == setting]
            if s.empty:
                continue
            for name, g in s.groupby("sequence"):
                rows += [{"setting": setting, "library": label, "sequence": name, "decision": d,
                          "episodes": int((g.decision == d).sum())} for d in decisions]
                rows.append({"setting": setting, "library": label, "sequence": name, "decision": "final_library_size",
                             "episodes": int(g.library_size.iloc[-1])})
            counts = [int((s.decision == d).sum()) for d in decisions]
            sizes = ", ".join(str(int(g.library_size.iloc[-1])) for _, g in s.groupby("sequence"))
            tex.append(f"{setting} & " + " & ".join(map(str, counts)) + f" & {sizes} \\\\")
    tex.append(r"\bottomrule")
    write_table("tab_rq2-decisions", tex, rows)

    # Costs per decision type
    def med(values) -> str:
        v = pd.Series(values, dtype=float).dropna()
        if v.empty:
            return "--"
        return f"{v.median():.1f} ({v.min():.1f}--{v.max():.1f})"

    cost_rows = []
    tex = [r"\toprule", r"Decision & Model (s) & Certify one (s) & Library search (s) & Generation (s) & "
                        r"LLM calls & Tokens (k) \\", r"\midrule"]
    for d in decisions:
        e, c = ep[ep.decision == d], ck[ck.decision == d]
        gen = e[e.get("generation_seconds", pd.Series(dtype=float)).notna()] if "generation_seconds" in e else e.iloc[0:0]
        search = e.certify_seconds if d != "retain" else pd.Series(dtype=float)
        tokens = (gen.prompt_tokens + gen.output_tokens) / 1000 if not gen.empty else pd.Series(dtype=float)
        metrics = {"model_construction_s": c.model_seconds, "certify_one_s": c.seconds,
                   "library_search_s": search, "generation_s": gen.get("generation_seconds", pd.Series(dtype=float)),
                   "llm_calls": gen.get("llm_calls", pd.Series(dtype=float)), "tokens_k": tokens,
                   "episode_s": e.seconds}
        for m, v in metrics.items():
            v = pd.Series(v, dtype=float).dropna()
            if len(v):
                cost_rows.append({"decision": d, "metric": m, "median": v.median(), "min": v.min(), "max": v.max(),
                                  "n": len(v), "split": "validation"})
        calls = metrics["llm_calls"]
        tex.append(f"{d.capitalize()} ({len(e)}) & {med(c.model_seconds)} & {med(c.seconds)} & {med(search)} & "
                   f"{med(metrics['generation_s'])} & {med(calls) if len(pd.Series(calls).dropna()) else '--'} & "
                   f"{med(tokens)} \\\\")
    tex.append(r"\bottomrule")
    write_table("tab_rq2-cost", tex, cost_rows)

    by_size = ck.groupby("n").agg(checks=("seconds", "size"), median_s=("seconds", "median"),
                                  min_s=("seconds", "min"), max_s=("seconds", "max"),
                                  median_model_s=("model_seconds", "median"),
                                  median_states=("states", "median"),
                                  timeouts=("outcome", lambda o: int((o == "timeout").sum())),
                                  inconclusive=("outcome", lambda o: int((o == "inconclusive").sum())),
                                  gauss_seidel_fallback=("solver", lambda s: int((~s.fillna("").isin(SOUND.values()))
                                                                                 .sum()))).reset_index()
    by_size.assign(split="validation").to_csv(OUT / "tables" / "rq2-cert-time-by-size.csv", index=False)
    print("wrote tables/rq2-cert-time-by-size.csv")

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    curve_rows = []
    for setting in ("Extended", "Frozen"):
        s = ep[ep.setting == setting]
        curves = {n: np.cumsum((g.sort_values("episode").decision == "generate").to_numpy())
                  for n, g in s.groupby("sequence")}
        length = max(len(c) for c in curves.values())
        mat = np.full((len(curves), length), np.nan)
        for i, c in enumerate(curves.values()):
            mat[i, :len(c)] = c
        x = np.arange(1, length + 1)
        total = np.nansum(mat, axis=0)
        st = STYLE[setting]
        ax.plot(x, total, color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.2, ms=4, drawstyle="steps-post",
                label=f"{setting.lower()} library")
        for name, c in curves.items():
            curve_rows += [{"setting": setting, "sequence": name, "position": i + 1, "cumulative_generations": v}
                           for i, v in enumerate(c)]
    ax.set_xlabel("position in the sequence (episode)")
    ax.set_ylabel(f"cumulative generations\n(sum over {ep.sequence.nunique()} sequences)")
    ax.set_xticks(range(1, expected + 1, 1))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "fig_rq2-extension")
    pd.DataFrame(curve_rows).to_csv(OUT / "figures" / "fig_rq2-extension.csv", index=False)


def rq2_episodes(study: str, example: Tuple[str, int, int] = ("s1", 2, 3)) -> None:
    """P3: one row per (setting, sequence, episode): the map, the decision, the list that served it and where that
    list came from (the seed library, generated in this episode, or generated earlier in the same sequence). Also the
    running example (`example`: sequence, the episode that generated, the next one): every library list's worst case
    and margin per requirement on the instance met, the generated list's on the next instance, and its size."""
    rows = []
    for mode, setting in (("growing", "Extended"), ("frozen", "Frozen")):
        for seq in sorted((RESULTS / "sequential" / study / mode).glob("s*")):
            e = pd.read_csv(seq / "episodes.csv")
            for r in e.itertuples(index=False):
                deployed = r.deployed if isinstance(r.deployed, str) else ""
                made = int(deployed[len("gen_t"):]) if deployed.startswith("gen_t") else None
                origin = ("seed library" if deployed.startswith("seed") else "" if not deployed else
                          "generated in this episode" if made == r.episode else "generated earlier in this sequence")
                rows.append({"setting": setting, "sequence": seq.name, "position": r.episode + 1,
                             "instance": r.instance, "map": r.family, "size": r.n, "decision": r.decision,
                             "served_by": deployed, "origin": origin,
                             "generated_at_position": made + 1 if made is not None else np.nan,
                             "reused_generated": origin == "generated earlier in this sequence",
                             "split": "validation"})
    pd.DataFrame(rows).to_csv(OUT / "tables" / "tab_rq2-episodes.csv", index=False)
    print("wrote tables/tab_rq2-episodes.csv")

    name, ep_gen, ep_next = example
    seq = RESULTS / "sequential" / study / "growing" / name
    checks = sequence_checks(study, "growing", name)
    library = {e["name"]: e for e in json.loads((seq / "library.json").read_text(encoding="utf-8"))}
    study_cfg = yaml.safe_load((ROOT / "configs" / "sequential" / f"{study}.yaml").read_text(encoding="utf-8"))
    domain = load_domain("gridworld", load_config(study_cfg["condition"]).domain.visible_extra)
    instances = {int(i.id): i for i in domain.load_instances(study_cfg["dataset"])}
    out = []
    for ep in (ep_gen, ep_next):
        for c in checks[checks.episode == ep].itertuples(index=False):
            entry = library[c.rule_list]
            size = list_size(entry["rules"])
            for q in domain.spec(instances[c.instance]).requirements:
                worst = getattr(c, f"worst_{q.name}")
                out.append({"sequence": name, "position": ep + 1, "instance": c.instance, "rule_list": c.rule_list,
                            "trained_on": " ".join(map(str, entry["training"])), "outcome": c.outcome,
                            "solver": c.solver, "requirement": q.name, "worst": worst, "threshold": q.threshold,
                            "margin": q.margin(worst), "smallest_margin": c.margin, **size, "split": "validation"})
    pd.DataFrame(out).to_csv(OUT / "tables" / "tab_rq2-example.csv", index=False)
    print("wrote tables/tab_rq2-example.csv")


# ---------------------------------------------------------------- sizes and costs

def _split_top(cond: str, sep: str) -> List[str]:
    """`cond` split at `sep` outside parentheses."""
    parts, depth, cur = [], 0, ""
    for ch in cond:
        depth += (ch == "(") - (ch == ")")
        if ch == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    return [p.strip() for p in parts + [cur] if p.strip()]


IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def list_size(rules) -> dict:
    """A rule list's size: rules, conjuncts per condition (top-level `&`; `true` has none), distinct names read."""
    conds = [str(c).strip() for c, _ in rules]
    conj = [0 if c in ("true", "") else len(_split_top(c, "&")) for c in conds]
    names = {n for c in conds for n in IDENT.findall(c) if n not in ("true", "false")}
    return {"rules": len(rules), "conjuncts_mean": float(np.mean(conj)) if conj else 0.0,
            "conjuncts_max": int(max(conj)) if conj else 0, "conjuncts_total": int(sum(conj)),
            "features": len(names)}


def controller_sizes() -> None:
    """P7: per controller, its size. Rule lists (Full, Restart, NoRepair, and the other LLMs' Full): rules,
    conjuncts per condition, distinct features. Trees: nodes and depth; PPO: parameters (src/recertify.py's
    sizes.json, written when it refits a baseline). Per benchmark mean and sd over controllers."""
    rows = []
    for bench in RQ1_BENCHES:
        b = EXP.spec["benchmarks"][bench]
        frozen = "final" if bench == "pacman" else "determinized"
        arms = {"Full": [(r, frozen) for r in runs_of(b.base)],
                "NoRepair": [(r, "norepair_determinized") for r in runs_of(b.base)],
                "Restart": [(r, frozen) for r in runs_of(RESTARTS.get(bench, ""), "ablations")] if bench in RESTARTS
                else []}
        for llm, suffix in OTHER_LLMS.items():
            arms[f"Full ({llm})"] = [(r, frozen) for r in runs_of(b.base + suffix, "ablations")]
        for arm, runs in arms.items():
            for run, rules in runs:
                for sid, lst in RULES[rules][0](run).items():
                    rows.append({"bench": bench, "controller": arm, "source": f"{run.name}/{sid}",
                                 "kind": "rule list", **list_size(lst)})
        for approach in APPROACHES[1:]:
            base = BASELINES[approach]
            base = base.get(bench) if isinstance(base, dict) else base
            spec = next((x for x in EXP.spec["baselines"] if x["name"] == base), None)
            if spec is None or bench not in spec["config"]:
                continue
            path = ROOT / spec["config"][bench]
            sources = Comparison._sources(spec["sources"][bench])["sources"]
            if approach == "Hand" and bench == "pacman":
                lists = yaml.safe_load(path.read_text(encoding="utf-8"))["params"]["lists"]
                rows += [{"bench": bench, "controller": approach, "source": s, "kind": "rule list",
                          **list_size(lists[s])} for s in sources]
                continue
            if approach == "Hand":
                rows.append({"bench": bench, "controller": approach, "source": "cautious",
                             "kind": "Python template (compiled per cell)"})
                continue
            sizes = BASELINES_DIR / path.stem / "sizes.json"
            data = json.loads(sizes.read_text(encoding="utf-8")) if sizes.exists() else {}
            for s in sources:
                kind = "network" if approach == "RL" else "tree"
                rows.append({"bench": bench, "controller": approach, "source": s, "kind": kind,
                             **(data.get(s) or {"missing": f"no {sizes.relative_to(ROOT)}"})})
    df = pd.DataFrame(rows)
    metrics = [c for c in ("rules", "conjuncts_mean", "conjuncts_max", "features", "nodes", "depth", "leaves",
                           "parameters") if c in df]
    summary = []
    for (bench, controller), g in df.groupby(["bench", "controller"], sort=False):
        entry = {"bench": bench, "controller": controller, "kind": g.kind.iloc[0], "n": len(g)}
        for m in metrics:
            if m in g and g[m].notna().any():
                entry[f"{m}_mean"], entry[f"{m}_sd"] = g[m].mean(), g[m].std(ddof=1) if g[m].notna().sum() > 1 else np.nan
        if "missing" in g and g.missing.notna().any():
            entry["missing"] = g.missing.dropna().iloc[0]
        summary.append(entry)
    pd.DataFrame(summary).to_csv(OUT / "tables" / "tab_rq1-sizes.csv", index=False)
    df.to_csv(OUT / "tables" / "tab_rq1-sizes-per-controller.csv", index=False)
    print("wrote tables/tab_rq1-sizes.csv, tab_rq1-sizes-per-controller.csv")


# $ per M prompt and output tokens, as OpenRouter lists them on 2026-10-19 (provider pinned)
PRICES = {"anthropic/claude-haiku-5.5": (0.10, 0.50), "anthropic/claude-sonnet-5.5": (2.00, 10.00),
          "openai/gpt-6-luna": (0.10, 0.50), "google/gemini-3.8-flash": (0.75, 3.75)}
# Condition suffixes of the other LLMs' core runs (run on a cluster; Full only). Sonnet 5.5's runs
# (ablations/port_*_sonnet) are incomplete and stay out.
OTHER_LLMS: Dict[str, str] = {"GPT-6 Luna": "_luna", "Gemini 3.8 Flash": "_gemini"}


def generation_cost(study: str) -> None:
    """P4: per condition, per training set: wall-clock minutes (the set's `total_time`, every chain and the pick),
    the LLM and PRISM seconds summed over every round of every chain (chains run concurrently, so these sums overlap
    each other and can exceed the wall clock), LLM calls, tokens and dollars. Median and range over training sets."""
    conditions = {f"Full ({b})": (EXP.spec["benchmarks"][b].base, "core") for b in RQ1_BENCHES}
    for bench in RQ1_BENCHES:
        base = EXP.spec["benchmarks"][bench].base
        for name, root in ((base + "_resample", "ablations"), (RESTARTS.get(bench), "ablations")):
            if name and runs_of(name, root):
                conditions[f"Restart ({bench})"] = (name, root)
        for llm, suffix in OTHER_LLMS.items():
            if runs_of(base + suffix, "ablations"):
                conditions[f"Full {llm} ({bench})"] = (base + suffix, "ablations")
    rows, per_set = [], []
    for label, (name, root) in conditions.items():
        for run in runs_of(name, root):
            model = json.loads((run / "config.json").read_text(encoding="utf-8"))["llm"]["model"]
            price = PRICES.get(model, (np.nan, np.nan))
            for sid, record in sample_records(run).items():
                its = record.get("iterations", [])
                prompt = sum(it.get("llm_prompt_tokens") or 0 for it in its)
                output = sum(it.get("llm_output_tokens") or 0 for it in its)
                per_set.append({"condition": label, "run_condition": name, "run": run.name, "set": sid, "model": model,
                                "wall_min": (record.get("total_time") or np.nan) / 60,
                                "llm_s": sum(it.get("llm_time") or 0 for it in its),
                                "prism_s": sum(it.get("prism_time") or 0 for it in its),
                                "llm_calls": sum(it.get("llm_calls") or 0 for it in its),
                                "tokens": prompt + output, "prompt_tokens": prompt, "output_tokens": output,
                                "dollars": (prompt * price[0] + output * price[1]) / 1e6})
    df = pd.DataFrame(per_set)
    for (label, model), g in df.groupby(["condition", "model"], sort=False):
        for m in ("wall_min", "llm_s", "prism_s", "llm_calls", "tokens", "dollars"):
            rows.append({"condition": label, "model": model, "metric": m, "median": g[m].median(), "min": g[m].min(),
                         "max": g[m].max(), "n_sets": len(g), "runs": g.run.nunique(),
                         "price_per_M": "/".join(f"{p:.2f}" for p in PRICES.get(model, (np.nan, np.nan)))})
    seq = sequential(study)["episodes"]
    gen = seq[seq.get("generation_seconds", pd.Series(dtype=float)).notna()]
    price = PRICES["anthropic/claude-haiku-5.5"]
    dollars = (gen.prompt_tokens * price[0] + gen.output_tokens * price[1]) / 1e6
    rows += [{"condition": f"Sequential study {study} (growing, all generations)", "model": "anthropic/claude-haiku-5.5",
              "metric": "dollars_total", "median": dollars.sum(), "min": np.nan, "max": np.nan, "n_sets": len(gen)},
             {"condition": f"Sequential study {study} (per generation)", "model": "anthropic/claude-haiku-5.5",
              "metric": "dollars", "median": dollars.median(), "min": dollars.min(), "max": dollars.max(),
              "n_sets": len(gen)},
             {"condition": f"Sequential study {study} (per generation)", "model": "anthropic/claude-haiku-5.5",
              "metric": "wall_min", "median": gen.generation_seconds.median() / 60,
              "min": gen.generation_seconds.min() / 60, "max": gen.generation_seconds.max() / 60, "n_sets": len(gen)},
             {"condition": f"Sequential study {study} (per generation)", "model": "anthropic/claude-haiku-5.5",
              "metric": "llm_calls", "median": gen.llm_calls.median(), "min": gen.llm_calls.min(),
              "max": gen.llm_calls.max(), "n_sets": len(gen)}]
    pd.DataFrame(rows).assign(split="validation").to_csv(OUT / "tables" / "tab_rq2-generation-cost.csv", index=False)
    df.to_csv(OUT / "tables" / "tab_rq2-generation-cost-per-set.csv", index=False)
    print("wrote tables/tab_rq2-generation-cost.csv, tab_rq2-generation-cost-per-set.csv")


# ---------------------------------------------------------------- RQ3: ingredients of generation

RQ3_BENCHES = {"grids": "Random grids", "families": "Grid-world families", "pacman": "Pac-Man",
               "mer": "MER across $x$", "mer_n": "MER across $n$"}
RESTARTS = {"mer": "port_mer_resample", "families": "port_fam_history_resample", "grids": "port_ts_history_resample",
            "pacman": "port_pacman_narrow_core_resample"}   # pure restarts of the core loop (planner.retry: always)


def training_passed(path: Path) -> List[float]:
    """Per list of a check file: the share of its own training members it certifies (certs.csv, training split,
    the same controller as the file: frozen lists for the frozen columns)."""
    df = certified_checks(path, "training")
    return [] if df.empty else df.groupby("source").certified.mean().tolist()


def calls_per_set(run: Path) -> float:
    records = sample_records(run)
    return sum(it.get("llm_calls") or 0 for r in records.values() for it in r.get("iterations", [])) / len(records)


def oneshot_csv(run: Path, dataset: str) -> Path:
    """src/oneshot.py's checks of a run (named as a transfer CSV, so `checked` maps it to the current rule's file)."""
    return run / f"transfer_{Path(dataset).stem}_oneshot.csv"


def oneshot_calls(run: Path) -> float:
    """LLM calls per training set of OneShot's answer: its round's calls (1, plus any re-asks)."""
    data = json.loads((run / "oneshot.json").read_text(encoding="utf-8"))
    return float(np.mean([e.get("llm_calls") or 1 for e in data.values()]))


def rq3_sealed_table() -> None:
    """T4 on the sealed test targets (P2): certified coverage per list of Full, Restart, NoRepair and OneShot, with the number
    of lists (a configuration whose runs are not all checked is left out). Training pass rates and LLM calls are the
    validation table's (they do not depend on the targets)."""
    rows, tex = [], [r"\toprule", r"Configuration & Certified coverage (\%) & Inconclusive (\%) & $n$ \\"]
    for bench, title in RQ3_BENCHES.items():
        if bench in SEALED_LEFT_OUT:
            continue
        b = EXP.spec["benchmarks"][bench]
        cfg = load_config(b.base)
        dataset = SEALED[bench].get("dataset") or b.dataset or cfg.domain.dataset
        sb = sealed_bench(b)
        frozen = "final" if bench == "pacman" else "determinized"     # Pac-Man: the narrowed lists, as in T1
        configs = {"Full": (runs_of(b.base), frozen),
                   "Restart": (runs_of(RESTARTS[bench], "ablations") if bench in RESTARTS else [], frozen),
                   "NoRepair": (runs_of(b.base), "norepair_determinized"),
                   "OneShot": (runs_of(b.base), "oneshot")}
        block = []
        for name, (runs, rules) in configs.items():
            path_of = (lambda r: oneshot_csv(r, dataset)) if rules == "oneshot" else                 (lambda r, rules=rules: csv_path(r, dataset, rules))
            if not runs or not all(not certified_checks(checked(path_of(r), "sealed")).empty for r in runs):
                continue
            cov = per_list(pd.concat([outcomes(path_of(r), sb, name, r.name, kind="sealed") for r in runs]))
            c = stats(cov.coverage)
            rows += [{"bench": bench, "configuration": name, "metric": m, "value": v, "split": "sealed test",
                      "runs": len(runs)}
                     for m, v in (("coverage_mean", c["mean"]), ("coverage_sd", c["sd"]), ("coverage_n", c["n"]),
                                  ("certified_per_list", cov.certified.mean()), ("targets_per_list", cov.targets.mean()),
                                  ("inconclusive_share", cov.inconclusive.mean()), ("timeout_share", cov.timeout.mean()))]
            block.append(f"{name} & {pct(c['mean'], c['sd'], c['n'])} & {share(cov.inconclusive.mean())} & "
                         f"{c['n']} \\\\")
        if block:
            tex += [r"\midrule", rf"\multicolumn{{4}}{{l}}{{\emph{{{title}}}}} \\"] + block
    tex.append(r"\bottomrule")
    write_table("tab_rq3-ablation-sealed", tex, rows)


def pacman_narrowing() -> None:
    """Pac-Man's core lists with and without the narrowing the loop added (`& !(xP = 3 & yP = 4)`, removed by the
    night-of-Oct-21 script; <run>/unnarrowed.json): certified held-out targets per list, validation and sealed test.
    `tab_pacman-narrowing.csv`."""
    b = EXP.spec["benchmarks"]["pacman"]
    dataset = b.dataset or load_config(b.base).domain.dataset
    stem = Path(dataset).stem
    rows = []
    for run in runs_of(b.base):
        for split, bench, kind in (("validation", b, None), ("sealed test", sealed_bench(b), "sealed")):
            for lists, name in (("narrowed", f"transfer_{stem}.csv"), ("unnarrowed", f"transfer_{stem}_unnarrowed.csv")):
                cov = per_list(outcomes(run / name, bench, lists, run.name, kind=kind))
                rows += [{"run": run.name, "split": split, "lists": lists, "list": r.list, "certified": r.certified,
                          "targets": r.targets} for r in cov.itertuples(index=False)]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tables" / "tab_pacman-narrowing.csv", index=False)
    print(df.pivot_table(index=["split", "run"], columns="lists", values="certified", aggfunc="sum").to_string())
    print("wrote tables/tab_pacman-narrowing.csv")


def rq3_tables() -> None:
    rows, tex = [], [r"\toprule", r"Configuration & Certified coverage (\%) & Training pass rate (\%) & LLM calls per set \\"]
    for bench, title in RQ3_BENCHES.items():
        b = EXP.spec["benchmarks"][bench]
        cfg = load_config(b.base)
        dataset = b.dataset or cfg.domain.dataset
        frozen = "final" if bench == "pacman" else "determinized"     # Pac-Man: the narrowed lists, as in T1
        configs = {"Full": (runs_of(b.base), frozen, False),
                   "Restart": (runs_of(RESTARTS[bench], "ablations") if bench in RESTARTS else [], frozen, False),
                   "NoRepair": (runs_of(b.base), "norepair_determinized", True),
                   "OneShot": (runs_of(b.base), "oneshot", False)}
        block = []
        for name, (runs, rules, norepair) in configs.items():
            path_of = (lambda r: oneshot_csv(r, dataset)) if rules == "oneshot" else                 (lambda r, rules=rules: csv_path(r, dataset, rules))
            if not runs or not all(not certified_checks(checked(path_of(r))).empty for r in runs):
                continue
            cov = pd.concat([outcomes(path_of(r), b, name, r.name) for r in runs])
            cov = per_list(cov)
            # OneShot's list was never checked on the training members as the loop ends: no pass rate
            passed = [x for r in runs for x in training_passed(checked(path_of(r)))]
            c, p = stats(cov.coverage), stats(passed)
            calls = [oneshot_calls(r) if rules == "oneshot" else calls_per_set(r) for r in runs]
            rows += [{"bench": bench, "configuration": name, "metric": m, "value": v, "split": "validation",
                      "runs": len(runs)}
                     for m, v in (("coverage_mean", c["mean"]), ("coverage_sd", c["sd"]), ("coverage_n", c["n"]),
                                  ("certified_per_list", cov.certified.mean()),
                                  ("targets_per_list", cov.targets.mean()),
                                  ("training_pass_mean", p["mean"]), ("training_pass_sd", p["sd"]),
                                  ("training_pass_n", p["n"]), ("llm_calls_per_set", np.mean(calls)),
                                  ("inconclusive_share", cov.inconclusive.mean()), ("timeout_share", cov.timeout.mean()))]
            block.append(f"{name} & {pct(c['mean'], c['sd'], c['n'])} & {pct(p['mean'], p['sd'], p['n'])} & "
                         f"${np.mean(calls):.1f}$ \\\\")
        if bench == "grids":           # the hand-written template does not use `closer`: without it, it is unchanged
            hand = per_list(outcomes(Comparison._csv("configs/baselines/template_grid_large.yaml"), b, "Hand", "",
                                     ["cautious"], {}))
            c = stats(hand.coverage)
            rows += [{"bench": bench, "configuration": "NoSP (Hand)", "metric": m, "value": v, "split": "validation",
                      "runs": 1} for m, v in (("coverage_mean", c["mean"]), ("coverage_n", c["n"]),
                                              ("certified_per_list", hand.certified.mean()))]
            block.append(f"NoSP (Hand) & {pct(c['mean'], c['sd'], c['n'])} & -- & -- \\\\")
        tex += [r"\midrule", rf"\multicolumn{{4}}{{l}}{{\emph{{{title}}}}} \\"] + block
    tex.append(r"\bottomrule")
    write_table("tab_rq3-ablation", tex, rows)

    # T4b: selection evidence, core loop (frozen lists)
    sel = []
    for bench in RQ1_BENCHES:
        b = EXP.spec["benchmarks"][bench]
        cfg = load_config(b.base)
        dataset = b.dataset or cfg.domain.dataset
        for run in runs_of(b.base):
            fz = "final" if bench == "pacman" else "determinized"
            lists = per_list(outcomes(csv_path(run, dataset, fz), b, "Full", run.name))
            for sid, record in sample_records(run).items():
                passed = all(bool(m.get("success")) for m in record.get("members", []))
                mine = lists[lists.list.str.startswith(f"{run.name}/{sid}@")]
                sel += [{"bench": bench, "run": run.name, "list": r.list, "passed_all_training": passed,
                         "certified": r.certified, "targets": r.targets} for r in mine.itertuples(index=False)]
    sel = pd.DataFrame(sel)
    summary = sel.groupby(["bench", "passed_all_training"]).agg(
        certified_mean=("certified", "mean"), certified_sd=("certified", "std"), n=("certified", "size"),
        targets_per_list=("targets", "mean")).reset_index().assign(split="validation", loop="core")
    summary.to_csv(OUT / "tables" / "tab_rq3-selection.csv", index=False)
    print("wrote tables/tab_rq3-selection.csv")

    # T4c: without `closer` on random grids, Full (port_ts_noclose, the core loop before the gridworld feedback fix)
    # against the hand-written template
    b = EXP.spec["benchmarks"]["grids_noclose"]
    # the loop solver's check files (never re-certified by src/recertify.py), certified under the rule by src/certs.py
    full = per_list(pd.concat([outcomes(csv_path(r, "grid_20_large.csv", "determinized"), b, "Full", r.name,
                                        kind="gs") for r in runs_of("port_ts_noclose")]))
    hand = per_list(outcomes(Comparison._csv("configs/baselines/template_grid_large.yaml"), b, "Hand", "",
                             ["cautious"], {}, kind="gs"))
    pd.DataFrame([{"benchmark": "grids_noclose (grid_20_large, targets 12-19)", "configuration": n,
                   "certified_per_list": g.certified.mean(), "certified_sd": g.certified.std(ddof=1),
                   "targets_per_list": g.targets.mean(), "n_lists": len(g),
                   "runs": g.list.str.split("/").str[0].nunique(), "loop": loop, "split": "validation",
                   "checks": "certs.csv"}
                  for n, g, loop in (("Full (NoSP)", full, "port_ts_noclose: core loop before the Oct 17 gridworld "
                                                          "feedback fix, no strategy hint"),
                                     ("Hand (NoSP)", hand, "not generated"))]) \
        .to_csv(OUT / "tables" / "tab_rq3-nosp.csv", index=False)
    print("wrote tables/tab_rq3-nosp.csv")


# Budget-curve panels: benchmark -> (replay file per run, what checked it). MER's replay predates the fixed rule.
# Budget-curve panels: name -> (benchmark, replay file per run, what checked it, sealed test targets). MER's replay
# predates the fixed rule. A panel is drawn only where its files exist.
BUDGET_PANELS = {"mer": ("mer", "budget_transfer_chains.csv", "loop solver", False),
                 "families": ("families", "budget_ii_rounds.csv", "fixed rule", False),
                 "families_test": ("families", "budget_ii_sealed_rounds.csv", "fixed rule", True)}


def _budget_arm(panel: str, runs: List[Path], name: str) -> Optional[pd.DataFrame]:
    """Per (run, training set, round budget R): LLM calls spent and held-out targets certified, from each run's replay
    file (BUDGET_PANELS); None unless every run has one."""
    bench, file, _, sealed = BUDGET_PANELS[panel]
    b = EXP.spec["benchmarks"][bench]
    b = sealed_bench(b) if sealed else b
    if not runs or not all((r / file).exists() for r in runs):
        return None
    per_run = []
    for r in runs:
        df = pd.read_csv(r / file)
        df = df[df.target.isin(held_out(b)) & ~df.trained.astype(bool)]
        df["certified"] = df.certified.fillna(False).astype(bool)
        per_run.append(df.groupby(["source", "round"]).agg(
            calls=("llm_calls", "first"), certified=("certified", "sum"), targets=("target", "size"),
            solved=("solved", "first"), picked_from=("picked_from", "first")).reset_index().assign(run=r.name, arm=name))
    return pd.concat(per_run, ignore_index=True)


def rq3_budget_figure() -> None:
    """F3: certified coverage of the held-out targets against LLM calls spent per training set, one panel per
    benchmark in BUDGET_PANELS: Full and, where its replays exist, Restart (per round budget R, the run's pick with
    repair over every chain cut to its first R rounds, `any` resolved; src/budget_transfer.py chain mode or
    src/budget_ii.py). One point per R: the runs' mean calls and mean coverage, the band their range. Where Restart is
    drawn, the dotted line is its mean total spend and the star Full's value there (each run's last R within that many
    calls). The diamond is OneShot: the first answer of chain 0, frozen, no selection or repair (1 call)."""

    def at(t: pd.DataFrame, budget: float) -> pd.Series:
        """Per run: the coverage of its last round within `budget` calls (mean over its training sets)."""
        out = {}
        for run, g in t.groupby("run"):
            within = g[g.calls <= budget + 1e-9].sort_values("round").groupby("source").tail(1)
            out[run] = (within.certified / within.targets).mean() if len(within) == g.source.nunique() else np.nan
        return pd.Series(out)

    panels = {}
    for panel, (bench, _, _, _) in BUDGET_PANELS.items():
        b = EXP.spec["benchmarks"][bench]
        arms = {"Full": _budget_arm(panel, runs_of(b.base), "Full"),
                "Restart": _budget_arm(panel, runs_of(RESTARTS[bench], "ablations"), "Restart")
                if bench in RESTARTS else None}
        arms = {k: v for k, v in arms.items() if v is not None}
        if "Full" in arms:
            panels[panel] = arms
    if not panels:
        return print("skipped fig_rq3-budget: no replay files")
    rows = []
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 2.3), squeeze=False)
    for ax, (panel, arms) in zip(axes[0], panels.items()):
        bench, _, checks, sealed = BUDGET_PANELS[panel]
        split = "sealed test" if sealed else "validation"
        b = EXP.spec["benchmarks"][bench]
        for name, t in arms.items():
            rows += [{"bench": bench, "arm": name, "run": r.run, "source": r.source, "round": r.round,
                      "llm_calls": r.calls, "certified": r.certified, "targets": r.targets,
                      "coverage": r.certified / r.targets, "solved_training": r.solved, "picked_from": r.picked_from,
                      "checks": checks, "split": split, "panel": panel} for r in t.itertuples(index=False)]
            r = t.assign(coverage=t.certified / t.targets).groupby("round").agg(
                calls=("calls", "mean"), mean=("coverage", "mean"), lo=("coverage", "min"), hi=("coverage", "max"))
            st = STYLE[name]
            ax.plot(r.calls, r["mean"], color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.2, ms=4,
                    mfc="none" if name != "Full" else st["color"], label=name)
            ax.fill_between(r.calls, r.lo, r.hi, color=st["color"], alpha=0.12, lw=0)
        if "Restart" in arms:
            spend = arms["Restart"].sort_values("round").groupby(["run", "source"]).calls.last().mean()
            full_there, restart_end = at(arms["Full"], spend), at(arms["Restart"], arms["Restart"].calls.max())
            ax.axvline(spend, color="#444444", ls=":", lw=0.8)
            ax.plot([spend], [full_there.mean()], marker="*", color=STYLE["Full"]["color"], ms=9, ls="none")
            ax.annotate(f"Full at Restart's\nspend: {full_there.mean():.2f}", (spend, full_there.mean()),
                        xytext=(spend * 0.6, 0.12), ha="right", fontsize=7,
                        arrowprops=dict(arrowstyle="-", color="#444444", lw=0.6))
            for arm, v in (("Full at Restart's spend", full_there), ("Restart at its spend", restart_end)):
                rows.append({"bench": bench, "arm": arm, "llm_calls": spend, "coverage": v.mean(),
                             "coverage_min": v.min(), "coverage_max": v.max(), "runs": len(v), "checks": checks,
                             "split": split, "panel": panel})
        cfg = load_config(b.base)
        dataset = (SEALED[bench].get("dataset") if sealed else None) or b.dataset or cfg.domain.dataset
        kind, tb = ("sealed", sealed_bench(b)) if sealed else (None, b)
        shots = [r for r in runs_of(b.base) if not certified_checks(checked(oneshot_csv(r, dataset), kind)).empty]
        if shots:
            cov = per_list(pd.concat([outcomes(oneshot_csv(r, dataset), tb, "OneShot", r.name, kind=kind)
                                      for r in shots]))
            calls = float(np.mean([oneshot_calls(r) for r in shots]))
            ax.plot([calls], [cov.coverage.mean()], marker="D", color="#444444", ms=5, ls="none", label="OneShot")
            rows.append({"bench": bench, "arm": "OneShot", "llm_calls": calls, "coverage": cov.coverage.mean(),
                         "coverage_min": cov.coverage.min(), "coverage_max": cov.coverage.max(), "runs": len(shots),
                         "checks": CHECKS, "split": split, "panel": panel})
        ax.set_title(RQ3_BENCHES[bench] + (" (test)" if sealed else "") +
                     ("" if checks == "fixed rule" else " (loop solver)"), fontsize=8)
        ax.set_xlabel("LLM calls spent (per training set)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(left=0)
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, loc="lower right", fontsize=7)
    axes[0][0].set_ylabel("certified coverage (fraction)")
    fig.tight_layout()
    save_figure(fig, "fig_rq3-budget")
    pd.DataFrame(rows).to_csv(OUT / "figures" / "fig_rq3-budget.csv", index=False)


# ---------------------------------------------------------------- RQ2 sequences, synthesis time, amortized cost
# Decisions come from each study's episodes.csv
# (what the manager did as it ran); verdicts and margins from certs.csv.

SEQ = RESULTS / "sequential"
DECISIONS = ["retain", "switch", "generate", "defer"]


def episodes_of(study: str, mode: str) -> pd.DataFrame:
    frames = [pd.read_csv(seq).assign(study=study, mode=mode, sequence=seq.parent.name)
              for seq in sorted((SEQ / study / mode).glob("*/episodes.csv"))]     # a sequence that died early has none
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def per_sequence(ep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (study, mode, seq), g in ep.groupby(["study", "mode", "sequence"]):
        counts = {d: int((g.decision == d).sum()) for d in DECISIONS}
        rows.append({"study": study, "mode": mode, "sequence": seq, "episodes": len(g), **counts,
                     "served": counts["retain"] + counts["switch"] + counts["generate"],
                     "generations_attempted": int(g.get("generation_seconds", pd.Series(dtype=float)).notna().sum()),
                     "final_library": int(g.library_size.iloc[-1]),
                     "llm_calls": g.get("llm_calls", pd.Series(dtype=float)).sum(),
                     "prompt_tokens": g.get("prompt_tokens", pd.Series(dtype=float)).sum(),
                     "output_tokens": g.get("output_tokens", pd.Series(dtype=float)).sum()})
    df = pd.DataFrame(rows)
    if not df.empty:
        price = PRICES["anthropic/claude-haiku-5.5"]
        df["llm_dollars"] = (df.prompt_tokens * price[0] + df.output_tokens * price[1]) / 1e6
        df["split"] = "validation"
    return df


def served_by_earlier(ep: pd.DataFrame) -> pd.Series:
    """Per episode of a growing run: served by a list generated earlier in the same sequence."""
    gen_at = ep.deployed.astype(str).str.extract(r"^gen_t(\d+)$")[0].astype(float)
    return ep.decision.isin(["retain", "switch"]) & gen_at.notna() & (gen_at < ep.episode)


def synthesis_time() -> None:
    """Item 1: synthesis (the joint query on the bare MDP) against certifying a stored list, per instance size, from
    the timing parts (tab_rq2-synthesis-time.<part>.csv; timings, not certifications)."""
    parts = sorted((OUT / "tables").glob("tab_rq2-synthesis-time.*.csv"))
    final = OUT / "tables" / "tab_rq2-synthesis-time.csv"
    if parts:
        pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(final, index=False)
    if not final.exists():
        return
    df = pd.read_csv(final)
    df["kind"] = np.where(df.method == "synthesis", "synthesis", "certification")
    df.groupby(["domain", "size", "kind"]).agg(
        checks=("instance", "count"), states_median=("states", "median"),
        construction_s_median=("construction_s", "median"), check_s_median=("check_s", "median"),
        check_s_max=("check_s", "max"), not_done=("outcome", lambda s: int((s != "done").sum()))).reset_index() \
        .to_csv(OUT / "tables" / "tab_rq2-synthesis-time-summary.csv", index=False)
    print("wrote tables/tab_rq2-synthesis-time.csv, tab_rq2-synthesis-time-summary.csv")


def sequence_tables() -> None:
    """Items 2, 4, 5 and 6: random sequences (reuse only), long sequences (frozen against growing), growing mode
    from the other seed libraries, and Pac-Man sequences with each generation attempt's verdict (certs.csv)."""
    rows = []
    for k, study in enumerate(["random_v0", "random_v0_lib2", "random_v0_lib3"], 1):
        for seq, g in episodes_of(study, "frozen").groupby("sequence") if (SEQ / study).exists() else []:
            c = {d: int((g.decision == d).sum()) for d in ("retain", "switch", "defer")}
            rows.append({"sequence": seq, "library": f"seed_none_{k}", **c, "served": c["retain"] + c["switch"],
                         "episodes": len(g), "split": "validation"})
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "tables" / "tab_rq2-random-sequences.csv", index=False)

    frozen, growing = episodes_of("long_v0", "frozen"), episodes_of("long_v0", "growing")
    if not (frozen.empty or growing.empty):
        per_ep = pd.concat([frozen.assign(served_by=frozen.deployed, generated_earlier_in_sequence=False),
                            growing.assign(served_by=growing.deployed,
                                           generated_earlier_in_sequence=served_by_earlier(growing))], ignore_index=True)
        per_ep.to_csv(OUT / "tables" / "tab_rq2-long-episodes.csv", index=False)
        summary = per_sequence(per_ep)
        m = frozen[["sequence", "episode", "decision"]].merge(
            growing.assign(earlier=served_by_earlier(growing))[["sequence", "episode", "decision", "earlier"]],
            on=["sequence", "episode"], suffixes=("_frozen", "_growing"))
        spared = m[(m.decision_frozen == "defer") & m.earlier].groupby("sequence").size()
        summary["frozen_defer_growing_served_by_earlier"] = [int(spared.get(r.sequence, 0)) if r["mode"] == "growing"
                                                             else np.nan for _, r in summary.iterrows()]
        summary.to_csv(OUT / "tables" / "tab_rq2-long-sequences.csv", index=False)

    frames = [f for f in (episodes_of(s, "growing") for s in ("shift_v0_lib2", "shift_v0_lib3")) if not f.empty]
    if frames:
        ep = pd.concat(frames, ignore_index=True)
        ep.to_csv(OUT / "tables" / "tab_rq2-growing-libs-episodes.csv", index=False)
        per_sequence(ep).to_csv(OUT / "tables" / "tab_rq2-growing-libs.csv", index=False)

    frames = [f for f in (episodes_of("pacman_v0", m) for m in ("frozen", "growing")) if not f.empty]
    if frames:
        per_sequence(pd.concat(frames, ignore_index=True)).to_csv(OUT / "tables" / "tab_rq2-pacman-sequences.csv",
                                                                   index=False)
        attempts = []         # every generation attempt (lengths 25 and 30: ids 7 and 9) with its certs.csv verdict
        for seq in sorted((SEQ / "pacman_v0" / "growing").glob("*")):
            if not (seq / "episodes.csv").exists():
                continue
            e, c = pd.read_csv(seq / "episodes.csv"), sequence_checks("pacman_v0", "growing", seq.name)
            for _, r in e[e.instance.isin([7, 9]) & e.get("generation_seconds", pd.Series(dtype=float)).notna()].iterrows():
                new = c[(c.episode == r.episode) & (c.rule_list == f"gen_t{r.episode}")]
                attempts.append({"sequence": seq.name, "episode": r.episode, "instance": r.instance,
                                 "length": {7: 25, 9: 30}[int(r.instance)],
                                 "verdict": new.outcome.iloc[0] if len(new) else r.get("new_outcome"),
                                 "margin": new.margin.iloc[0] if len(new) else np.nan,
                                 "llm_calls": r.get("llm_calls"), "output_tokens": r.get("output_tokens"),
                                 "split": "validation"})
        pd.DataFrame(attempts).to_csv(OUT / "tables" / "tab_rq2-pacman-generations.csv", index=False)
    print("wrote tables/tab_rq2-random-sequences.csv and the sequence tables")


def llm_spend() -> None:
    """Item 3: tokens and dollars of every LLM's core runs, from the runs' LLM task logs (llm_tasks.jsonl): the
    provider's reported cost where calls logged it, and tokens at the list price for all calls."""
    rows = []
    for llm, (suffix, root) in {"Claude Haiku 5.5": ("", "core"), **{k: (v, "ablations") for k, v in OTHER_LLMS.items()}}.items():
        for bench in RQ1_BENCHES:
            runs = runs_of(EXP.spec["benchmarks"][bench].base + suffix, root)
            if not runs:
                continue
            model = json.loads((runs[0] / "config.json").read_text(encoding="utf-8"))["llm"]["model"]
            price = PRICES.get(model, (np.nan, np.nan))
            prompt = output = calls = n_rep = 0
            reported = 0.0
            for run in runs:
                path = run / "llm_tasks.jsonl"
                for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
                    res = json.loads(line).get("result") or {}
                    prompt += res.get("prompt_tokens") or 0
                    output += res.get("output_tokens") or 0
                    calls += 1
                    cost = ((res.get("meta") or {}).get("usage") or {}).get("cost")
                    if cost is not None:
                        reported += cost
                        n_rep += 1
            rows.append({"llm": llm, "model": model, "bench": bench, "runs": len(runs), "calls": calls,
                         "prompt_tokens": prompt, "output_tokens": output,
                         "dollars_at_list_price": (prompt * price[0] + output * price[1]) / 1e6,
                         "dollars_reported": reported if n_rep else np.nan, "calls_with_reported_cost": n_rep})
    pd.DataFrame(rows).to_csv(OUT / "tables" / "tab_rq1-llm-spend.csv", index=False)
    print("wrote tables/tab_rq1-llm-spend.csv")


def llm_ablation(sealed: bool = False) -> None:
    """Item 7: NoRepair for the other LLMs (Full and NoRepair, T4's layout), on the validation targets or, with
    `sealed`, the sealed test targets (`tab_rq3-ablation-llm-sealed`; the training pass rate and calls are the
    lists' own, the same in both)."""
    kind, split = ("sealed", "sealed test") if sealed else (None, "validation")
    rows, tex = [], [r"\toprule",
                     r"LLM / configuration & Certified coverage (\%) & Training pass rate (\%) & LLM calls per set \\"]
    for bench, title in RQ3_BENCHES.items():
        b = EXP.spec["benchmarks"][bench]
        dataset = train_data = b.dataset or load_config(b.base).domain.dataset
        if sealed:
            b, dataset = sealed_bench(b), SEALED[bench].get("dataset") or dataset
        frozen = "final" if bench == "pacman" else "determinized"
        block = []
        for label, suffix in OTHER_LLMS.items():
            full = runs_of(b.base + suffix, "ablations")
            for name, rules in (("Full", frozen), ("NoRepair", "norepair_determinized")):
                if not full or any(certified_checks(checked(csv_path(r, dataset, rules), kind)).empty for r in full):
                    continue
                cov = per_list(pd.concat([outcomes(csv_path(r, dataset, rules), b, name, r.name, kind=kind)
                                          for r in full]))
                passed = [x for r in full for x in training_passed(checked(csv_path(r, train_data, rules)))]
                c, p = stats(cov.coverage), stats(passed)
                calls = [calls_per_set(r) for r in full]
                rows += [{"bench": bench, "llm": label, "configuration": name, "metric": m, "value": v,
                          "split": split, "runs": len(full)}
                         for m, v in (("coverage_mean", c["mean"]), ("coverage_sd", c["sd"]), ("coverage_n", c["n"]),
                                      ("certified_per_list", cov.certified.mean()),
                                      ("training_pass_mean", p["mean"]), ("training_pass_n", p["n"]),
                                      ("llm_calls_per_set", np.mean(calls)),
                                      ("inconclusive_share", cov.inconclusive.mean()),
                                      ("timeout_share", cov.timeout.mean()))]
                block.append(f"{label}, {name} & {pct(c['mean'], c['sd'], c['n'])} & {pct(p['mean'], p['sd'], p['n'])}"
                             f" & ${np.mean(calls):.1f}$ \\\\")
        if block:
            tex += [r"\midrule", rf"\multicolumn{{4}}{{l}}{{\emph{{{title}}}}} \\"] + block
    write_table("tab_rq3-ablation-llm" + ("-sealed" if sealed else ""), tex + [r"\bottomrule"], rows)


def amortized_cost() -> None:
    """Generation plus certification time per held-out instance solved, per LLM and benchmark (Full, frozen lists,
    validation): per list (generation wall time of its training set + the seconds of every held-out check of it) /
    held-out instances it certifies, and pooled (sums over lists / total certified). Certification seconds come from
    certs.csv, replaced by the cluster measurement where the same list was timed on the same instance (item 1:
    tab_rq2-synthesis-time.csv, wall_s), which takes precedence. Beside them, the measured synthesis
    time per instance (the joint query on the bare MDP) for the same benchmark. `tab_rq2-amortized-cost.csv`."""
    item1 = OUT / "tables" / "tab_rq2-synthesis-time.csv"
    timed = pd.read_csv(item1) if item1.exists() else pd.DataFrame(columns=["method", "domain", "instance", "wall_s"])
    timed_cert = {}                                      # (bench, run, list, instance) -> measured certification seconds
    for r in timed[timed.method != "synthesis"].itertuples(index=False):
        m = re.match(r"(fam|pacman)(\d)_list(\d)", r.method)
        if m and r.domain in ("families", "pacman"):
            timed_cert[(r.domain, f"seed_none_{m.group(2)}", m.group(3), int(r.instance))] = r.wall_s
    synth = timed[timed.method == "synthesis"].groupby("domain").wall_s.median().to_dict()
    rows = []
    for bench in RQ1_BENCHES:
        b = EXP.spec["benchmarks"][bench]
        dataset = b.dataset or load_config(b.base).domain.dataset
        frozen = "final" if bench == "pacman" else "determinized"
        arms = {"Claude Haiku 5.5": (runs_of(b.base), "local machine")}
        arms.update({llm: (runs_of(b.base + s, "ablations"), "cluster") for llm, s in OTHER_LLMS.items()})
        for llm, (runs, gen_source) in arms.items():
            for run in runs:
                records = sample_records(run)
                checks = certified_checks(checked(csv_path(run, dataset, frozen)), "validation")
                if checks.empty:
                    continue
                checks = checks[checks.target.isin(held_out(b))]
                own = timed_cert if llm == "Claude Haiku 5.5" else {}   # item 1 timed the core (Haiku) lists only
                for sid, g in checks.groupby("source"):
                    record = records.get(int(sid), {})
                    secs = [own.get((bench, run.name, str(sid), int(t)), s) for t, s in zip(g.target, g.seconds)]
                    n_timed = sum((bench, run.name, str(sid), int(t)) in own for t in g.target)
                    gen = record.get("total_time") or np.nan
                    cert = float(np.nansum(secs))
                    solved = int(g.certified.sum())
                    rows.append({"bench": bench, "llm": llm, "run": run.name, "list": sid, "generation_s": gen,
                                 "generation_source": gen_source, "certification_s": cert,
                                 "certification_checks": len(g), "certification_from_timing": n_timed,
                                 "solved": solved, "amortized_s": (gen + cert) / solved if solved else np.nan,
                                 "synthesis_s_per_instance_timed": synth.get(bench), "split": "validation"})
    df = pd.DataFrame(rows)
    pooled = df.groupby(["bench", "llm"]).apply(lambda g: pd.Series({
        "lists": len(g), "solved": g.solved.sum(), "generation_s": g.generation_s.sum(),
        "certification_s": g.certification_s.sum(),
        "pooled_amortized_s": (g.generation_s.sum() + g.certification_s.sum()) / g.solved.sum() if g.solved.sum() else np.nan,
        "mean_amortized_s": g.amortized_s.mean(), "lists_solving_none": int((g.solved == 0).sum()),
        "synthesis_s_per_instance_timed": g.synthesis_s_per_instance_timed.iloc[0]})).reset_index()
    df.to_csv(OUT / "tables" / "tab_rq2-amortized-cost-per-list.csv", index=False)
    pooled.assign(split="validation").to_csv(OUT / "tables" / "tab_rq2-amortized-cost.csv", index=False)
    print("wrote tables/tab_rq2-amortized-cost.csv, -per-list.csv")
    print(pooled.round(1).to_string(index=False))


EXP = None


def main(argv=None):
    global EXP, CHECKS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", default="shift_v0", help="out/results/sequential/<study>")
    parser.add_argument("--episodes", type=int, default=12, help="episodes per sequence (figure axis)")
    args = parser.parse_args(argv)
    EXP = experiment(RQ1_SPEC)
    data = rq1_outcomes(EXP)
    rq1_tables(data, llm_outcomes(EXP))
    llm_table(llm_outcomes(EXP, sealed=True), "tab_rq1-coverage-llm-sealed", "sealed test")
    llm_spend()
    llm_ablation()
    llm_ablation(sealed=True)
    synthesis_time()
    amortized_cost()
    sequence_tables()
    sealed = rq1_outcomes(EXP, sealed=True)
    rq1_sealed_table(sealed)
    rq1_size_figure(data, sealed)
    rq1_by_family(data, sealed)
    solvability()
    rq1_misses(data)
    rq2_tables(sequential(args.study), args.episodes)
    rq2_episodes(args.study)
    generation_cost(args.study)
    controller_sizes()
    rq3_tables()
    rq3_sealed_table()
    rq3_budget_figure()
    pacman_narrowing()


if __name__ == "__main__":
    main()
