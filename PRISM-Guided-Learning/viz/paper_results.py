"""Tables and figures for the SEAMS evaluation section, built only from existing runs (no LLM or PRISM calls).

RQ1 (the paper's RQ1, certified coverage) reads the core runs and baselines of configs/experiments/rq1_comparison.yaml;
RQ2 (adaptation over sequences) reads out/results/sequential/<study>/; RQ3 (ingredients of generation; the
repository's "rq2") reads the core runs, their no-repair picks and the pure-restart runs of the same loop. What each
item reports, which split it uses and what is missing is in out/paper/RESULTS-NOTES.md (written by hand).

Usage (from PRISM-Guided-Learning/): python viz/paper_results.py [--study shift_v0]
Writes out/paper/tables/<label>.tex (a tabular body) and .csv (long format), and out/paper/figures/<label>.pdf.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402
from core.domain import load_domain  # noqa: E402
from experiments.base import experiment, held_out  # noqa: E402
from experiments.comparison import Comparison  # noqa: E402
from results_io import sample_records  # noqa: E402
from transfer import csv_path, parse_ids  # noqa: E402

RESULTS = ROOT / "out" / "results"
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
    return sorted(p for p in (RESULTS / root / name).glob("seed_*") if (p / "config.json").exists())


# ---------------------------------------------------------------- RQ1: per (controller, held-out target) outcomes

def outcomes(path: Path, bench, approach: str, run: str = "", sources: Optional[List[str]] = None,
             source_groups: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """One row per (rule list or controller, held-out target) of a transfer CSV, on the list's own group's targets
    where the benchmark has groups (a source checked on every group, "*", counts once per group)."""
    if path is None or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["source"] = df.source.astype(str)
    if sources is not None:
        df = df[df.source.isin(sources)]
    trained = df.trained.astype(bool) if "trained" in df else False
    df = df[df.target.isin(held_out(bench)) & ~trained].copy()
    df["certified"] = df.certified.fillna(False).astype(bool)
    errors = df["error"].fillna("").astype(str) if "error" in df else pd.Series("", index=df.index)
    df["timeout"] = errors.str.contains("Timeout")
    df["error_other"] = (errors != "") & ~df.timeout
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


def rq1_outcomes(exp) -> Dict[str, pd.DataFrame]:
    """{"frozen": ..., "permissive": ...}: per-target outcomes of every RQ1 controller. Full: frozen = the list with
    `any` resolved by the pick's order (determinized; Pac-Man: the narrowed lists as kept, already resolved),
    permissive = the list as kept, `any` open (none for Pac-Man). Baselines: their CSV, and the ties-open CSV as the
    permissive controller where one exists (otherwise the controller has one action per state and both coincide)."""
    spec = exp.spec
    frozen, permissive = [], []
    baselines = {b["name"]: b for b in spec["baselines"]}
    for name, bench in spec["benchmarks"].items():
        if name not in RQ1_BENCHES and name not in EXTRA_BENCHES:
            continue
        cfg = load_config(bench.base)
        dataset = bench.dataset or cfg.domain.dataset
        for run in runs_of(bench.base):
            fz = "final" if name == "pacman" else "determinized"
            frozen.append(outcomes(csv_path(run, dataset, fz), bench, "Full", run.name))
            if name != "pacman":
                permissive.append(outcomes(csv_path(run, dataset, "final"), bench, "Full", run.name))
        for approach in APPROACHES[1:]:
            base = BASELINES[approach]
            base = base.get(name) if isinstance(base, dict) else base
            b = baselines.get(base)
            if b is None or name not in b["config"]:
                continue
            src = Comparison._sources(b["sources"][name])
            config = b["config"][name]
            frozen.append(outcomes(Comparison._csv(config), bench, approach, "", src["sources"],
                                   src["source_groups"]))
            opened = Comparison._csv(config, "open")
            permissive.append(outcomes(opened if opened.exists() else Comparison._csv(config), bench, approach, "",
                                       src["sources"], src["source_groups"]))
    return {"frozen": pd.concat(frozen, ignore_index=True), "permissive": pd.concat(permissive, ignore_index=True)}


def per_list(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["bench", "approach", "list"]).agg(
        coverage=("certified", "mean"), certified=("certified", "sum"), targets=("target", "count"),
        timeout=("timeout", "mean"), error=("error_other", "mean")).reset_index()


def rq1_tables(data: Dict[str, pd.DataFrame]) -> None:
    lists = {k: per_list(v) for k, v in data.items()}
    rows = []
    for bench in list(RQ1_BENCHES) + list(EXTRA_BENCHES):
        for approach in APPROACHES:
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
                    entry.update({"timeout_share": g.timeout.mean(), "error_share": g.error.mean()})
            if len(entry) > 2:
                entry.update({"inconclusive_share": np.nan, "split": "validation",
                              "llm": "anthropic/claude-haiku-5.5" if approach == "Full" else ""})
                rows.append(entry)
    table = pd.DataFrame(rows)
    long = table.melt(id_vars=["bench", "approach", "split", "llm"], var_name="metric").dropna(subset=["value"])
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
            tex.append(f"{r.approach} & {cells[0]} & {cells[1]} & -- & ${100 * r.timeout_share:.1f}$ \\\\")
    tex.append(r"\bottomrule")
    write_table("tab_rq1-coverage", tex, long.to_dict("records"))

    # Per LLM (Full only): only Haiku 5.5 has runs of the core loop
    tex = [r"\toprule", r"LLM & Permissive (\%) & Frozen (\%) & Inconclusive (\%) & Timeout (\%) \\"]
    llm_rows = []
    for bench, title in RQ1_BENCHES.items():
        r = table[(table.bench == bench) & (table.approach == "Full")].iloc[0]
        tex += [r"\midrule", rf"\multicolumn{{5}}{{l}}{{\emph{{{title}}}}} \\",
                f"Claude Haiku 5.5 & {pct(r.get('permissive_mean'), r.get('permissive_sd'), count(r.get('permissive_n')))}"
                f" & {pct(r.frozen_mean, r.frozen_sd, int(r.frozen_n))} & -- & ${100 * r.timeout_share:.1f}$ \\\\"]
        llm_rows += [{"bench": bench, "llm": "anthropic/claude-haiku-5.5", "metric": m, "value": r.get(m)}
                     for m in ("permissive_mean", "permissive_sd", "permissive_n", "frozen_mean", "frozen_sd",
                               "frozen_n", "timeout_share") if not pd.isna(r.get(m))]
    tex.append(r"\bottomrule")
    write_table("tab_rq1-coverage-llm", tex, [{**x, "split": "validation"} for x in llm_rows])


def rq1_size_figure(data: Dict[str, pd.DataFrame]) -> None:
    """Certified coverage per held-out size: Full's permissive lists (Pac-Man: its frozen lists, which have no
    permissive form), each baseline's permissive controller (= frozen where it has one action per state)."""
    df = pd.concat([data["permissive"], data["frozen"][(data["frozen"].bench == "pacman")
                                                      & (data["frozen"].approach == "Full")]])
    rows = []
    fig, axes = plt.subplots(1, len(SIZE), figsize=(7.16, 2.1), sharey=True)
    for ax, (bench, (column, xlabel, largest)) in zip(axes, SIZE.items()):
        b = df[df.bench == bench]
        sizes = _sizes(bench, column)
        b = b.assign(size=b.target.map(sizes))
        span = max(b["size"].max(), largest) - min(b["size"].min(), largest)
        for k, approach in enumerate(APPROACHES):
            a = b[b.approach == approach]
            if a.empty:
                continue
            by_list = a.groupby(["size", "list"]).certified.mean().reset_index()
            curve = by_list.groupby("size").certified.agg(["mean", "std", "count"]).reset_index()
            st = STYLE[approach]
            dodge = (k - 2) * 0.012 * span          # small horizontal offsets so coinciding lines stay visible
            ax.plot(curve["size"] + dodge, curve["mean"], color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.2,
                    ms=4.5, mfc="none" if approach != "Full" else st["color"], label=approach)
            rows += [{"bench": bench, "approach": approach, "size": s, "coverage_mean": m, "coverage_sd": sd,
                      "n_lists": c, "targets_per_size": int(a[a["size"] == s].target.nunique()),
                      "controller": "frozen" if (bench == "pacman" and approach == "Full") else "permissive",
                      "split": "validation"}
                     for s, m, sd, c in curve.itertuples(index=False)]
        ax.axvline(largest, color="#444444", ls=(0, (4, 3)), lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(RQ1_BENCHES[bench], fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("certified coverage (fraction)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    save_figure(fig, "fig_rq1-size")
    pd.DataFrame(rows).to_csv(OUT / "figures" / "fig_rq1-size.csv", index=False)


def _sizes(bench: str, column: str) -> Dict[int, float]:
    exp_bench = EXP.spec["benchmarks"][bench]
    cfg = load_config(exp_bench.base)
    dataset = exp_bench.dataset or cfg.domain.dataset
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
                         "requirement": worst, "timeout": r.timeout, "split": "validation"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "tables" / "tab_rq1-misses.csv", index=False)
    print("wrote tables/tab_rq1-misses.csv")


# ---------------------------------------------------------------- RQ2: the sequential study

def sequential(study: str) -> Dict[str, pd.DataFrame]:
    episodes, checks = [], []
    for mode, setting in (("growing", "Extended"), ("frozen", "Frozen")):
        for seq in sorted((RESULTS / "sequential" / study / mode).glob("s*")):
            e = pd.read_csv(seq / "episodes.csv").assign(setting=setting, sequence=seq.name)
            c = pd.read_csv(seq / "checks.csv").assign(setting=setting, sequence=seq.name)
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
                                  gauss_seidel_fallback=("solver", lambda s: int((s.fillna("") != "interval iteration")
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


# ---------------------------------------------------------------- RQ3: ingredients of generation

RQ3_BENCHES = {"grids": "Random grids", "families": "Grid-world families", "pacman": "Pac-Man",
               "mer": "MER across $x$", "mer_n": "MER across $n$"}
RESTARTS = {"mer": "port_mer_resample"}          # pure restarts of the core loop (planner.retry: always)


def members_passed(run: Path, norepair: bool = False) -> List[float]:
    """Per training set, the fraction of its members the final list passes (worst case), as the loop ends. A
    no-repair pick back-filled after its run (norepair.json, MER) records only whether every member passes: 1 if so,
    unknown otherwise."""
    out = []
    backfilled = json.loads((run / "norepair.json").read_text(encoding="utf-8")) if norepair else {}
    for sid, record in sample_records(run).items():
        if norepair and "norepair" not in record.get("pick", {}):
            out.append(1.0 if backfilled[str(sid)]["success"] else np.nan)
            continue
        members = record["pick"]["norepair"]["members"] if norepair else record.get("members", [])
        out.append(np.mean([bool(m.get("success")) for m in members]) if members else np.nan)
    return out


def calls_per_set(run: Path) -> float:
    records = sample_records(run)
    return sum(it.get("llm_calls") or 0 for r in records.values() for it in r.get("iterations", [])) / len(records)


def rq3_tables() -> None:
    rows, tex = [], [r"\toprule", r"Configuration & Certified coverage (\%) & Training pass rate (\%) \\"]
    for bench, title in RQ3_BENCHES.items():
        b = EXP.spec["benchmarks"][bench]
        cfg = load_config(b.base)
        dataset = b.dataset or cfg.domain.dataset
        frozen = "final" if bench == "pacman" else "determinized"     # Pac-Man: the narrowed lists, as in T1
        configs = {"Full": (runs_of(b.base), frozen, False),
                   "Restart": (runs_of(RESTARTS[bench], "ablations") if bench in RESTARTS else [], frozen, False),
                   "NoRepair": (runs_of(b.base), "norepair_determinized", True)}
        block = []
        for name, (runs, rules, norepair) in configs.items():
            if not runs or not all(csv_path(r, dataset, rules).exists() for r in runs):
                continue
            cov = pd.concat([outcomes(csv_path(r, dataset, rules), b, name, r.name) for r in runs])
            cov = per_list(cov)
            passed = [x for r in runs for x in members_passed(r, norepair)]
            c, p = stats(cov.coverage), stats(passed)
            calls = [calls_per_set(r) for r in runs]
            rows += [{"bench": bench, "configuration": name, "metric": m, "value": v, "split": "validation",
                      "runs": len(runs)}
                     for m, v in (("coverage_mean", c["mean"]), ("coverage_sd", c["sd"]), ("coverage_n", c["n"]),
                                  ("certified_per_list", cov.certified.mean()),
                                  ("targets_per_list", cov.targets.mean()),
                                  ("training_pass_mean", p["mean"]), ("training_pass_sd", p["sd"]),
                                  ("training_pass_n", p["n"]), ("llm_calls_per_set", np.mean(calls)))]
            block.append(f"{name} & {pct(c['mean'], c['sd'], c['n'])} & {pct(p['mean'], p['sd'], p['n'])} \\\\")
        if bench == "grids":           # the hand-written template does not use `closer`: without it, it is unchanged
            hand = per_list(outcomes(Comparison._csv("configs/baselines/template_grid_large.yaml"), b, "Hand", "",
                                     ["cautious"], {}))
            c = stats(hand.coverage)
            rows += [{"bench": bench, "configuration": "NoSP (Hand)", "metric": m, "value": v, "split": "validation",
                      "runs": 1} for m, v in (("coverage_mean", c["mean"]), ("coverage_n", c["n"]),
                                              ("certified_per_list", hand.certified.mean()))]
            block.append(f"NoSP (Hand) & {pct(c['mean'], c['sd'], c['n'])} & -- \\\\")
        tex += [r"\midrule", rf"\multicolumn{{3}}{{l}}{{\emph{{{title}}}}} \\"] + block
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
    full = per_list(pd.concat([outcomes(csv_path(r, "grid_20_large.csv", "determinized"), b, "Full", r.name)
                               for r in runs_of("port_ts_noclose")]))
    hand = per_list(outcomes(Comparison._csv("configs/baselines/template_grid_large.yaml"), b, "Hand", "",
                             ["cautious"], {}))
    pd.DataFrame([{"benchmark": "grids_noclose (grid_20_large, targets 12-19)", "configuration": n,
                   "certified_per_list": g.certified.mean(), "certified_sd": g.certified.std(ddof=1),
                   "targets_per_list": g.targets.mean(), "n_lists": len(g),
                   "runs": g.list.str.split("/").str[0].nunique(), "loop": loop, "split": "validation"}
                  for n, g, loop in (("Full (NoSP)", full, "port_ts_noclose: core loop before the Oct 17 gridworld "
                                                          "feedback fix, no strategy hint"),
                                     ("Hand (NoSP)", hand, "not generated"))]) \
        .to_csv(OUT / "tables" / "tab_rq3-nosp.csv", index=False)
    print("wrote tables/tab_rq3-nosp.csv")


def rq3_budget_figure(bench: str = "mer") -> None:
    """F3: certified coverage of the held-out targets against LLM calls spent per training set, for Full and Restart
    (src/budget_transfer.py's chain curves: per round budget R, the pick over every chain cut to its first R rounds,
    `any` resolved). One point per R: the runs' mean calls and mean coverage, the band their range. The dotted line
    is Restart's mean total spend; the star, Full's value there (each run's last R within that many calls)."""
    b = EXP.spec["benchmarks"][bench]
    arms = {"Full": runs_of(b.base), "Restart": runs_of(RESTARTS[bench], "ablations")}
    tables = {}
    for name, runs in arms.items():
        per_run = []
        for r in runs:
            path = r / "budget_transfer_chains.csv"
            if not path.exists():
                return print(f"skipped fig_rq3-budget: no {path.relative_to(ROOT)}")
            df = pd.read_csv(path)
            df = df[df.target.isin(held_out(b)) & ~df.trained.astype(bool)]
            df["certified"] = df.certified.fillna(False).astype(bool)
            per_run.append(df.groupby(["source", "round"]).agg(
                calls=("llm_calls", "first"), certified=("certified", "sum"), targets=("target", "size"),
                solved=("solved", "first"), picked_from=("picked_from", "first")).reset_index().assign(run=r.name))
        tables[name] = pd.concat(per_run, ignore_index=True)
    rows = []
    for name, t in tables.items():
        rows += [{"arm": name, "run": r.run, "source": r.source, "round": r.round, "llm_calls": r.calls,
                  "certified": r.certified, "targets": r.targets, "coverage": r.certified / r.targets,
                  "solved_training": r.solved, "picked_from": r.picked_from, "bench": bench, "split": "validation"}
                 for r in t.itertuples(index=False)]

    def at(t: pd.DataFrame, budget: float) -> pd.Series:
        """Per run: the coverage of its last round within `budget` calls (mean over its training sets)."""
        out = {}
        for run, g in t.groupby("run"):
            within = g[g.calls <= budget + 1e-9].sort_values("round").groupby("source").tail(1)
            out[run] = (within.certified / within.targets).mean() if len(within) == g.source.nunique() else np.nan
        return pd.Series(out)

    restart_spend = tables["Restart"].sort_values("round").groupby(["run", "source"]).calls.last().mean()
    top = max(t.calls.max() for t in tables.values())
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for name, t in tables.items():        # one point per round budget R: mean calls and coverage over runs
        r = t.assign(coverage=t.certified / t.targets).groupby("round").agg(
            calls=("calls", "mean"), mean=("coverage", "mean"), lo=("coverage", "min"), hi=("coverage", "max"))
        st = STYLE[name]
        ax.plot(r.calls, r["mean"], color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.2, ms=4,
                mfc="none" if name != "Full" else st["color"], label=name)
        ax.fill_between(r.calls, r.lo, r.hi, color=st["color"], alpha=0.12, lw=0)
    full_there = at(tables["Full"], restart_spend)
    ax.axvline(restart_spend, color="#444444", ls=":", lw=0.8)
    ax.plot([restart_spend], [full_there.mean()], marker="*", color=STYLE["Full"]["color"], ms=9, ls="none")
    ax.annotate(f"Full at Restart's\nspend: {full_there.mean():.2f}", (restart_spend, full_there.mean()),
                xytext=(restart_spend - 4, 0.12), ha="right", fontsize=7,
                arrowprops=dict(arrowstyle="-", color="#444444", lw=0.6))
    rows.append({"arm": "Full at Restart's spend", "llm_calls": restart_spend, "coverage": full_there.mean(),
                 "coverage_min": full_there.min(), "coverage_max": full_there.max(), "runs": len(full_there),
                 "bench": bench, "split": "validation"})
    restart_end = at(tables["Restart"], top)
    rows.append({"arm": "Restart at its spend", "llm_calls": restart_spend, "coverage": restart_end.mean(),
                 "coverage_min": restart_end.min(), "coverage_max": restart_end.max(), "runs": len(restart_end),
                 "bench": bench, "split": "validation"})
    ax.set_xlabel("LLM calls spent (per training set)")
    ax.set_ylabel("certified coverage (fraction)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(left=0)
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, "fig_rq3-budget")
    pd.DataFrame(rows).to_csv(OUT / "figures" / "fig_rq3-budget.csv", index=False)


EXP = None


def main(argv=None):
    global EXP
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", default="shift_v0", help="out/results/sequential/<study>")
    parser.add_argument("--episodes", type=int, default=12, help="episodes per sequence (figure axis)")
    args = parser.parse_args(argv)
    EXP = experiment(RQ1_SPEC)
    data = rq1_outcomes(EXP)
    rq1_tables(data)
    rq1_size_figure(data)
    rq1_misses(data)
    rq2_tables(sequential(args.study), args.episodes)
    rq3_tables()
    rq3_budget_figure()


if __name__ == "__main__":
    main()
