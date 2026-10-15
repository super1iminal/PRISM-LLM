"""RQ1: how our best configurations compare with the baselines, per benchmark. For our cells, the numbers are means
over runs (range in parentheses); a baseline is one fit (src/run_baseline.py), its lists the run's lists. Columns:
held-out targets certified per rule list (on its own group, and on every target), the best list, the library (targets
certified by at least one list), certified with ties left open, training sets solved and cost.

Each configuration (a cell or a baseline) belongs to a method (`method`, default its name), e.g. trees over raw
variables and over our features are both 1-2-3-Go!. Per benchmark, a method's best configuration is the one with the
highest mean certified per list on the experiment's targets (the validation targets), then the largest library, then
the lowest cost; only fully certified configurations compete. The tables keep every configuration; the figures show
only the best of each method (rq1_<benchmark>.png, and the horizon figure `horizon_figure` describes). The choice is
written to chosen.csv: the test pass uses the configurations chosen on validation and never re-chooses on test results.
"""
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import load_config, load_file
from experiments.base import Benchmark, Experiment, held_out, span
from experiments.metrics import transfer_numbers
from experiments.store import CORE_PATH
from run_baseline import BASELINES_PATH, BaselineConfig
from settings import DOMAINS_PATH, PROJECT_ROOT
from transfer import DEFAULT_CONFIG, TransferConfig, csv_path, parse_ids


def short_label(bench: Benchmark) -> str:
    """A benchmark's label without its parenthesized details, for figure titles."""
    return (bench.label or bench.name).split(" (")[0]


def _pyplot():
    sys.path.insert(0, str(PROJECT_ROOT / "viz"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


class Comparison(Experiment):
    root = CORE_PATH

    def baselines(self) -> List[Dict[str, Any]]:
        return self.spec.get("baselines") or []

    def methods(self) -> Dict[str, str]:
        """Configuration name -> the method it belongs to."""
        out = {c.name: c.method or c.name for c in self.spec["cells"]}
        out.update({b["name"]: b.get("method") or b["name"] for b in self.baselines()})
        return out

    @staticmethod
    def _csv(config: str, rules: str = "final") -> Path:
        settings = load_file(PROJECT_ROOT / config, BaselineConfig)
        stem = Path(settings.target_dataset or settings.dataset).stem
        return BASELINES_PATH / Path(config).stem / f"transfer_{stem}{'_open' if rules == 'open' else ''}.csv"

    @staticmethod
    def _sources(spec) -> Dict[str, Any]:
        """(source list, source -> group) from a baseline's `sources` on one benchmark: a list, or group -> list
        (a source listed for every group checks every group: "*")."""
        if isinstance(spec, dict):
            groups_of: Dict[str, set] = {}
            for group, names in spec.items():
                for name in names:
                    groups_of.setdefault(str(name), set()).add(group)
            mapping = {s: ("*" if len(g) == len(spec) and len(spec) > 1 else next(iter(g))) for s, g in groups_of.items()}
            return {"sources": list(mapping), "source_groups": mapping}
        return {"sources": [str(s) for s in spec], "source_groups": {}}

    def extra_jobs(self) -> List[Callable[[int], str]]:
        """Fit and certify each baseline whose CSV misses targets (as written, and with ties open where it has ties),
        on the benchmark's targets only (a config may also list its training instances: run src/run_baseline.py
        directly for those)."""
        jobs = []
        for base in self.baselines():
            for bench_name, config in base["config"].items():
                spec = self.spec["benchmarks"][bench_name].targets
                targets = parse_ids(spec)
                for rules in (("final", "open") if base.get("ties", True) else ("final",)):
                    path = self._csv(config, rules)
                    if path.exists() and set(targets) <= set(pd.read_csv(path).target):
                        continue
                    log = path.parent / f"run_baseline{'_open' if rules == 'open' else ''}.log"

                    def job(workers, config=config, rules=rules, log=log, spec=spec):
                        ok = self._call([sys.executable, "src/run_baseline.py", config, "--rules", rules,
                                         "--set", f"workers={workers}", "--set", f"targets={spec}"], log)
                        return f"[baseline {Path(config).stem} {rules}] {'done' if ok else f'failed (see {log})'}"
                    jobs.append(job)
        return jobs

    def collect(self) -> pd.DataFrame:
        """Our runs and the baselines, one row each. A baseline is counted on the benchmark's held-out targets (for
        Pac-Man, without the horizons our lists train on), so every method is counted on the same targets."""
        runs = super().collect()
        rows = []
        for base in self.baselines():
            for bench_name, config in base["config"].items():
                bench = self.spec["benchmarks"][bench_name]
                targets = held_out(bench)
                src = self._sources(base["sources"][bench_name])
                final = transfer_numbers(self._csv(config), targets, bench.groups, src["source_groups"],
                                         src["sources"])
                opened = transfer_numbers(self._csv(config, "open"), targets, bench.groups, src["source_groups"],
                                          src["sources"]) if base.get("ties", True) else final
                rows.append({"bench": bench_name, "cell": base["name"], "code": base.get("code", base["name"]),
                             "dir": f"baselines/{Path(config).stem}", "baseline": True, **final,
                             **{f"{k}_open": opened.get(k) for k in ("checked", "certified", "certified_all",
                                                                       "library", "by_group")}})
        out = pd.concat([runs.assign(baseline=False), pd.DataFrame(rows)], ignore_index=True)
        return out.assign(method=out.cell.map(self.methods())) if not out.empty else out

    # ------------------------------------------------------------ best configuration per method

    def chosen(self, runs: pd.DataFrame) -> Dict[Tuple[str, str], str]:
        """(benchmark, method) -> its best configuration: the highest mean certified per list, then the largest
        library, then the lowest cost (dollars per set; baselines cost none). Configurations with an unchecked run
        or CSV do not compete."""
        best = {}
        if runs.empty:
            return best
        order = {name: i for i, name in enumerate(self.methods())}
        for (bench, method), g in runs.groupby(["bench", "method"], sort=False):
            ranked = []
            for name, c in g.groupby("cell", sort=False):
                if not c.checked.fillna(False).astype(bool).all() or c.certified.isna().any():
                    continue
                cost = c.dollars.fillna(0).mean() if "dollars" in c else 0.0
                ranked.append((-c.certified.mean(), -c.library.mean(), cost, order.get(name, 0), name))
            if ranked:
                best[(bench, method)] = min(ranked)[-1]
        return best

    def write_chosen(self, best: Dict[Tuple[str, str], str]) -> Path:
        path = self.out / "chosen.csv"
        pd.DataFrame([{"bench": b, "method": m, "configuration": c} for (b, m), c in best.items()],
                     columns=["bench", "method", "configuration"]).to_csv(path, index=False)
        return path

    # ------------------------------------------------------------ report

    def report(self, runs: pd.DataFrame) -> str:
        self.out.mkdir(parents=True, exist_ok=True)
        best = self.chosen(runs)
        self.write_chosen(best)
        lines = [f"# {self.title}", "", self.spec.get("question", "").strip(), "",
                 "Certified: held-out targets certified per rule list (on its own group's targets where there are "
                 "groups). Every target: the same over all targets, whatever the group. Library: targets certified by "
                 "at least one of a run's (or baseline's) lists. Ties open: certified with the action orders left "
                 "open, i.e. however ties are broken. Ours: mean over runs, range in parentheses.", "",
                 "**Best configuration per method** (✓, the only one in the figures): per benchmark, the method's "
                 "configuration with the highest mean certified per list on these validation targets; ties: the "
                 "larger library, then the lower cost. Only fully certified configurations compete. The test pass "
                 "uses the configurations chosen here (`chosen.csv`) and never re-chooses on test results.", ""]
        names = [c.name for c in self.spec["cells"]] + [b["name"] for b in self.baselines()]
        methods = self.methods()
        for bench in self.spec["benchmarks"].values():
            b = runs[runs.bench == bench.name] if not runs.empty else runs
            if b.empty:
                continue
            n = len(held_out(bench))
            lines += [f"## {bench.label or bench.name}", "",
                      f"Targets {bench.targets}: {n} held out (each list's own training instances are left out).", "",
                      "| method | configuration | figure | runs | certified | every target | best list | library "
                      "| ties open | sets solved | $ / set |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            for name in sorted((x for x in names if (b.cell == x).any()), key=lambda x: list(methods.values())
                               .index(methods[x])):
                g = b[b.cell == name]
                unchecked = int((~g.checked.fillna(False).astype(bool)).sum())
                ours = not bool(g.baseline.iloc[0])
                solved = span(g.solved / g.sets, ".0%") if ours else "–"
                mark = "✓" if best.get((bench.name, methods[name])) == name else ""
                lines.append(f"| {methods[name]} | {'**' + name + '**' if ours else name}"
                             f"{f' ({unchecked} unchecked)' if unchecked else ''} | {mark} | "
                             f"{len(g) if ours else '–'} | {span(g.certified, '.2f')} | "
                             f"{span(g.certified_all, '.2f')} | {span(g.certified_best, '.0f')} | "
                             f"{span(g.library, '.1f')} | {span(g.certified_open, '.2f')} | {solved} | "
                             f"{span(g.dollars, '.3f') if ours and 'dollars' in g else '–'} |")
            if bench.groups:
                lines += ["", "Per group (certified per rule list on its own group):", "",
                          "| configuration | " + " | ".join(f"{k} (of {len(parse_ids(v))})" for k, v in bench.groups.items())
                          + " |", "|---|" + "---|" * len(bench.groups)]
                for name in names:
                    g = b[b.cell == name]
                    if g.empty:
                        continue
                    cells = [pd.Series([d.get(k) for d in g.by_group.dropna()], dtype=float).mean()
                             for k in bench.groups]
                    lines.append(f"| {name} | " + " | ".join("–" if pd.isna(c) else f"{c:.1f}" for c in cells) + " |")
            figure = self.figure(runs, best, bench)
            if figure:
                lines += ["", f"![{bench.label or bench.name}]({figure.name})"]
            if (self.spec.get("horizon_figure") or {}).get("benchmark") == bench.name:
                horizon = self.horizon_figure(best)
                if horizon:
                    lines += ["", self.horizon_note(), "", f"![Worst-case crash probability per horizon]({horizon.name})"]
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------ figures

    def figure(self, runs: pd.DataFrame, best: Dict[Tuple[str, str], str], bench: Benchmark) -> Optional[Path]:
        """One bar per method (its best configuration, named in small type under the bar): certified per rule list,
        mean over runs with the range as a whisker; the library as a diamond. With groups (the families), one panel
        for the mean over groups and one per group, each out of a group's size."""
        b = runs[runs.bench == bench.name]
        entries = [(m, best[(bench.name, m)]) for m in dict.fromkeys(self.methods().values())
                   if (bench.name, m) in best]
        if not entries:
            return None
        plt = _pyplot()
        from theme import BASELINE, GRID, INK, INK_2, OURS, SURFACE, style
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        groups = list(bench.groups)
        configs = {m: set(b[b.method == m].cell) for m, _ in entries}       # configurations of each method here
        size = max((len(parse_ids(t)) for t in bench.groups.values()), default=len(held_out(bench)))
        panels = [None] + groups
        fig, axes = plt.subplots(1, len(panels), figsize=(max(8.0, 4.6 + 3.3 * (len(panels) - 1)), 0.62 * len(entries) + (1.9 if groups else 2.4)),
                                 sharey=True, squeeze=False, facecolor=SURFACE)
        for k, (ax, group) in enumerate(zip(axes[0], panels)):
            top = size if group is None else len(parse_ids(bench.groups[group]))
            y = np.arange(len(entries))[::-1]
            for yi, (method, name) in zip(y, entries):
                g = b[b.cell == name]
                if group is None:
                    values = g.certified.dropna().tolist()
                    library = (g.library / max(len(groups), 1)).mean()
                else:
                    values = [d.get(group) for d in g.by_group.dropna()]
                    library = np.mean([d.get(group, np.nan) for d in g.library_by_group.dropna()])
                values = [v for v in values if v is not None and not np.isnan(v)]
                if not values:
                    continue
                mean, ours = float(np.mean(values)), not bool(g.baseline.iloc[0])
                ax.barh(yi, mean, height=0.5, color=OURS if ours else BASELINE, zorder=2)
                if min(values) != max(values):
                    ax.plot([min(values), max(values)], [yi, yi], color=INK, linewidth=1.2, zorder=3)
                    for v in (min(values), max(values)):
                        ax.plot([v, v], [yi - 0.11, yi + 0.11], color=INK, linewidth=1.2, zorder=3)
                if not np.isnan(library):
                    ax.scatter([library], [yi], marker="D", s=34, facecolor=SURFACE, edgecolor=INK, linewidth=1.3,
                               zorder=4)
                right = max(mean, max(values))           # the value next to its bar, past the diamond if they meet
                if not np.isnan(library) and right - top * 0.04 <= library <= right + top * 0.16:
                    right = library + top * 0.02
                ax.text(right + top * 0.03, yi, f"{mean:.2f}".rstrip("0").rstrip("."), va="center",
                        fontsize=8.5, color=INK, fontweight="bold" if ours else "normal")
                if k == 0 and len(configs[method]) > 1:
                    ax.text(0, yi - 0.41, name, fontsize=6.5, color=INK_2, va="center")
            style(ax, "per family (mean)" if group is None and groups else (group or ""))
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            ax.spines["bottom"].set_visible(False)
            ax.set_xlim(0, top * 1.15)
            ax.set_xticks(range(0, top + 1, 1 if top <= 6 else 2 if top <= 10 else 5))
            ax.set_ylim(-0.75, len(entries) - 0.45)
            ax.set_xlabel(f"held-out targets certified (of {top}{' per family' if group is None and groups else ''})",
                          fontsize=8, color=INK_2)
        axes[0][0].set_yticks(np.arange(len(entries))[::-1])
        axes[0][0].set_yticklabels([m for m, _ in entries], fontsize=8.5, color=INK)
        fig.suptitle(f"{short_label(bench)}\nbest configuration of each method (validation, PRISM-certified)",
                     x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
        handles = [Patch(color=OURS, label="ours: certified per rule list (mean over runs; whisker: range)"),
                   Patch(color=BASELINE, label="baselines: certified per rule list"),
                   Line2D([], [], marker="D", linestyle="none", markerfacecolor=SURFACE, markeredgecolor=INK,
                          label="library: certified by at least one of the lists"
                                + (" (per family)" if groups else ""))]
        fig.legend(handles=handles, loc="lower left", ncol=3 if groups else 1, fontsize=7.5, frameon=False, labelcolor=INK_2)
        fig.tight_layout(rect=(0, 0.06 if groups else 0.17, 1, 0.95))
        path = self.out / f"rq1_{bench.name}.png"
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)
        return path

    def _horizon_lists(self, bench: Benchmark, name: str) -> List[pd.DataFrame]:
        """The transfer CSV rows of one configuration's lists on `bench` (as written): a run's lists, or a
        baseline's sources."""
        for u in self.units:
            if u.cell.name == name and u.bench.name == bench.name:
                frames = [pd.read_csv(p) for p in (csv_path(r, u.dataset(), "final") for r in u.runs) if p.exists()]
                return [df[df.source == s] for df in frames for s in df.source.unique()]
        for base in self.baselines():
            if base["name"] == name and bench.name in base["config"]:
                path = self._csv(base["config"][bench.name])
                if not path.exists():
                    return []
                df = pd.read_csv(path)
                sources = self._sources(base["sources"][bench.name])["sources"]
                return [df[df.source.astype(str) == s] for s in sources if (df.source.astype(str) == s).any()]
        return []

    def horizon_note(self) -> str:
        spec = self.spec["horizon_figure"]
        published = spec.get("published") or {}
        beyond = {h: v for h, v in (published.get("points") or {}).items() if h > spec["max_horizon"]}
        return (f"Worst-case {spec['value']} of each method's frozen lists (best configuration) per horizon, up to "
                f"{spec['max_horizon']}: {spec.get('why_max', '')} Line: mean over the lists, band: their range; hollow "
                f"markers: a list's own training horizons. {published.get('label', 'Published')}: "
                + ", ".join(f"{v} at {h}" for h, v in (published.get("points") or {}).items())
                + (f" ({', '.join(str(h) for h in beyond)} beyond the axis)." if beyond else "."))

    def horizon_figure(self, best: Dict[Tuple[str, str], str]) -> Optional[Path]:
        """`horizon_figure` in the experiment file: one line per method (its best configuration on the benchmark),
        the worst case of `value` against the horizon, with the optimum, the threshold and published points. Sealed
        targets and horizons beyond `max_horizon` are left out."""
        spec = self.spec["horizon_figure"]
        bench = self.spec["benchmarks"][spec["benchmark"]]
        cfg = load_config(bench.base)
        dataset = bench.dataset or cfg.domain.dataset
        horizons = pd.read_csv(DOMAINS_PATH / cfg.domain.name / "data" / dataset)[spec["horizon"]]
        sealed = load_file(DEFAULT_CONFIG, TransferConfig).sealed.get(Path(dataset).name)
        hidden = set(horizons.index) if sealed == "*" else set(parse_ids(sealed)) if sealed else set()
        shown = {int(i): int(h) for i, h in horizons.items() if i not in hidden and h <= spec["max_horizon"]}
        column = f"worst_{spec['value']}"
        lines = []
        for method in dict.fromkeys(self.methods().values()):
            name = best.get((bench.name, method))
            lists = [df[df.target.isin(shown)] for df in self._horizon_lists(bench, name)] if name else []
            lists = [df for df in lists if column in df and not df.empty]
            if lists:
                lines.append((method, name, lists))
        if not lines:
            return None
        plt = _pyplot()
        from theme import BASELINE, GRID, INK, INK_2, METHOD_COLORS, SURFACE, style

        fig, ax = plt.subplots(figsize=(8.6, 4.6), facecolor=SURFACE)
        published = spec.get("published") or {}
        points = {h: v for h, v in (published.get("points") or {}).items() if h <= spec["max_horizon"]}
        xs_all = sorted(set(shown.values()) | set(points))
        x_end = xs_all[-1]
        ax.axhline(spec["optimum"], color=GRID, linewidth=2.5, zorder=1)
        ax.text(x_end + 0.6, spec["optimum"] - 0.004, f"optimum {spec['optimum']}", fontsize=8, color=INK_2, va="top")
        ax.axhline(spec["threshold"], color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.text(x_end + 0.6, spec["threshold"] + 0.004, f"threshold {spec['threshold']}", fontsize=8, color=INK_2,
                va="bottom")
        spare = iter(c for c in ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7")
                     if c not in METHOD_COLORS.values())
        ends = []
        for method, name, lists in lines:
            color = METHOD_COLORS.get(method) or next(spare, INK)
            table = pd.concat([df.assign(list=i) for i, df in enumerate(lists)])
            table = table.assign(horizon=table.target.map(shown))
            per = table.groupby("horizon")[column].agg(["mean", "min", "max"])
            ax.fill_between(per.index, per["min"], per["max"], color=color, alpha=0.15, linewidth=0, zorder=2)
            several = sum(m == method for m in self.methods().values()) > 1
            ax.plot(per.index, per["mean"], color=color, linewidth=2, zorder=3,
                    label=method + (f" ({name})" if several else ""))
            trained = set(table[table.trained.astype(bool)].horizon)
            for h, v in per["mean"].items():
                ax.plot([h], [v], marker="o", markersize=6, zorder=4, markeredgewidth=1.5,
                        markeredgecolor=color if h in trained else SURFACE,
                        markerfacecolor=SURFACE if h in trained else color)
            ends.append([per["mean"].iloc[-1], per.index[-1], method])
        if points:
            ax.plot(list(points), list(points.values()), linestyle="none", marker="D", markersize=8, color=BASELINE,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4, label=published.get("label", "published"))
        ends.sort()
        for k in range(1, len(ends)):                # direct labels, kept apart
            ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.022)
        for v, x, method in ends:
            ax.text(x_end + 0.6, v, method, fontsize=8, color=INK, va="center")
        ax.set_xticks(xs_all)
        ax.set_xlim(xs_all[0] - 0.8, x_end + 7.5)
        ax.set_ylim(spec["optimum"] - 0.03, None)
        ax.set_xlabel("horizon (moves); hollow markers: the lists' training horizons", fontsize=8.5, color=INK_2)
        style(ax, f"{short_label(bench)}: worst-case crash probability per horizon (best configuration of each method)",
              "worst-case crash probability (lower is better)")
        ax.legend(loc="upper left", fontsize=7.5, frameon=False, labelcolor=INK_2)
        fig.tight_layout()
        path = self.out / f"rq1_{bench.name}_horizon.png"
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        plt.close(fig)
        return path
