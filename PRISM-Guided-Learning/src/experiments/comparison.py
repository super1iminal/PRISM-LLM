"""RQ1: how our best configurations compare with the baselines, per benchmark. For our cells, the numbers are means
over runs (range in parentheses); a baseline is one fit (src/run_baseline.py), its lists the run's lists. Columns:
held-out targets certified per rule list (on its own group, and on every target), the best list, the library (targets
certified by at least one list), certified with ties left open, training sets solved and cost.
"""
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd

from config import load_file
from experiments.base import Experiment, span
from experiments.metrics import transfer_numbers
from experiments.store import CORE_PATH
from run_baseline import BASELINES_PATH, BaselineConfig
from settings import PROJECT_ROOT
from transfer import parse_ids


class Comparison(Experiment):
    root = CORE_PATH

    def baselines(self) -> List[Dict[str, Any]]:
        return self.spec.get("baselines") or []

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
        """Fit and certify each baseline whose CSV misses targets (as written, and with ties open where it has ties)."""
        jobs = []
        for base in self.baselines():
            for bench_name, config in base["config"].items():
                targets = parse_ids(self.spec["benchmarks"][bench_name].targets)
                for rules in (("final", "open") if base.get("ties", True) else ("final",)):
                    path = self._csv(config, rules)
                    if path.exists() and set(targets) <= set(pd.read_csv(path).target):
                        continue
                    log = path.parent / f"run_baseline{'_open' if rules == 'open' else ''}.log"

                    def job(workers, config=config, rules=rules, log=log):
                        import sys
                        ok = self._call([sys.executable, "src/run_baseline.py", config, "--rules", rules,
                                         "--set", f"workers={workers}"], log)
                        return f"[baseline {Path(config).stem} {rules}] {'done' if ok else f'failed (see {log})'}"
                    jobs.append(job)
        return jobs

    def collect(self) -> pd.DataFrame:
        runs = super().collect()
        rows = []
        for base in self.baselines():
            for bench_name, config in base["config"].items():
                bench = self.spec["benchmarks"][bench_name]
                targets = parse_ids(bench.targets)
                src = self._sources(base["sources"][bench_name])
                final = transfer_numbers(self._csv(config), targets, bench.groups, src["source_groups"],
                                         src["sources"])
                opened = transfer_numbers(self._csv(config, "open"), targets, bench.groups, src["source_groups"],
                                          src["sources"]) if base.get("ties", True) else final
                rows.append({"bench": bench_name, "cell": base["name"], "code": base.get("code", base["name"]),
                             "dir": f"baselines/{Path(config).stem}", "baseline": True, **final,
                             **{f"{k}_open": opened.get(k) for k in ("checked", "certified", "certified_all",
                                                                       "library", "by_group")}})
        return pd.concat([runs.assign(baseline=False), pd.DataFrame(rows)], ignore_index=True)

    def report(self, runs: pd.DataFrame) -> str:
        lines = [f"# {self.title}", "", self.spec.get("question", "").strip(), "",
                 "Certified: held-out targets certified per rule list (on its own group's targets where there are "
                 "groups). Every target: the same over all targets, whatever the group. Library: targets certified by "
                 "at least one of a run's (or baseline's) lists. Ties open: certified with the action orders left "
                 "open, i.e. however ties are broken. Ours: mean over runs, range in parentheses.", ""]
        names = [c.name for c in self.spec["cells"]] + [b["name"] for b in self.baselines()]
        for bench in self.spec["benchmarks"].values():
            b = runs[runs.bench == bench.name]
            if b.empty:
                continue
            n = len(parse_ids(bench.targets))
            lines += [f"## {bench.label or bench.name}", "", f"Targets {bench.targets} ({n}); each list's own training instances are left out.", "",
                      "| method | runs | certified | every target | best list | library | ties open | sets solved "
                      "| $ / set |", "|---|---|---|---|---|---|---|---|---|"]
            for name in names:
                g = b[b.cell == name]
                if g.empty:
                    continue
                unchecked = int((~g.checked.fillna(False).astype(bool)).sum())
                ours = not bool(g.baseline.iloc[0])
                solved = span(g.solved / g.sets, ".0%") if ours else "–"
                lines.append(f"| {'**' + name + '**' if ours else name}{f' ({unchecked} unchecked)' if unchecked else ''}"
                             f" | {len(g) if ours else '–'} | {span(g.certified, '.2f')} | "
                             f"{span(g.certified_all, '.2f')} | {span(g.certified_best, '.0f')} | "
                             f"{span(g.library, '.1f')} | {span(g.certified_open, '.2f')} | {solved} | "
                             f"{span(g.dollars, '.3f') if ours else '–'} |")
            if bench.groups:
                lines += ["", "Per group (certified per rule list on its own group):", "",
                          "| method | " + " | ".join(f"{k} (of {len(parse_ids(v))})" for k, v in bench.groups.items())
                          + " |", "|---|" + "---|" * len(bench.groups)]
                for name in names:
                    g = b[b.cell == name]
                    if g.empty:
                        continue
                    cells = [pd.Series([d.get(k) for d in g.by_group.dropna()], dtype=float).mean()
                             for k in bench.groups]
                    lines.append(f"| {name} | " + " | ".join("–" if pd.isna(c) else f"{c:.1f}" for c in cells) + " |")
            lines.append("")
        return "\n".join(lines)
