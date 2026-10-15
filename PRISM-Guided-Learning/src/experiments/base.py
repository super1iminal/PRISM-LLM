"""An experiment: cells (configurations) on benchmarks, each with a number of repeats, evaluated by certifying the
frozen rules of every run on the benchmark's held-out targets. The two research questions are two kinds of
experiment over one store of runs (docs/experiments.md):

  * comparison (RQ1, experiments/comparison.py): our best configurations against the baselines;
  * ablation (RQ2, experiments/ablation.py): one factor at a time against a reference configuration.

A cell is a condition (configs/run/conditions/) plus overrides. Its runs are those in the run store with the same
identity (experiments/store.py), whatever condition name made them, so a configuration used by several experiments
runs once. Missing repeats are launched under <root>/<condition>[+<cell slug>]/seed_none_<k>/ (root: core/ for
comparison experiments, ablations/ for ablations). Each run's rules are certified twice by src/transfer.py: as
written (`final`) and with their action orders left open (`open`: certified however ties are broken).
"""
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import yaml

from config import CONFIG_DIR, Config, load_config
from experiments.metrics import run_numbers, transfer_numbers
from experiments.store import ABLATIONS_PATH, RunStore, fingerprint
from settings import PROJECT_ROOT, RESULTS_PATH
from transfer import csv_path, parse_ids

EXPERIMENTS_DIR = CONFIG_DIR / "experiments"
REPORTS_PATH = RESULTS_PATH / "experiments"
PYTHON = sys.executable
RULES = ("final", "open")


@dataclass
class Benchmark:
    name: str
    base: str                                   # the condition cells start from
    targets: str                                # held-out target ids, e.g. "12-19"
    label: str = ""
    dataset: Optional[str] = None               # target dataset (default: the runs' own)
    groups: Dict[str, str] = field(default_factory=dict)          # group -> its target ids (e.g. families)
    source_groups: Dict[str, str] = field(default_factory=dict)   # rule list (sample id) -> its group


@dataclass
class Cell:
    name: str                                   # descriptive name, used in prose and tables
    code: str                                   # short label for figures
    set: Dict[str, List[str]]                   # benchmark -> run-config overrides
    condition: Dict[str, str]                   # benchmark -> condition to start from instead of the base
    only: List[str]                             # benchmarks the cell runs on
    repeats: int
    launch: bool = True                         # false: only use existing runs (e.g. Sonnet, which needs approval)
    note: str = ""
    method: str = ""                            # comparisons: the method this configuration belongs to


@dataclass
class Unit:
    """One cell on one benchmark: its resolved config, its runs, and where new runs go."""
    cell: Cell
    bench: Benchmark
    condition: str
    overrides: List[str]
    cfg: Config
    run_name: str
    root: Path
    runs: List[Path]
    pending: List[Path]
    default_name: str = ""                      # the run name the cell would get without existing runs

    @property
    def missing(self) -> int:
        return max(self.cell.repeats - len(self.runs) - len(self.pending), 0) if self.cell.launch else 0

    def dataset(self) -> str:
        return self.bench.dataset or self.cfg.domain.dataset


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _per_bench(value, benches: List[str]) -> Dict[str, Any]:
    """A YAML value given once for every benchmark, or as a mapping benchmark -> value (only those it names)."""
    if isinstance(value, dict):
        unknown = set(value) - set(benches)
        if unknown:
            raise ValueError(f"unknown benchmarks {sorted(unknown)} (the experiment has {', '.join(benches)})")
        return dict(value)
    return {b: value for b in benches}


def _cell_benches(spec: dict, benches: List[str]) -> List[str]:
    """The benchmarks a cell runs on: `only` (default: all), and, when its `set` or `condition` is given per
    benchmark, only those the mapping names. A benchmark the mapping leaves out is skipped, never run as the base
    configuration; naming it in `only` is an error."""
    named = set(benches)
    for key in ("set", "condition"):
        if isinstance(spec.get(key), dict) and spec[key]:
            named &= set(_per_bench(spec[key], benches))
    if spec.get("only"):
        _per_bench(dict.fromkeys(spec["only"]), benches)            # unknown benchmarks
        missing = [b for b in spec["only"] if b not in named]
        if missing:
            raise ValueError(f"cell {spec['name']!r}: `only` names {missing}, which its per-benchmark `set` or "
                             "`condition` leaves out")
        return list(spec["only"])
    return [b for b in benches if b in named]


def benchmark_specs(data: dict) -> Dict[str, dict]:
    """An experiment's benchmarks: a mapping, or the name of a set in configs/experiments/benchmarks.yaml."""
    if isinstance(data["benchmarks"], str):
        shared = yaml.safe_load((EXPERIMENTS_DIR / "benchmarks.yaml").read_text(encoding="utf-8"))
        return shared[data["benchmarks"]]
    return data["benchmarks"]


def load_experiment(path: Path) -> dict:
    """The experiment file as plain data, with benchmarks and cells parsed (docs/experiments.md)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    benches = {name: Benchmark(name=name, **spec) for name, spec in benchmark_specs(data).items()}
    names = list(benches)
    cells = []
    for spec in data["cells"]:
        sets = _per_bench(spec.get("set") or [], names)
        cells.append(Cell(name=spec["name"], code=spec.get("code", spec["name"]),
                          set={b: list(sets.get(b) or []) for b in names},
                          condition=dict(spec.get("condition") or {}), only=_cell_benches(spec, names),
                          repeats=spec.get("repeats", data.get("repeats", 3)), launch=spec.get("launch", True),
                          note=spec.get("note", ""), method=spec.get("method", "")))
    return {**data, "benchmarks": benches, "cells": cells, "path": Path(path)}


class Experiment:
    """Plans, launches and evaluates an experiment file's runs; subclasses write the report."""
    root: Path = ABLATIONS_PATH

    def __init__(self, path: Path, store: Optional[RunStore] = None):
        self.spec = load_experiment(path)
        self.name = Path(path).stem
        self.title = self.spec.get("title", self.name)
        self.prices = self.spec.get("prices") or {}
        self.store = store or RunStore()
        self.out = REPORTS_PATH / self.name
        self.units = self.resolve()

    # ------------------------------------------------------------ planning

    def resolve(self) -> List[Unit]:
        units = []
        for cell in self.spec["cells"]:
            for b in cell.only:
                bench = self.spec["benchmarks"][b]
                condition = cell.condition.get(b, bench.base)
                overrides = cell.set[b]
                cfg = load_config(condition, overrides)
                runs, pending = self.store.runs(cfg), self.store.pending(cfg)
                default = condition + (f"+{slug(cell.name)}" if overrides else "")
                if runs or pending:          # more repeats go next to the existing ones
                    first = (runs or pending)[0]
                    root, run_name = first.parents[1], first.parent.name
                else:
                    root, run_name = self.root, default
                units.append(Unit(cell, bench, condition, overrides, cfg, run_name, root, runs[:cell.repeats],
                                  pending, default))
        return units

    def plan(self) -> List[str]:
        """One line per unit: its runs (and the names they were made under), pending and missing repeats."""
        lines = []
        for u in self.units:
            names = sorted({r.parent.name for r in u.runs})
            shared = f" (runs of {', '.join(names)})" if names and names != [u.default_name] else ""
            lines.append(f"{u.bench.name:>10} | {u.cell.name}: {len(u.runs)}/{u.cell.repeats} runs{shared}"
                         + (f", {len(u.pending)} pending" if u.pending else "")
                         + (f", {u.missing} to launch -> {u.root.name}/{u.run_name}" if u.missing else "")
                         + ("" if u.cell.launch else ", launch: false"))
        return lines

    # ------------------------------------------------------------ running

    def _new_dirs(self, unit: Unit) -> List[Path]:
        taken = {p.name for p in (unit.root / unit.run_name).glob("seed_none_*")}
        out, k = [], 1
        while len(out) < unit.missing:
            if f"seed_none_{k}" not in taken:
                out.append(unit.root / unit.run_name / f"seed_none_{k}")
            k += 1
        return out

    @staticmethod
    def _call(cmd: List[str], log: Path) -> bool:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as out:
            return subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=out, stderr=subprocess.STDOUT).returncode == 0

    def check(self, run_dir: Path, unit: Unit, workers: int) -> str:
        """Certify a run's rules on the unit's targets, as written and with ties left open (skipped when done)."""
        log = run_dir.parent / f"{run_dir.name}.log"
        targets = parse_ids(unit.bench.targets)
        for rules in RULES:
            if transfer_done(run_dir, unit.dataset(), targets, rules):
                continue
            cmd = [PYTHON, "src/transfer.py", str(run_dir), "--targets", unit.bench.targets, "--rules", rules,
                   "--workers", str(workers)] + (["--data", unit.bench.dataset] if unit.bench.dataset else [])
            if not self._call(cmd, log):
                return f"[{run_dir.parent.name}/{run_dir.name}] {rules} transfer failed (see {log})"
        return f"[{run_dir.parent.name}/{run_dir.name}] checked"

    def launch(self, run_dir: Path, unit: Unit, workers: int) -> str:
        """Run one repeat (src/run_ablation.py), then check it."""
        k = int(run_dir.name.rsplit("_", 1)[1])
        cmd = [PYTHON, "src/run_ablation.py", unit.condition, "--unseeded", str(k), "--repeat", str(k),
               "--name", unit.run_name, "--out-root", str(unit.root)]
        cmd += [x for s in unit.overrides for x in ("--set", s)]
        if not self._call(cmd, run_dir.parent / f"{run_dir.name}.log"):
            return f"[{unit.run_name}/{run_dir.name}] run failed"
        return self.check(run_dir, unit, workers)

    def extra_jobs(self) -> List[Callable[[int], str]]:
        """Further jobs of the experiment kind (comparison: the baselines)."""
        return []

    # ------------------------------------------------------------ evaluation

    def collect(self) -> pd.DataFrame:
        """One row per run of every unit: training numbers, and transfer numbers as written and with ties open."""
        rows = []
        for u in self.units:
            targets = parse_ids(u.bench.targets)
            for r in u.runs:
                final = transfer_numbers(csv_path(r, u.dataset(), "final"), targets, u.bench.groups,
                                         u.bench.source_groups)
                opened = transfer_numbers(csv_path(r, u.dataset(), "open"), targets, u.bench.groups,
                                          u.bench.source_groups)
                rows.append({"bench": u.bench.name, "cell": u.cell.name, "code": u.cell.code,
                             "dir": f"{r.parent.name}/{r.name}", **run_numbers(r, self.prices), **final,
                             **{f"{k}_open": opened.get(k) for k in ("checked", "certified", "certified_all",
                                                                       "library", "by_group")}})
        return pd.DataFrame(rows)

    def report(self, runs: pd.DataFrame) -> str:
        raise NotImplementedError

    def write(self) -> Path:
        runs = self.collect()
        self.out.mkdir(parents=True, exist_ok=True)
        if not runs.empty:
            runs.drop(columns=[c for c in runs if c.startswith("by_") or c.endswith(("by_group", "by_group_open"))
                               or c == "per_source"]).to_csv(self.out / "runs.csv", index=False)
        page = self.report(runs)
        path = self.out / "REPORT.md"
        path.write_text(page, encoding="utf-8")
        return path


def jobs(experiments: List["Experiment"], run: bool, check: bool, baselines: bool) -> List[Callable[[int], str]]:
    """The jobs of several experiments, each taking the number of transfer workers, in the order they should start:
    launches of missing repeats (`run`; the longest jobs, each also checks its run), then checks of existing runs
    (with `check` or `run`), then baseline fits (`baselines`). A configuration that several experiments need is
    launched once, for the largest number of missing repeats."""
    checks, extra, seen, launches = [], [], set(), {}
    for exp in experiments:
        for u in exp.units:
            for r in u.runs if (check or run) else []:
                if (r, u.bench.targets) not in seen:
                    seen.add((r, u.bench.targets))
                    checks.append(lambda w, e=exp, r=r, u=u: e.check(r, u, w))
            key = fingerprint(u.cfg)
            if run and u.missing and (key not in launches or u.missing > launches[key][1].missing):
                launches[key] = (exp, u)
        extra += exp.extra_jobs() if baselines else []
    out = [lambda w, e=exp, d=d, u=u: e.launch(d, u, w) for exp, u in launches.values() for d in exp._new_dirs(u)]
    return out + checks + extra


def held_out(bench: Benchmark) -> List[int]:
    """The benchmark's targets a rule list of its base condition is certified on: without the training instances
    when they are among the targets (Pac-Man's training horizons are in `pacman_transfer.csv`)."""
    cfg = load_config(bench.base)
    targets = parse_ids(bench.targets)
    if bench.dataset and Path(bench.dataset).name != Path(cfg.domain.dataset).name:
        return targets
    trained = {i for s in cfg.run.train_sets for i in s}
    return [t for t in targets if t not in trained]


def transfer_done(run_dir: Path, dataset: str, targets: List[int], rules: str = "final") -> bool:
    """Whether the run's transfer CSV for `dataset` (and `rules`) already has every target in `targets`."""
    path = csv_path(run_dir, dataset, rules)
    return path.exists() and set(targets) <= set(pd.read_csv(path).target)


def span(values, fmt: str) -> str:
    """Mean, and the range in parentheses when the values differ."""
    values = pd.Series(values, dtype=float).dropna()
    if values.empty:
        return "–"
    mean = format(values.mean(), fmt)
    return mean if len(values) == 1 or values.min() == values.max() else \
        f"{mean} ({format(values.min(), fmt)}–{format(values.max(), fmt)})"


def core_names() -> List[str]:
    """Run names (out/results/core/<name>/) of the comparison experiments' cells: the method's main runs."""
    names = []
    for path in sorted(EXPERIMENTS_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data.get("kind") != "comparison":
            continue
        specs = benchmark_specs(data)
        for spec in data["cells"]:
            benches = list(specs)
            sets = _per_bench(spec.get("set") or [], benches)
            for b in _cell_benches(spec, benches):
                condition = (spec.get("condition") or {}).get(b, specs[b]["base"])
                names.append(condition + (f"+{slug(spec['name'])}" if sets.get(b) else ""))
    return sorted(set(names))


def experiment(path: Path, store: Optional[RunStore] = None) -> Experiment:
    """The experiment in `path`, as its kind's class."""
    from experiments.ablation import Ablation
    from experiments.comparison import Comparison
    kind = yaml.safe_load(Path(path).read_text(encoding="utf-8")).get("kind")
    classes = {"comparison": Comparison, "ablation": Ablation}
    if kind not in classes:
        raise ValueError(f"{path}: kind must be one of {', '.join(classes)}, not {kind!r}")
    return classes[kind](path, store)

