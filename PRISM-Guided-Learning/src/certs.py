"""One certification verdict for every (controller, instance) pair the paper and Rajan's report use: out/paper/certs.csv.

The rule (Oct 21):
  * interval iteration in PRISM at 1e-9; a check certifies iff the less favourable bound meets every threshold;
  * where interval iteration does not converge, or its interval contains a threshold: Storm `--sound`, then
    `--exact` where that fails (Docker image movesrwth/storm:stable);
  * no linear programming, no Gauss-Seidel; a timeout counts as not certified and is recorded as a timeout.
Existing interval-iteration verdicts are reused where it converged; only checks that linear programming or
Gauss-Seidel decided (PRISM 4.10.1 runs Gauss-Seidel for `-lp`), or that were inconclusive, are re-run, plus checks no
file has yet (the training split where a file lacks it, and the other LLMs' lists on the test instances).

Gridworld requirements are LTL (nested until, conjunctions), which Storm's PRISM front end and PRISM's exact engine
do not take as such: each requirement is checked on its monitor product (a deterministic monitor obtained by formula
progression, multiplied into PRISM's exported model of the check) as a reachability query. Other domains' properties
go to Storm as they are.

Stages (from PRISM-Guided-Learning/; WORK is a scratch directory):
    python src/certs.py inventory WORK        every check in scope and its existing result -> WORK/inventory.csv
    python src/certs.py prism WORK            interval iteration for checks without one -> WORK/prism.csv
    python src/certs.py storm-prep WORK       Storm inputs for every check that needs Storm -> WORK/storm/
    python src/certs.py storm WORK            run Storm (--sound, then --exact) in Docker
    python src/certs.py collect WORK          out/paper/certs.csv and WORK/report.md
"""
import argparse, json, re, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from baselines import load_method
from config import load_config, load_file, load_run_config
from core.domain import load_domain
from core.prism import PrismCall, PrismError, PrismRunner, run_call
from core.rules import RuleError, open_ties
from core.verifier import PolicyVerifier, exact_solvers
from run_baseline import BaselineConfig
from settings import PROJECT_ROOT
from transfer import RULES, parse_ids, training_instances

RESULTS = PROJECT_ROOT / "out" / "results"
CERTS = PROJECT_ROOT / "out" / "paper" / "certs.csv"
II = "interval iteration"
STORM_IMAGE = "movesrwth/storm:stable"
SOUND_TIMEOUT, EXACT_TIMEOUT = 10, 1800
# interval iteration's iteration cap for new checks: past it the check goes to Storm (the cap decides only when Storm
# takes over, never a verdict); the existing checks used 10^6
II_MAX_ITERS = 10 ** 5
LOCK = threading.Lock()

# condition -> benchmark; LLM by run directory suffix
BENCH = {"port_ts_history": "grids", "port_fam_history": "families", "port_pacman_narrow_core": "pacman",
         "port_mer": "mer", "port_mer_n": "mer_n", "port_mer_one": "mer_one", "port_mer_n_one": "mer_n_one",
         "port_ts_noclose": "grids_noclose"}
LLMS = {"": "Claude Haiku 5.5", "_luna": "GPT-6 Luna", "_gemini": "Gemini 3.8 Flash"}
DATASETS = {"grids": ("grid_20_large", "grid_test_large"), "families": ("families_v0", "families_v0"),
            "pacman": ("pacman_transfer", "pacman_transfer"), "mer": ("mer_transfer", "mer_transfer"),
            "mer_n": ("mer_n_transfer", "mer_n_transfer"), "mer_one": ("mer_transfer", None),
            "mer_n_one": ("mer_n_transfer", None), "grids_noclose": ("grid_20_large", None)}
# validation / test targets of each benchmark (configs/experiments/rq1_comparison.yaml, configs/transfer)
VALIDATION = {"grids": "12-19", "grids_noclose": "12-19", "families": "16-35,76-95,136-155", "pacman": "0-9",
              "mer": "0-8", "mer_n": "0-8", "mer_one": "0-8", "mer_n_one": "0-8"}
TEST = {"grids": "0-41", "families": "36-59,96-119,156-179", "pacman": "10-11", "mer": "9-11", "mer_n": "9-11"}
SUFFIX = {"": "final", "_determinized": "determinized", "_norepair_determinized": "norepair_determinized",
          "_norepair": "norepair", "_open": "open", "_oneshot": "oneshot", "_unnarrowed": "unnarrowed"}
STEMS = ("grid_20_large", "grid_test_large", "families_v0", "pacman_transfer", "mer_transfer", "mer_n_transfer")
COLUMNS = ["benchmark", "split", "controller", "llm", "run", "list_id", "instance", "size", "episode", "verdict",
           "solver", "margin", "binding", "worst", "states", "seconds", "source_file", "old_verdict", "old_solver",
           "old_margin", "old_seconds", "model_seconds"]
# seconds: the check that gave the verdict; old_seconds and model_seconds: the original check as it ran (the
# sequential studies' cost tables report the manager's own times)


# ---------------------------------------------------------------- what is in scope

def parse_file(path: Path) -> Optional[dict]:
    """Benchmark, controller, LLM, run and rule set of a held-out check file (None: out of scope). Files without
    `exact`/`sealed` are the loop solver's checks (T4c reads two): they get a certification here."""
    rel = path.relative_to(RESULTS)
    m = re.match(r"transfer_(?:(exact|sealed)_)?(.+)\.csv$", path.name)
    if not m or "_sonnet" in str(rel):
        return None
    stem, suffix = m.group(2), ""
    for s in STEMS:
        if stem.startswith(s):
            stem, suffix = s, stem[len(s):]
    if suffix not in SUFFIX:
        return None
    test = m.group(1) == "sealed"
    top, name = rel.parts[0], rel.parts[1]
    if top == "baselines":
        cfg = PROJECT_ROOT / "configs" / "baselines" / f"{name}.yaml"
        b = ("pacman" if name.startswith("pacman") else "mer_n" if name.startswith("mer_n") else
             "mer" if name.startswith("mer") else "families" if "families" in name else
             "grids" if "grid" in name else None)
        kind = ("DT-raw" if "tree_raw" in name else "DT-voc" if "tree_features" in name else
                "RL" if "ppo" in name else "Hand" if ("template" in name or "maze_rules" in name) else None)
        if b is None or kind is None:
            return None
        seed = re.search(r"_seed(\d)$", name)
        return {"benchmark": b, "controller": kind + (" (permissive)" if suffix == "_open" else ""),
                "llm": "", "run": f"seed{seed.group(1) if seed else 0}", "test": test, "rules": SUFFIX[suffix],
                "kind": "baseline", "config": str(cfg.relative_to(PROJECT_ROOT)), "stem": stem}
    if top not in ("core", "ablations"):
        return None
    llm_suffix = next((s for s in ("_luna", "_gemini") if s in name), "")
    base = name.replace(llm_suffix, "")
    restart = base.endswith("_resample")
    base = base.replace("_resample", "")
    bench = BENCH.get(base)
    if bench is None:
        return None
    rules = SUFFIX[suffix]
    frozen = "final" if bench == "pacman" else "determinized"
    controller = {"final": "Full (frozen)" if bench == "pacman" else "Full (permissive)",
                  "determinized": "Full (frozen)", "norepair_determinized": "NoRepair", "norepair": "NoRepair (permissive)",
                  "oneshot": "OneShot", "unnarrowed": "Full without narrowing", "open": "Full (open)"}[rules]
    if restart:
        if rules != frozen:
            return None
        controller = "Restart"
    if bench == "grids_noclose":
        controller = "NoSP (Full)"
    return {"benchmark": bench, "controller": controller, "llm": LLMS[llm_suffix], "run": f"{rel.parts[0]}/{name}/{rel.parts[2]}",
            "test": test, "rules": rules, "kind": "run", "stem": stem}


def instance_sizes() -> Dict[str, pd.DataFrame]:
    out = {}
    for domain, stems in (("gridworld", ("grid_20_large", "grid_test_large", "families_v0", "sequential_v0")),
                          ("pacman", ("pacman_transfer",)), ("mer", ("mer_transfer", "mer_n_transfer"))):
        for s in stems:
            out[s] = pd.read_csv(PROJECT_ROOT / "domains" / domain / "data" / f"{s}.csv")
    return out


def size_of(sizes, stem: str, bench: str, i: int) -> str:
    row = sizes[stem].iloc[int(i)]
    if "max_steps" in row:
        return str(int(row.max_steps))
    if "x" in row and bench in ("mer", "mer_one"):
        return f"x={row.x}"
    return str(int(row.n))


def worst_of(row) -> str:
    return json.dumps({k[6:]: row[k] for k in row.index if k.startswith("worst_") and pd.notna(row[k])})


def old_verdict(outcome) -> str:
    return {"certified": "certified", "failed": "failed", "inconclusive": "inconclusive", "unparsed": "failed",
            "not_certifiable": "timeout", "timeout": "timeout"}.get(outcome, str(outcome))


def inventory(work: Path) -> None:
    """Every check in scope: the check files the paper's and Rajan's tables read (WORK/reads.txt, traced from
    viz/paper_results.py and the merged viz/rajan_results.py), the training split where those files lack it, the other LLMs'
    lists on the test instances, and every check of every sequential study."""
    sizes = instance_sizes()
    allowed = baseline_sources()
    rows = []
    reads = [PROJECT_ROOT / p for p in (work / "reads.txt").read_text().split()]
    files = sorted(p for p in reads if re.match(r"transfer_.*\.csv$", p.name) and parse_file(p))
    for f in files:
        info = parse_file(f)
        bench = info["benchmark"]
        if bench not in VALIDATION:
            continue
        df = pd.read_csv(f)
        if info["kind"] == "baseline":                 # only the sources the RQ1 spec names for the benchmark
            stem_cfg = re.sub(r"_seed\d$", "", Path(info["config"]).stem)    # RL seeds 1-2: seed 0's sources
            # a baseline file can serve several benchmarks (mer and mer_one), each with its own sources
            names = set().union(*[v for (cfg_stem, _), v in allowed.items() if cfg_stem == stem_cfg] or [set()])
            df = df[df.source.astype(str).isin(names)]
        rel = f.relative_to(PROJECT_ROOT).as_posix()
        wanted = set(parse_ids(TEST[bench] if info["test"] else VALIDATION[bench]))
        for _, r in df.iterrows():
            t = int(r.target)
            trained = bool(r.get("trained", False)) if "trained" in df else False
            if not trained and t not in wanted:
                continue
            split = "training" if trained else "test" if info["test"] else "validation"
            solver = r.get("solver") if "solver" in df else None
            outcome = r.get("outcome") if "outcome" in df else None
            if outcome is None or pd.isna(outcome):
                need = "prism"                                  # a loop-solver file: no certification yet
            elif solver == II and outcome in ("certified", "failed"):
                need = "reuse"
            elif outcome in ("not_certifiable", "timeout"):
                need = "reuse"                                  # a timeout stays a timeout (not re-run)
            elif outcome == "unparsed":
                need = "reuse"
            else:
                need = "storm"                                  # LP (= Gauss-Seidel), Gauss-Seidel or inconclusive
            rows.append({**{k: info[k] for k in ("benchmark", "controller", "llm", "run")}, "split": split,
                         "list_id": str(r.source), "instance": t, "size": size_of(sizes, info["stem"], bench, t),
                         "episode": None, "old_verdict": old_verdict(outcome) if outcome is not None else None,
                         "old_solver": solver, "old_margin": r.get("margin"), "worst": worst_of(r),
                         "binding": r.get("binding"),
                         "states": r.get("states"), "seconds": r.get("seconds"), "old_seconds": r.get("seconds"),
                         "model_seconds": r.get("model_seconds"), "source_file": rel, "need": need,
                         "kind": info["kind"], "rules": info["rules"], "config": info.get("config"), "stem": info["stem"]})
    inv = pd.DataFrame(rows)
    # families: a run's list is used on its own family's members only (as the tables count it)
    fam = (inv.kind == "run") & (inv.benchmark == "families") & (inv.split != "training")
    keep = [own_family(r) for _, r in inv[fam].iterrows()]
    inv = pd.concat([inv[~fam], inv[fam][keep]], ignore_index=True)
    rows = []
    # the training split, where a list's file has no training rows: each list on its own training members
    have = set(zip(inv[inv.split == "training"].source_file, inv[inv.split == "training"].list_id))
    for (src, lid), g in inv[(inv.split == "validation")].groupby(["source_file", "list_id"]):
        if (src, lid) in have:
            continue
        r0 = g.iloc[0]
        ids = training_ids(r0)
        for t in ids or []:
            rows.append({**r0.to_dict(), "split": "training", "instance": int(t),
                         "size": size_of(sizes, r0.stem, r0.benchmark, t), "old_verdict": None, "old_solver": None,
                         "old_margin": None, "worst": "{}", "states": None, "seconds": None, "need": "prism"})
    # the other LLMs' lists on the test instances (their runs have validation files only)
    for (src, lid), g in inv[(inv.split == "validation") & inv.llm.isin(["GPT-6 Luna", "Gemini 3.8 Flash"])
                             & inv.controller.isin(["Full (frozen)", "Full (permissive)", "NoRepair"])].groupby(["source_file", "list_id"]):
        r0 = g.iloc[0]
        if r0.benchmark not in TEST:
            continue
        test_stem = DATASETS[r0.benchmark][1]
        test_file = src.replace("transfer_exact_", "transfer_sealed_").replace(r0.stem, test_stem)
        for t in own_targets(r0.benchmark, lid, TEST[r0.benchmark], r0):
            rows.append({**r0.to_dict(), "split": "test", "instance": int(t), "stem": test_stem,
                         "size": size_of(sizes, test_stem, r0.benchmark, t), "old_verdict": None,
                         "old_solver": None, "old_margin": None, "worst": "{}", "states": None, "seconds": None,
                         "source_file": test_file, "need": "prism"})
    inv = pd.concat([inv, pd.DataFrame(rows)], ignore_index=True)
    inv = pd.concat([inv, sequential_inventory(sizes)], ignore_index=True)
    inv.to_csv(work / "inventory.csv", index=False)
    print(inv.groupby(["split", "need"]).size().unstack(fill_value=0).to_string())
    print(inv.groupby(["benchmark", "controller", "llm"]).size().to_string())


def baseline_sources() -> Dict[tuple, set]:
    """(baseline config stem, benchmark) -> the sources configs/experiments/rq1_comparison.yaml names (as
    viz/paper_results.py selects them)."""
    from experiments.base import experiment
    from experiments.comparison import Comparison
    exp = experiment(PROJECT_ROOT / "configs" / "experiments" / "rq1_comparison.yaml")
    out: Dict[tuple, set] = {}
    for b in exp.spec["baselines"]:
        for bench, config in b["config"].items():
            if bench in b.get("sources", {}):
                key = (Path(config).stem, "grids" if bench == "grids_noclose" else bench)
                out.setdefault(key, set()).update(Comparison._sources(b["sources"][bench])["sources"])
    return out


FAMILIES = {"rooms": (0, 59), "crossing": (60, 119), "adjacent": (120, 179)}


def own_family(row) -> bool:
    """Whether `row.instance` is in the family of the members its list was trained on."""
    trained = training_ids(row) or []
    for lo, hi in FAMILIES.values():
        if trained and lo <= trained[0] <= hi:
            return lo <= int(row.instance) <= hi
    return True


def own_targets(bench: str, list_id: str, spec: str, row) -> List[int]:
    """Targets of `spec` a list is checked on: families lists on their own family's members only."""
    ids = parse_ids(spec)
    if bench != "families":
        return ids
    groups = {"rooms": (0, 59), "crossing": (60, 119), "adjacent": (120, 179)}
    trained = training_ids(row) or []
    fam = next((g for g, (lo, hi) in groups.items() if trained and lo <= trained[0] <= hi), None)
    return [i for i in ids if fam is None or groups[fam][0] <= i <= groups[fam][1]]


_train_cache: Dict = {}


def training_ids(row) -> Optional[List[int]]:
    """The instances list `row.list_id` of `row`'s run (or baseline) was trained on, in `row.stem`'s numbering."""
    key = (row.source_file, row.list_id)
    if key in _train_cache:
        return _train_cache[key]
    ids = None
    if row.kind == "run":
        run = PROJECT_ROOT / Path(row.source_file).parent
        try:
            ids = training_instances(run).get(int(row.list_id))
        except (ValueError, FileNotFoundError):
            ids = None
    else:
        settings = load_file(PROJECT_ROOT / row.config, BaselineConfig, [])
        if settings.train and Path(settings.dataset).stem == row.stem and "Hand" not in row.controller:
            k = re.search(r"_set(\d+)$", str(row.list_id))
            if k and (settings.params or {}).get("sets"):     # trees and PPO: source <name>_set<k> was fitted on sets[k]
                ids = settings.params["sets"][int(k.group(1))]
    _train_cache[key] = [int(i) for i in ids] if ids else None
    return _train_cache[key]


def seq_binding(r) -> Optional[str]:
    """The requirement with the smallest margin of a sequential check (its file records worst cases, not the binding
    one). Thresholds: gridworld goals and sequences 0.8, avoidance 0.7 (lower bounds); Pac-Man crash 0.56 (upper)."""
    m = {}
    for k in r.index:
        if k.startswith("worst_") and pd.notna(r[k]):
            name = k[6:]
            m[name] = (0.56 - r[k]) if name == "crash" else r[k] - (0.7 if name.startswith("avoid") else 0.8)
    return min(m, key=m.get) if m else None


def sequential_inventory(sizes) -> pd.DataFrame:
    rows = []
    for checks in sorted((RESULTS / "sequential").glob("*/*/*/checks.csv")):
        study, mode, seq = checks.parts[-4:-1]
        conf = yaml.safe_load((PROJECT_ROOT / "configs" / "sequential" / f"{study}.yaml").read_text())
        stem = Path(conf["dataset"]).stem
        bench = "pacman" if "pacman" in conf["condition"] else "grid worlds"
        df = pd.read_csv(checks)
        for _, r in df.iterrows():
            outcome, solver = r.get("outcome"), r.get("solver")
            need = ("reuse" if (solver == II and outcome in ("certified", "failed")) or outcome in
                    ("timeout", "not_certifiable", "unparsed") else "storm")
            rows.append({"benchmark": bench, "split": "validation", "controller": f"Library ({study}, {mode})",
                         "llm": "Claude Haiku 5.5", "run": f"{study}/{mode}/{seq}", "list_id": str(r.rule_list),
                         "instance": int(r.instance), "size": size_of(sizes, stem, bench, int(r.instance)),
                         "episode": int(r.episode), "old_verdict": old_verdict(outcome), "old_solver": solver,
                         "old_margin": r.get("margin"), "worst": worst_of(r), "states": r.get("states"),
                         "binding": seq_binding(r), "seconds": r.get("seconds"), "old_seconds": r.get("seconds"),
                         "model_seconds": r.get("model_seconds"),
                         "source_file": checks.relative_to(PROJECT_ROOT).as_posix(),
                         "need": need, "kind": "sequential", "rules": None, "config": conf["condition"], "stem": stem})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- rebuilding a check's controller

_ctx: Dict = {}
_ctx_lock: Dict = {}


def run_config(run: Path):
    """The run's configuration; the other LLMs' runs carry keys this branch does not know (llm.openrouter.*), and
    certification reads none of the LLM settings, so they use their Haiku condition's."""
    try:
        cfg = load_run_config(run)
    except Exception:
        base = run.parent.name
        for s in ("_luna", "_gemini"):
            base = base.replace(s, "")
        cfg = load_config(base)
    if cfg.rules.early_narrow:
        cfg = replace(cfg, rules=replace(cfg.rules, shared_literals=True))
    return cfg


def context(row):
    """(domain, cfg, instances of row.stem by id, controller): a function list id -> rules, or ("explicit", baseline)."""
    key = (row.source_file if row.kind != "sequential" else row.run, row.stem)
    with LOCK:
        lock = _ctx_lock.setdefault(key, threading.Lock())
    with lock:
        if key in _ctx:
            return _ctx[key]
        if row.kind == "sequential":
            cfg = load_config(row.config, [f"domain.dataset={row.stem}.csv"])
            cfg = replace(cfg, rules=replace(cfg.rules, shared_literals=True)) if cfg.rules.early_narrow else cfg
            domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
            seq_dir = PROJECT_ROOT / Path(row.source_file).parent
            lib = {e["name"]: [tuple(x) for x in e["rules"]] for e in json.loads(
                (seq_dir / "library.json").read_text(encoding="utf-8"))}
            narrowed = bool(cfg.rules.early_narrow)

            def ctl(lid, lib=lib, seq_dir=seq_dir, narrowed=narrowed):
                if lid in lib:
                    return lib[lid]
                # a generated list that was not certified is not stored in the library: its generation run has it
                gen = seq_dir / lid / "seed_none_1"
                lists = RULES["final" if narrowed else "determinized"][0](gen)
                return next(iter(lists.values()))
        elif row.kind == "baseline":
            settings = load_file(PROJECT_ROOT / row.config, BaselineConfig, [])
            cfg = load_config(settings.condition)
            domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
            base = load_method(settings.method)(domain, settings.params)
            base.config = cfg
            nets = PROJECT_ROOT / Path(row.source_file).parent / "nets.pt"
            if nets.exists():
                import torch
                for k, v in torch.load(nets, weights_only=False).items():
                    setattr(base, k, v)
            else:
                train = {int(i.id): i for i in domain.load_instances(settings.dataset)}
                base.fit([train[i] for i in parse_ids(settings.train)] if settings.train else [])
            if base.explicit:
                ctl = ("explicit", base)
            else:
                insts = {int(i.id): i for i in domain.load_instances(f"{row.stem}.csv")}
                actions = list(domain.spec(next(iter(insts.values()))).actions)
                opened = row.rules == "open"
                ctl = lambda lid, insts=insts: None       # resolved per target below
                ctl = ("rules_per_target", base, opened, actions)
        else:
            run = PROJECT_ROOT / Path(row.source_file).parent
            cfg = run_config(run)
            domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
            if row.rules in ("oneshot", "unnarrowed"):
                data = json.loads((run / f"{row.rules}.json").read_text(encoding="utf-8"))
                lists = {str(s): [tuple(x) for x in e["rules"]] for s, e in data.items()}
            else:
                lists = {str(s): r for s, r in RULES[row.rules][0](run).items()}
            ctl = lambda lid, lists=lists: lists[str(lid)]
        insts = {int(i.id): i for i in domain.load_instances(f"{row.stem}.csv")}
        _ctx[key] = (domain, cfg, insts, ctl)
        return _ctx[key]


def rules_for(row, domain, insts, ctl):
    """The rule list of `row`'s controller on its instance."""
    if isinstance(ctl, tuple) and ctl[0] == "rules_per_target":
        _, base, opened, actions = ctl
        r = base.rules(insts[int(row.instance)])[str(row.list_id)]
        return open_ties(r, actions) if opened else r
    return ctl(row.list_id)


# ---------------------------------------------------------------- stage prism: interval iteration

def margins(spec, worst: Dict[str, float]) -> Dict[str, float]:
    return {r.name: r.margin(worst[r.name]) for r in spec.requirements}


def ii_check(row) -> dict:
    """Interval iteration on one check: verdict from the less favourable bound, or "storm" where it does not converge
    or its interval contains a threshold, or a timeout."""
    domain, cfg, insts, ctl = context(row)
    inst = insts[int(row.instance)]
    start = time.time()
    name, args = exact_solvers(replace(cfg.prism, exact_methods=["intervaliter"], ii_max_iters=II_MAX_ITERS))[0]
    prism = replace(cfg.prism, method="", max_iters=cfg.prism.exact_max_iters)
    runner = PrismRunner(prism, extra_args=args)
    try:
        if isinstance(ctl, tuple) and ctl[0] == "explicit":
            from core.explicit import ExplicitMDP
            mdp = ExplicitMDP(domain, inst, cfg)
            spec = mdp.spec
            model = mdp.under(ctl[1].policies(mdp)[str(row.list_id)])
            model = model.part(model.reachable())
            states = model.S
            if model.S == 1:                           # PRISM cannot import a one-state model; numpy is exact here
                worst = {r.name: model.value(r, worst=True) for r in spec.requirements}
            else:
                names = sorted({n for r in spec.requirements for n in re.findall(r'"([^"]+)"', r.formula)})
                with tempfile.TemporaryDirectory(prefix="certs_") as tmp:
                    files = model.write(Path(tmp) / "model", names)
                    props = Path(tmp) / "p.props"
                    props.write_text("".join(f"{r.worst_op()}=? [ {r.formula} ];\n" for r in spec.requirements))
                    out = run_call(PrismCall((runner.prism_path, "-importmodel", files, str(props), "-explicit",
                                              "-maxiters", str(cfg.prism.exact_max_iters), *args), prism.timeout_s,
                                             prism.max_concurrent))
                results = re.findall(r"^Result: (\S+)", out, re.M)
                if "did not converge" in out:
                    return {"verdict": "storm", "seconds": time.time() - start, "states": states}
                if len(results) != len(spec.requirements):
                    raise PrismError(out[-400:])
                worst = {r.name: float(v) for r, v in zip(spec.requirements, results)}
        else:
            from recertify import check_policy
            verifier = PolicyVerifier(domain, inst, cfg, runner)
            spec = verifier.spec
            try:
                policy = spec.policy(rules_for(row, domain, insts, ctl), spec.vocabulary(cfg.rules))
            except RuleError as e:
                return {"verdict": "failed", "solver": "unparsed", "seconds": time.time() - start, "error": str(e)[:200]}
            worst, result = check_policy(verifier, policy, runner)
            states = len(result.states)
    except subprocess.TimeoutExpired:
        return {"verdict": "timeout", "solver": II, "seconds": time.time() - start}
    except PrismError as e:
        text = str(e)
        if "did not converge" in text or "converge" in text:
            return {"verdict": "storm", "seconds": time.time() - start}
        if "OutOfMemory" in text or "heap" in text:
            return {"verdict": "timeout", "solver": II, "seconds": time.time() - start, "error": "out of memory"}
        return {"verdict": "error", "seconds": time.time() - start, "error": text.strip().splitlines()[0][:200]}
    m = margins(spec, worst)
    band = 1e-9                                     # interval iteration's precision: the bound's half-width at most
    if any(v < -band for v in m.values()):
        verdict = "failed"
    elif all(v > band for v in m.values()):
        verdict = "certified"
    else:
        return {"verdict": "storm", "seconds": time.time() - start, "worst": json.dumps(worst), "states": states}
    binding = min(m, key=m.get)
    return {"verdict": verdict, "solver": II, "margin": m[binding], "binding": binding, "worst": json.dumps(worst),
            "states": states, "seconds": time.time() - start}


PRISM_COLUMNS = ["key", "verdict", "solver", "margin", "binding", "worst", "states", "seconds", "error"]


def stage_prism(work: Path, workers: int) -> None:
    inv = pd.read_csv(work / "inventory.csv", dtype={"list_id": str})
    out = work / "prism.csv"
    done = set()
    if out.exists():
        prev = pd.read_csv(out, dtype={"list_id": str})
        done = set(prev["key"])
    todo = inv[inv.need == "prism"].copy()
    todo["key"] = todo.source_file + "|" + todo.list_id + "|" + todo.instance.astype(str) + "|" + todo.split
    todo = todo[~todo.key.isin(done)]
    # the large Pac-Man models one at a time (a 22 GB heap each), the rest in parallel
    big = todo.benchmark.eq("pacman") & todo.split.eq("test")
    print(f"{len(todo)} interval-iteration checks ({int(big.sum())} large)", flush=True)

    def one(row):
        try:
            res = ii_check(row)
        except Exception as e:
            res = {"verdict": "error", "error": f"{type(e).__name__}: {str(e)[:200]}"}
        with LOCK:
            pd.DataFrame([{"key": row.key, **res}], columns=PRISM_COLUMNS).to_csv(out, mode="a", header=not out.exists(),
                                                                                  index=False)
            if res["verdict"] in ("error",):
                print(f"  {row.key}: {res}", flush=True)
        return res

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, [r for _, r in todo[~big].iterrows()]))
    for _, r in todo[big].iterrows():
        one(r)
    print("PRISM_DONE", flush=True)


# ---------------------------------------------------------------- monitors (gridworld LTL -> reachability)

TOKEN = re.compile(r'\s*("[^"]+"|=>|[!&|()]|U\b|F\b|G\b|true\b|false\b)')


def parse(text: str):
    toks, pos = [], 0
    while pos < len(text.strip()):
        m = TOKEN.match(text, pos)
        if not m:
            raise ValueError(f"cannot parse {text!r}")
        toks.append(m.group(1)); pos = m.end()
    toks.append(None)
    i = 0

    def take():
        nonlocal i
        i += 1
        return toks[i - 1]

    def until():
        left = imp()
        if toks[i] == "U":
            take(); return ("U", left, until())
        return left

    def imp():
        left = disj()
        if toks[i] == "=>":
            take(); return ("or", ("not", left), imp())
        return left

    def disj():
        left = conj()
        while toks[i] == "|":
            take(); left = ("or", left, conj())
        return left

    def conj():
        left = unary()
        while toks[i] == "&":
            take(); left = ("and", left, unary())
        return left

    def unary():
        t = take()
        if t == "!":
            return ("not", unary())
        if t in ("F", "G"):
            return (t, unary())
        if t == "(":
            f = until(); take(); return f
        if t in ("true", "false"):
            return t == "true"
        return ("ap", t[1:-1])

    return until()


def temporal(f) -> set:
    if isinstance(f, bool) or f[0] == "ap":
        return set()
    return ({f[0]} if f[0] in ("U", "F", "G") else set()) | set().union(*(temporal(g) for g in f[1:]))


def labels_of(f) -> set:
    if isinstance(f, bool):
        return set()
    return {f[1]} if f[0] == "ap" else set().union(*(labels_of(g) for g in f[1:]))


TRUE, FALSE = frozenset({frozenset()}), frozenset()


def d_or(a, b):
    c = a | b
    return frozenset(x for x in c if not any(o < x for o in c))


def d_and(a, b):
    return d_or(frozenset(), frozenset(x | y for x in a for y in b))


def prog(f, letter):
    """Formula progression into disjunctive normal form over the formula's temporal nodes."""
    if isinstance(f, bool):
        return TRUE if f else FALSE
    op = f[0]
    if op == "ap":
        return TRUE if f[1] in letter else FALSE
    if op == "not":
        g = prog(f[1], letter)
        if temporal(f[1]):
            raise ValueError("negated temporal formula")
        return FALSE if g == TRUE else TRUE
    if op == "and":
        return d_and(prog(f[1], letter), prog(f[2], letter))
    if op == "or":
        return d_or(prog(f[1], letter), prog(f[2], letter))
    node = frozenset({frozenset({f})})
    if op == "U":
        return d_or(prog(f[2], letter), d_and(prog(f[1], letter), node))
    if op == "F":
        return d_or(prog(f[1], letter), node)
    return d_and(prog(f[1], letter), node)            # G


def step(residual, letter):
    out = FALSE
    for clause in residual:
        c = TRUE
        for node in clause:
            c = d_and(c, prog(node, letter))
        out = d_or(out, c)
    return out


class Monitor:
    """Deterministic monitor of an LTL formula (co-safe: target = true; safety (with G): target = false)."""

    def __init__(self, formula: str, letters, initial_letter):
        f = parse(formula)
        kinds = temporal(f)
        self.safety = "G" in kinds
        if self.safety and kinds - {"G"}:
            raise ValueError(f"mixed formula {formula}")
        start = prog(f, initial_letter)
        self.states, index, self.delta = [start], {start: 0}, {}
        k = 0
        while k < len(self.states):
            for L in letters:
                nxt = step(self.states[k], L)
                if nxt not in index:
                    index[nxt] = len(self.states); self.states.append(nxt)
                self.delta[k, L] = index[nxt]
            k += 1
        self.target = index.get(FALSE if self.safety else TRUE)


def read_explicit(tra: Path, lab: Path):
    names = dict((int(k), v) for k, v in re.findall(r'(\d+)="([^"]*)"', lab.read_text().splitlines()[0]))
    labels, init = {}, None
    for line in lab.read_text().splitlines()[1:]:
        s, _, ids = line.partition(":")
        got = {names[int(k)] for k in ids.split()}
        if "init" in got:
            init = int(s)
        labels[int(s)] = frozenset(got - {"init", "deadlock"})
    lines = tra.read_text().splitlines()
    S = int(lines[0].split()[0])
    choices = [dict() for _ in range(S)]
    for line in lines[1:]:
        p = line.split()
        if len(p) < 4:
            continue
        exact = Fraction(p[3])
        q = exact.limit_denominator(10 ** 9)
        if abs(q - exact) > Fraction(1, 10 ** 13):
            raise ValueError(f"probability {p[3]} has no small rational")
        choices[int(p[0])].setdefault(int(p[1]), []).append((int(p[2]), q))
    out = []
    for s in range(S):
        cs = [choices[s][c] for c in sorted(choices[s])]
        if any(sum(x for _, x in c) != 1 for c in cs):
            raise ValueError(f"state {s}: probabilities do not sum to 1")
        out.append(cs)
    return out, init, [labels.get(s, frozenset()) for s in range(S)]


def product_text(choices, init, labels, req) -> (Optional[str], Optional[str], bool):
    """PRISM-language product of the model with `req`'s monitor, the reachability query, and whether the requirement
    is a safety one (value = 1 - result). (None, None, safety) when the monitor cannot decide it (a constant)."""
    used = labels_of(parse(req.formula))
    lets = [frozenset(L & used) for L in labels]
    letters = sorted(set(lets), key=sorted)
    lid = {L: k for k, L in enumerate(letters)}
    mon = Monitor(req.formula, letters, lets[init])
    if mon.target is None:
        return None, None, mon.safety
    out = ["mdp", ""]
    for L in letters:
        expr = str(mon.delta[len(mon.states) - 1, L])
        for k in range(len(mon.states) - 2, -1, -1):
            expr = f"(q={k} ? {mon.delta[k, L]} : {expr})"
        out.append(f"formula d_{lid[L]} = {expr};")
    out += ["", "module product", f"  s : [0..{len(choices) - 1}] init {init};",
            f"  q : [0..{len(mon.states) - 1}] init 0;"]
    for s, cs in enumerate(choices):
        for c in cs:
            out.append(f"  [] s={s} -> " + " + ".join(f"{p.numerator}/{p.denominator} : (s'={t}) & (q'=d_{lid[lets[t]]})"
                                                      for t, p in c) + ";")
    out += ["endmodule", ""]
    worst_min = req.maximize
    op = ("Pmax" if worst_min else "Pmin") if mon.safety else ("Pmin" if worst_min else "Pmax")
    return "\n".join(out), f"{op}=? [ F q={mon.target} ]", mon.safety


# ---------------------------------------------------------------- stage storm-prep / storm

def storm_rows(work: Path) -> pd.DataFrame:
    inv = pd.read_csv(work / "inventory.csv", dtype={"list_id": str})
    inv["key"] = inv.source_file + "|" + inv.list_id + "|" + inv.instance.astype(str) + "|" + inv.split + "|" + \
        inv.episode.fillna(-1).astype(int).astype(str)
    need = inv.need == "storm"
    if (work / "prism.csv").exists():
        p = pd.read_csv(work / "prism.csv")
        storm_keys = set(p[p.verdict == "storm"].key)
        k4 = inv.source_file + "|" + inv.list_id + "|" + inv.instance.astype(str) + "|" + inv.split
        need |= (inv.need == "prism") & k4.isin(storm_keys)
    return inv[need]


def export_model(row, tmp: Path):
    """(spec, .tra, .lab) of the model `row`'s check certifies, as PRISM builds it."""
    domain, cfg, insts, ctl = context(row)
    inst = insts[int(row.instance)]
    if isinstance(ctl, tuple) and ctl[0] == "explicit":
        from core.explicit import ExplicitMDP
        mdp = ExplicitMDP(domain, inst, cfg)
        m = mdp.under(ctl[1].policies(mdp)[str(row.list_id)])
        m = m.part(m.reachable())
        names = sorted({n for r in mdp.spec.requirements for n in re.findall(r'"([^"]+)"', r.formula)})
        m.write(tmp / "model", names)
        return mdp.spec, tmp / "model.tra", tmp / "model.lab", None
    runner = PrismRunner(cfg.prism)
    v = PolicyVerifier(domain, inst, cfg, runner)
    spec = v.spec
    text = v.compose(spec.policy(rules_for(row, domain, insts, ctl), spec.vocabulary(cfg.rules)))
    (tmp / "m.prism").write_text(text, encoding="utf-8")
    if cfg.domain.name != "gridworld":
        return spec, None, None, text                 # Storm takes the PRISM model and properties as they are
    log = run_call(PrismCall((runner.prism_path, str(tmp / "m.prism"), "-exportmodel", str(tmp / "out.tra,lab"),
                              "-exportmodelprecision", "17"), 1800, cfg.prism.max_concurrent))
    if not (tmp / "out.tra").exists():
        raise RuntimeError("export failed: " + log[-300:])
    return spec, tmp / "out.tra", tmp / "out.lab", None


def stage_storm_prep(work: Path, workers: int) -> None:
    rows = storm_rows(work)
    root = work / "storm"; root.mkdir(exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {k: v for k, v in manifest.items() if "error" not in v}      # redo the ones that failed
    have = {v["key"] for v in manifest.values()}
    rows = rows[~rows.key.isin(have)]
    start_id = 1 + max([int(k[1:]) for k in manifest] + [-1])   # never reuse an id (a removed entry leaves a gap)
    print(f"{len(rows)} checks to prepare", flush=True)

    def one(item):
        k, row = item
        cid = f"c{start_id + k:05d}"
        try:
            with tempfile.TemporaryDirectory(prefix="certs_") as tmp:
                spec, tra, lab, model_text = export_model(row, Path(tmp))
                d = root / cid; d.mkdir(exist_ok=True)
                reqs = {}
                if model_text is not None:
                    (d / "model.prism").write_text(model_text, encoding="utf-8")
                    for r in spec.requirements:
                        (d / f"{r.name}.props").write_text(f"{r.worst_op()}=? [ {r.formula} ]", encoding="utf-8")
                        reqs[r.name] = {"threshold": r.threshold, "bound": r.bound, "safety": False, "fixed": None,
                                        "model": "model.prism", "reward": r.reward is not None}
                else:
                    choices, init, labels = read_explicit(tra, lab)
                    for r in spec.requirements:
                        text, prop, safety = product_text(choices, init, labels, r)
                        entry = {"threshold": r.threshold, "bound": r.bound, "safety": safety, "fixed": None,
                                 "model": f"{r.name}.prism", "reward": False}
                        if text is None:
                            entry["fixed"] = 1.0 if safety else 0.0
                        else:
                            (d / f"{r.name}.prism").write_text(text, encoding="utf-8")
                            (d / f"{r.name}.props").write_text(prop, encoding="utf-8")
                        reqs[r.name] = entry
            res = {"key": row.key, "requirements": reqs}
        except Exception as e:
            res = {"key": row.key, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        with LOCK:
            manifest[cid] = res
        return cid

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, enumerate(r for _, r in rows.iterrows())))
    manifest_path.write_text(json.dumps(manifest, indent=0), encoding="utf-8")
    print(f"STORM_PREP_DONE {len(manifest)} checks, {sum('error' in v for v in manifest.values())} errors", flush=True)


RUN_SH = r"""#!/bin/bash
# one Storm call per (check, requirement): --sound first (capped at SOUND s), --exact where it gives no result
cd /work
one() { d=$(dirname $1); r=$(basename $1 .props); m=$(cat $d/$r.model)
  if [ ! -f $d/$r.sound.out ] && [ ! -f $d/$r.exact.out ]; then s=$(date +%s%N)
    timeout SOUND storm --prism $d/$m --prop "$(cat $1)" --sound --precision 1e-9 --absolute > $d/$r.sound.out 2>&1
    echo "rc=$? ms=$(( ($(date +%s%N)-s)/1000000 ))" >> $d/$r.sound.out; fi
  if ! grep -qs "^Result (for initial states)" $d/$r.sound.out && [ ! -f $d/$r.exact.out ]; then s=$(date +%s%N)
    timeout EXACT storm --prism $d/$m --prop "$(cat $1)" --exact > $d/$r.exact.out 2>&1
    echo "rc=$? ms=$(( ($(date +%s%N)-s)/1000000 ))" >> $d/$r.exact.out; fi; }
export -f one
cat /work/$1 | xargs -P WORKERS -I{} bash -c "one {}"
echo ALL_DONE
"""


def stage_storm(work: Path, workers: int) -> None:
    """Storm on every requirement product, in two passes: first each check's binding requirement (its smallest margin
    in the existing check; every requirement where there is none), then the rest of the checks that pass 1 did not
    already find failed (one violated requirement settles a failed verdict; a certified one needs them all)."""
    root = work / "storm"
    manifest = json.loads((root / "manifest.json").read_text())
    for cid, c in manifest.items():
        for name, e in c.get("requirements", {}).items():
            if e["fixed"] is None:
                (root / cid / f"{name}.model").write_text(e["model"], encoding="utf-8")
    (root / "run.sh").write_text(RUN_SH.replace("SOUND", str(SOUND_TIMEOUT)).replace("EXACT", str(EXACT_TIMEOUT))
                                 .replace("WORKERS", str(workers)), encoding="utf-8", newline="\n")
    host = subprocess.run(["cygpath", "-w", str(root)], capture_output=True, text=True).stdout.strip() or str(root)
    old = {r.key: json.loads(r.worst) if isinstance(r.worst, str) else {} for r in storm_rows(work).itertuples()}

    def run(listing: List[str], name: str) -> None:
        (root / name).write_text("\n".join(listing) + "\n", encoding="utf-8", newline="\n")
        print(f"{name}: {len(listing)} requirement products", flush=True)
        subprocess.run(["docker", "run", "--rm", "-v", f"{host}:/work", STORM_IMAGE, "bash", "/work/run.sh", name],
                       check=True, env={**__import__("os").environ, "MSYS_NO_PATHCONV": "1"})

    first, rest = [], {}
    for cid, c in manifest.items():
        reqs = {n: e for n, e in c.get("requirements", {}).items() if e["fixed"] is None}
        worst = old.get(c.get("key"), {})
        margin = {n: ((worst[n] - e["threshold"]) if e["bound"] == ">=" else (e["threshold"] - worst[n]))
                  for n, e in reqs.items() if n in worst}
        lead = [min(margin, key=margin.get)] if margin else list(reqs)
        first += [f"{cid}/{n}.props" for n in lead]
        rest[cid] = [f"{cid}/{n}.props" for n in reqs if n not in lead]
    run(first, "pass1.txt")
    verdicts = storm_verdicts(work)
    failed = {cid for cid, c in manifest.items() if "requirements" in c and
              decide(c["requirements"], verdicts.get(c["key"], {}).get("worst", {}))[0] == "failed"}
    run([p for cid, ps in rest.items() if cid not in failed for p in ps], "pass2.txt")
    print(f"STORM_DONE ({len(failed)} checks failed on their binding requirement)", flush=True)


RESULT = re.compile(r"Result \(for initial states\): ([0-9/.eE+-]+?)\.?(?:\s|$)")
BOUNDS = re.compile(r"Solution bounds: \[([^,\]]+), ([^\]]+)\]")
TIME = re.compile(r"^rc=(\d+) ms=(\d+)", re.M)


def storm_value(path: Path):
    """(value, low, high, seconds) of one Storm output: exact results have low = high = value; None if no result."""
    if not path.exists():
        return None
    text = path.read_text(errors="replace")
    t = TIME.search(text)
    secs = int(t.group(2)) / 1000 if t else 0.0
    m = RESULT.search(text)
    if not m:
        return None, None, None, secs, ("timeout" if t and int(t.group(1)) == 124 else "error")
    if path.name.endswith(".exact.out") and "Exact solution" not in text:
        return None, None, None, secs, "not an exact solution"      # never accept a silent fallback (cf. PRISM's -lp)
    v = Fraction(m.group(1))
    b = BOUNDS.search(text)
    if b:                                              # --sound: the result's guaranteed interval
        return v, Fraction(b.group(1)), Fraction(b.group(2).rstrip(".")), secs, None
    if "Exact solution" not in text:                   # a sound value without bounds: precision 1e-9, absolute
        return v, v - Fraction(1, 10 ** 9), v + Fraction(1, 10 ** 9), secs, None
    return v, v, v, secs, None


def storm_verdicts(work: Path) -> Dict[str, dict]:
    root = work / "storm"
    manifest = json.loads((root / "manifest.json").read_text())
    out = {}
    for cid, c in manifest.items():
        if "error" in c:
            out[c["key"]] = {"verdict": "error", "error": c["error"]}
            continue
        worst, decided, secs, solvers, why = {}, {}, 0.0, set(), []
        for name, e in c["requirements"].items():
            thr, maximize = Fraction(str(e["threshold"])), e["bound"] == ">="
            if e["fixed"] is not None:
                worst[name] = (Fraction(str(e["fixed"])),) * 3
                continue
            got = None
            for mode in ("exact", "sound"):                 # an exact result supersedes a sound one
                r = storm_value(root / cid / f"{name}.{mode}.out")
                if r is None:
                    continue
                v, lo, hi, s, fail = r
                secs += s
                if fail is None:
                    got, solver = (v, lo, hi), f"Storm --{mode}"
                    break
                why.append(f"{name} {mode}: {fail}")
            if got is None:
                continue
            v, lo, hi = got
            if e["safety"]:
                v, lo, hi = 1 - v, 1 - hi, 1 - lo
            worst[name] = (v, lo, hi)
            solvers.add(solver)
        out[c["key"]] = {"worst": worst, "seconds": secs, "solvers": solvers, "why": why,
                         "requirements": c["requirements"]}
    return out


def decide(spec_reqs: dict, worst: dict):
    """Verdict from the less favourable bound of each requirement (margins as Requirement.margin)."""
    m_low, m_high = {}, {}
    for name, e in spec_reqs.items():
        if name not in worst:
            continue
        thr, maximize = Fraction(str(e["threshold"])), e["bound"] == ">="
        v, lo, hi = worst[name]
        scale = abs(thr) if e.get("reward") and thr else 1
        m_low[name] = ((lo - thr) if maximize else (thr - hi)) / scale     # the less favourable bound's margin
        m_high[name] = ((hi - thr) if maximize else (thr - lo)) / scale
    if any(x < 0 for x in m_high.values()):
        bad = min(m_high, key=m_high.get)
        return "failed", float(m_high[bad]), bad
    if len(m_low) == len(spec_reqs) and all(x >= 0 for x in m_low.values()):
        b = min(m_low, key=m_low.get)
        return "certified", float(m_low[b]), b
    return None, None, None


# ---------------------------------------------------------------- collect

def collect(work: Path) -> None:
    inv = pd.read_csv(work / "inventory.csv", dtype={"list_id": str})
    inv["key4"] = inv.source_file + "|" + inv.list_id + "|" + inv.instance.astype(str) + "|" + inv.split
    inv["key"] = inv.key4 + "|" + inv.episode.fillna(-1).astype(int).astype(str)
    prism = pd.read_csv(work / "prism.csv").set_index("key") if (work / "prism.csv").exists() else pd.DataFrame()
    storm = storm_verdicts(work) if (work / "storm" / "manifest.json").exists() else {}
    rows = []
    for _, r in inv.iterrows():
        row = {c: r.get(c) for c in COLUMNS if c in r}
        if r.need == "reuse":
            row.update(verdict=r.old_verdict, solver=r.old_solver if isinstance(r.old_solver, str) else II,
                       margin=r.old_margin, binding=r.get("binding"))
        elif r.need == "prism" and r.key4 in prism.index and prism.loc[r.key4, "verdict"] != "storm":
            p = prism.loc[r.key4]
            row.update(verdict=p.verdict, solver=p.get("solver"), margin=p.get("margin"), binding=p.get("binding"),
                       worst=p.get("worst"), states=p.get("states"), seconds=p.get("seconds"))
        else:
            s = storm.get(r.key)
            if s is None or "error" in s:
                row.update(verdict="error", solver=None, margin=None)
            else:
                verdict, margin, binding = decide(s["requirements"], s["worst"])
                if verdict is None:
                    verdict = "timeout" if any("timeout" in w for w in s["why"]) else "inconclusive"
                row.update(verdict=verdict, solver=" + ".join(sorted(s["solvers"])) or "monitor (graph, exact)", margin=margin,
                           binding=binding, worst=json.dumps({k: float(v[0]) for k, v in s["worst"].items()}),
                           seconds=s["seconds"] + (0 if pd.isna(r.seconds) else 0))
        rows.append(row)
    certs = pd.DataFrame(rows)
    CERTS.parent.mkdir(parents=True, exist_ok=True)
    certs[[c for c in COLUMNS if c in certs]].to_csv(CERTS, index=False)
    report(certs, work)
    print(f"wrote {CERTS.relative_to(PROJECT_ROOT)}: {len(certs)} verdicts", flush=True)


def md(df: pd.DataFrame, index: bool = True) -> str:
    """A DataFrame as a Markdown table."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines)


def report(certs: pd.DataFrame, work: Path) -> None:
    c = certs.copy()
    had = c.old_verdict.notna()
    changed = had & (c.old_verdict != c.verdict)
    lines = ["# certs.csv: what changed", "",
             f"{len(c)} verdicts; {int(had.sum())} existed before, {int((~had).sum())} are new checks.", "",
             "## Verdicts", "", c.groupby(["split", "verdict"]).size().unstack(fill_value=0).pipe(md), "",
             "## Changed verdicts, by controller and benchmark", "",
             (c[changed].fillna({"llm": "-"}).groupby(["benchmark", "controller", "llm", "old_verdict", "verdict"]).size()
              .rename("checks").reset_index().pipe(md, index=False) if changed.any() else "None."), "",
             "## Solvers", "", c.groupby("solver", dropna=False).size().rename("checks").to_frame().pipe(md), "",
             f"Checks that needed Storm: {int(c.solver.fillna('').str.contains('Storm').sum())}; timed out: "
             f"{int((c.verdict == 'timeout').sum())} (of which new: {int(((c.verdict == 'timeout') & ~had).sum())}).",
             ""]
    seq = c[c.controller.str.startswith("Library") & changed]
    lines += ["## Sequential studies", "",
              (f"{len(seq)} changed verdicts feed sequential studies:\n\n" +
               seq[["run", "episode", "list_id", "instance", "old_verdict", "verdict", "margin"]].pipe(md, index=False))
              if len(seq) else "No changed verdict feeds a decision in a sequential study.", ""]
    (work / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["inventory", "prism", "storm-prep", "storm", "collect"])
    ap.add_argument("work", type=Path)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    {"inventory": lambda: inventory(a.work), "prism": lambda: stage_prism(a.work, a.workers),
     "storm-prep": lambda: stage_storm_prep(a.work, a.workers), "storm": lambda: stage_storm(a.work, a.workers),
     "collect": lambda: collect(a.work)}[a.stage]()


if __name__ == "__main__":
    main()
