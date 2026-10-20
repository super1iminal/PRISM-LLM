"""Re-certify frozen controllers with the exact solvers on every check (the paper's certification rule).

`src/transfer.py` and `src/run_baseline.py` check held-out targets with the loop's solver and re-check only near a
threshold, so they cannot say when a verdict is inconclusive. This applies the certification rule
(`core.verifier.exact_verdict`, as `sequential.certify`) to every (controller, target) pair: the exact solvers in turn
(`prism.exact_methods`: interval iteration at `prism.exact_epsilon`, capped at `prism.ii_max_iters` iterations,
then PRISM's linear programming where it stalls), falling back to Gauss-Seidel at `prism.exact_fallback_epsilon`;
certified iff an exact solver finished and the smallest margin over the requirements is beyond its band
(`prism.exact_epsilon` for interval iteration, `prism.lp_tolerance` for linear programming); inconclusive within the
band, or when only the fallback finished and the margin is >= -band; failed otherwise. A check where no solver
finishes, or whose model cannot be built, is "not certifiable within limits", with the reason. It changes nothing in
the loop or the other checkers.

Usage (from PRISM-Guided-Learning/):
    python src/recertify.py run <run dir> --rules determinized [--targets 12-19] [--data D] [--workers 2]
        [--prefix transfer_exact] [--unseal]
    python src/recertify.py baseline configs/baselines/<name>.yaml [--rules final|open] [--set key=value ...]
        [--workers 2] [--prefix transfer_exact] [--unseal]
    python src/recertify.py reverdict <csv> [<csv> ...]
    python src/recertify.py combine <dir or csv> [...]
`run` writes <run dir>/<prefix>_<dataset stem><rules suffix>.csv (suffixes as src/transfer.py's); `baseline` refits
the baseline as src/run_baseline.py does and writes out/results/baselines/<name>/<prefix>_<dataset stem>[_open].csv.
The prefix defaults to transfer_exact (transfer_ii or transfer_lp when `prism.exact_methods` names one method). Rows
of an existing CSV for targets not checked again are kept. `reverdict` re-applies the certification rule with the
current bands to written CSVs, from their margins and solvers. `combine` builds transfer_exact_<x>.csv from checks
made with one method each: interval iteration's row where it converged (transfer_ii_<x>.csv), else linear
programming's (transfer_lp_<x>.csv). Columns: source, target, trained, outcome (certified | failed | inconclusive |
not_certifiable | unparsed), certified, solver, fallback, margin, binding (the requirement with the smallest margin),
worst_<requirement>, seconds, model_seconds, states, error.
"""
import argparse
import json
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from baselines import load_method
from baselines.explicit_mdp import ExplicitMDP
from config import load_config, load_file, load_run_config
from core.domain import load_domain
from core.prism import PrismCall, PrismError, PrismRunner, run_call, run_now
from core.rules import RuleError, open_ties
from core.verifier import SOUND, PolicyVerifier, exact_solvers, exact_verdict
from run_baseline import BASELINES_PATH, BaselineConfig
from transfer import RULES, guard_sealed, merge, parse_ids, training_instances

MODEL_TIME = re.compile(r"Time for model construction:\s*([\d.]+)")


PREFIXES = {("intervaliter",): "transfer_ii", ("lp",): "transfer_lp"}


def prefix_of(cfg) -> str:
    """The output prefix for `cfg`'s exact solvers."""
    return PREFIXES.get(tuple(cfg.prism.exact_methods), "transfer_exact")


def summarize(row: dict, spec, worst: Dict[str, float], solver: str, prism) -> dict:
    margins = {r.name: r.margin(worst[r.name]) for r in spec.requirements}
    binding = min(margins, key=margins.get)
    outcome = exact_verdict(margins, solver, prism)
    return {**row, "outcome": outcome, "certified": outcome == "certified", "solver": solver,
            "fallback": solver not in SOUND.values(), "margin": margins[binding], "binding": binding,
            **{f"worst_{k}": worst[k] for k in margins}}


def certify_rules(domain, cfg, instance, sources: Dict[str, list]) -> List[dict]:
    """Every source rule list on one target, by the exact solvers through the rule-list model."""
    exact = replace(cfg.prism, method="", max_iters=cfg.prism.exact_max_iters)
    solvers = exact_solvers(cfg.prism)
    rows = []
    try:
        verifier = PolicyVerifier(domain, instance, cfg, PrismRunner(cfg.prism))
    except Exception as e:      # the instance's model cannot be rendered
        return [{"source": s, "target": int(instance.id), "outcome": "not_certifiable", "certified": False,
                 "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"} for s in sources]
    spec = verifier.spec
    vocab = spec.vocabulary(cfg.rules)
    for source, rules in sources.items():
        row = {"source": source, "target": int(instance.id)}
        start = time.time()
        try:
            policy = spec.policy(rules, vocab)
        except RuleError as e:
            rows.append({**row, "outcome": "unparsed", "certified": False, "error": str(e).splitlines()[0]})
            continue
        v, solver, errors = None, None, []
        for name, args in solvers:
            try:
                runner = PrismRunner(exact, extra_args=args)
                v, solver = run_now(verifier.verify_steps(policy, analysis=False, runner=runner)), name
                break
            except (PrismError, subprocess.TimeoutExpired) as e:
                v, solver = None, None
                errors.append(f"{name}: {type(e).__name__}: {str(e).strip().splitlines()[0][:160] if str(e).strip() else ''}")
        seconds = time.time() - start
        if v is None:
            rows.append({**row, "outcome": "not_certifiable", "certified": False, "seconds": seconds,
                         "error": " | ".join(errors)})
            continue
        stdout = getattr(v.result, "stdout", "") or ""
        rows.append({**summarize(row, spec, v.worst, solver, cfg.prism), "seconds": seconds,
                     "model_seconds": sum(float(x) for x in MODEL_TIME.findall(stdout)),
                     "states": len(v.result.states), "error": " | ".join(errors)})
    return rows


def certify_explicit(domain, cfg, instance, policies_of) -> List[dict]:
    """Per-state policies on one target, through the target's explicit model restricted to each policy (as
    ExplicitMDP.certify), with the exact solvers and the same verdict rule. `policies_of(mdp)` -> {source: policy}."""
    try:
        mdp = ExplicitMDP(domain, instance, cfg)
    except Exception as e:      # the bare model cannot be built (memory, timeout, size)
        return [{"source": "*", "target": int(instance.id), "outcome": "not_certifiable", "certified": False,
                 "error": f"bare model: {type(e).__name__}: {str(e).strip().splitlines()[0][:200] if str(e).strip() else ''}"}]
    reqs = mdp.spec.requirements
    rows = []
    for source, policy in policies_of(mdp).items():
        row = {"source": source, "target": int(instance.id)}
        start = time.time()
        model = mdp.under(policy)
        model = model.part(model.reachable())
        if model.S == 1:        # PRISM cannot import a one-state model; its values are exact in numpy
            worst, solver, stdout, errors = ({r.name: model.value(r, worst=True) for r in reqs},
                                             SOUND[cfg.prism.exact_methods[0]], "", [])
        else:
            worst, solver, stdout, errors = prism_worst(mdp, model)
        seconds = time.time() - start
        if worst is None:
            rows.append({**row, "outcome": "not_certifiable", "certified": False, "seconds": seconds,
                         "states": model.S, "error": " | ".join(errors)})
            continue
        rows.append({**summarize(row, mdp.spec, worst, solver, cfg.prism), "seconds": seconds,
                     "model_seconds": sum(float(x) for x in MODEL_TIME.findall(stdout)), "states": model.S,
                     "error": " | ".join(errors)})
    del mdp
    return rows


def prism_worst(mdp: ExplicitMDP, model) -> Tuple[Optional[Dict[str, float]], Optional[str], str, List[str]]:
    """ExplicitMDP._prism_worst, also returning the solver used, PRISM's output and each solver's failure."""
    reqs = mdp.spec.requirements
    names = sorted({n for r in reqs for n in re.findall(r'"([^"]+)"', r.formula)})
    cfg = mdp.config.prism
    solvers = exact_solvers(cfg)
    errors = []
    with tempfile.TemporaryDirectory(prefix="recertify_") as tmp:
        files = model.write(Path(tmp) / "model", names)
        props = Path(tmp) / "props.props"
        props.write_text("".join(f"{r.worst_op()}=? [ {r.formula} ];\n" for r in reqs), encoding="utf-8")
        for name, args in solvers:
            cmd = [mdp.runner.prism_path, "-importmodel", files, str(props), "-explicit", "-javamaxmem",
                   mdp.runner.java_max_mem, "-maxiters", str(cfg.exact_max_iters), *args]
            try:
                out = run_call(PrismCall(tuple(cmd), mdp.runner.timeout, mdp.runner.max_concurrent))
            except (PrismError, subprocess.TimeoutExpired) as e:
                errors.append(f"{name}: {type(e).__name__}")
                continue
            results = re.findall(r"^Result: (\S+)", out, re.M)
            if "Error" not in out and "did not converge" not in out and len(results) == len(reqs):
                return {r.name: float(v) for r, v in zip(reqs, results)}, name, out, errors
            errors.append(f"{name}: " + ("did not converge" if "did not converge" in out else "error"))
    return None, None, "", errors


def own_targets(spec: Optional[str]):
    """`spec`: JSON, source -> its target ids (e.g. '{"0": "36-59"}'; sources not named are checked everywhere).
    Returns a filter of a {source: rules} dict to the sources checked on one target."""
    allowed = {str(s): set(parse_ids(ids)) for s, ids in json.loads(spec).items()} if spec else {}
    return lambda sources, t: {s: r for s, r in sources.items() if s not in allowed or int(t.id) in allowed[s]}


def write(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    merge(df, out).sort_values(["source", "target"], key=lambda c: c.astype(str)).to_csv(out, index=False)
    counts = df.outcome.value_counts().to_dict()
    print(f"{out}: {len(df)} checks, {counts}", flush=True)


def run_cmd(args) -> None:
    cfg = with_prism(load_run_config(args.run_dir), args.prism)
    if cfg.rules.early_narrow:   # narrowings name places with numbers (as transfer.py)
        cfg = replace(cfg, rules=replace(cfg.rules, shared_literals=True))
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    dataset = args.data or cfg.domain.dataset
    targets = domain.load_instances(dataset)
    if args.targets:
        wanted = set(parse_ids(args.targets))
        targets = [t for t in targets if int(t.id) in wanted]
    guard_sealed(dataset, [int(t.id) for t in targets], args.unseal)
    loader, suffix = RULES[args.rules]
    sources = {str(s): r for s, r in loader(args.run_dir).items()}
    if args.rules == "open":
        actions = list(domain.spec(targets[0]).actions)
        sources = {s: open_ties(r, actions) for s, r in sources.items()}
    only = own_targets(args.source_targets)
    with ThreadPoolExecutor(args.workers) as pool:
        rows = [row for part in pool.map(lambda t: certify_rules(domain, cfg, t, only(sources, t)), targets)
                for row in part]
    df = pd.DataFrame(rows)
    trained = training_instances(args.run_dir) if Path(dataset).name == Path(cfg.domain.dataset).name else {}
    df["trained"] = [t in trained.get(int(s), []) for s, t in zip(df.source, df.target)]
    write(df, args.run_dir / f"{args.prefix or prefix_of(cfg)}_{Path(dataset).stem}{suffix}.csv")


FITTED = ("nets", "scale", "ids")      # what a fitted network baseline (baselines/ppo.py) needs to act again


def save_fit(baseline, path: Path) -> None:
    """The fitted networks of `baseline` (and what their observations need), for `--nets` to load them later."""
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: getattr(baseline, k) for k in FITTED if getattr(baseline, k, None) is not None}, path)
    print(f"saved {path}", flush=True)


def torch_load(path) -> dict:
    import torch
    return torch.load(path, weights_only=False)


def with_prism(cfg, overrides: List[str]):
    """`cfg` with PRISM settings overridden by `key=value` strings (e.g. max_concurrent=12, java_max_mem=12g)."""
    if not overrides:
        return cfg
    values = {}
    for item in overrides:
        key, _, value = item.partition("=")
        current = getattr(cfg.prism, key)
        values[key] = type(current)(value) if current is not None and not isinstance(current, str) else value
    return replace(cfg, prism=replace(cfg.prism, **values))


def baseline_cmd(args) -> None:
    settings = load_file(args.config, BaselineConfig, args.set)
    target_dataset = settings.target_dataset or settings.dataset
    guard_sealed(target_dataset, parse_ids(settings.targets), args.unseal)
    cfg = with_prism(load_config(settings.condition), args.prism)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    instances = {int(i.id): i for i in domain.load_instances(settings.dataset)}
    train = parse_ids(settings.train) if settings.train else []
    baseline = load_method(settings.method)(domain, settings.params)
    baseline.config = cfg
    if args.nets and Path(args.nets).exists():      # the networks of an earlier fit, so other targets use the same ones
        for key, value in torch_load(args.nets).items():
            setattr(baseline, key, value)
        print(f"loaded {args.nets}", flush=True)
    else:
        baseline.fit([instances[i] for i in train])
        if args.nets:
            save_fit(baseline, Path(args.nets))
    same = Path(target_dataset).name == Path(settings.dataset).name
    pool_of = instances if same else {int(i.id): i for i in domain.load_instances(target_dataset)}
    targets = [pool_of[i] for i in parse_ids(settings.targets)]
    if baseline.explicit:
        if args.rules == "open":
            raise SystemExit("--rules open: per-state policies have no ties to leave open")
        rows = []
        for t in targets:       # one target at a time: a model of millions of states takes gigabytes
            rows += certify_explicit(domain, cfg, t, baseline.policies)
            print(f"  {t.id} done", flush=True)
    else:
        actions = list(domain.spec(targets[0]).actions)

        only = own_targets(args.source_targets)

        def lists(t):
            return only({str(s): (open_ties(r, actions) if args.rules == "open" else r)
                         for s, r in baseline.rules(t).items()}, t)
        with ThreadPoolExecutor(args.workers) as pool:
            rows = [row for part in pool.map(lambda t: certify_rules(domain, cfg, t, lists(t)), targets) for row in part]
    df = pd.DataFrame(rows)
    df["trained"] = [same and t in (baseline.training(s) or train) for s, t in zip(df.source, df.target)]
    df["method"] = settings.method
    suffix = "_open" if args.rules == "open" else ""
    out_dir = BASELINES_PATH / args.config.stem
    write(df, out_dir / f"{args.prefix or prefix_of(cfg)}_{Path(target_dataset).stem}{suffix}.csv")
    (out_dir / "sizes.json").write_text(json.dumps(controller_sizes(baseline), indent=1), encoding="utf-8")


def controller_sizes(baseline) -> Dict[str, dict]:
    """Per source: a tree's nodes, depth and leaves, or a network's parameter count (for the paper's size table)."""
    out = {}
    for name, tree in getattr(baseline, "trees", {}).items():
        out[name] = {"nodes": int(tree.tree_.node_count), "depth": int(tree.get_depth()),
                     "leaves": int(tree.get_n_leaves())}
    for name, info in (getattr(baseline, "info", None) or {}).items():
        if isinstance(info, dict) and "nodes" in info:
            out[name] = {k: info[k] for k in ("nodes", "depth", "leaves") if k in info}
    for name, net in getattr(baseline, "nets", {}).items():
        out[name] = {"parameters": int(sum(p.numel() for p in net.parameters()))}
    return out


def reverdict(path: Path, prism) -> pd.DataFrame:
    """`path`'s outcome and certified columns again from its margins and solvers, with `prism`'s bands."""
    df = pd.read_csv(path)
    rated = df.margin.notna() & df.outcome.isin(["certified", "failed", "inconclusive"])
    df.loc[rated, "outcome"] = [exact_verdict({"smallest": m}, s, prism) for m, s in zip(df.margin[rated], df.solver[rated])]
    df["certified"] = df.outcome == "certified"
    df.to_csv(path, index=False)
    return df


def combine(stem_dir: Path, stem: str, prism) -> pd.DataFrame:
    """transfer_exact_<stem>.csv in `stem_dir` from its one-method checks: per (source, target), a row already there,
    else interval iteration's where it converged, else linear programming's, else whatever was checked; verdicts again
    with `prism`'s bands."""
    key = ["source", "target"]
    parts = {p: pd.read_csv(stem_dir / f"{p}_{stem}.csv") for p in ("transfer_exact", "transfer_ii", "transfer_lp")
             if (stem_dir / f"{p}_{stem}.csv").exists()}
    ii, lp = parts.get("transfer_ii"), parts.get("transfer_lp")
    picks = [parts["transfer_exact"]] if "transfer_exact" in parts else []
    if ii is not None:
        picks.append(ii[(ii.solver == SOUND["intervaliter"]) & ii.outcome.isin(["certified", "failed"])])
    if lp is not None:
        picks.append(lp[lp.solver == SOUND["lp"]])
    picks += [d for d in (ii, lp) if d is not None]
    df = pd.concat([d.assign(source=d.source.astype(str)) for d in picks], ignore_index=True).drop_duplicates(key)
    out = stem_dir / f"transfer_exact_{stem}.csv"
    df.sort_values(key, key=lambda c: c.astype(str)).to_csv(out, index=False)
    return reverdict(out, prism)


def combine_cmd(paths: List[Path], prism) -> None:
    """`combine` over CSVs and the transfer_ii / transfer_lp CSVs under directories."""
    stems = set()
    for path in paths:
        for f in (sorted(path.rglob("transfer_*.csv")) if path.is_dir() else [path]):
            m = re.match(r"transfer_(ii|lp)_(.+)\.csv$", f.name)
            if m:
                stems.add((f.parent, m.group(2)))
    for stem_dir, stem in sorted(stems):
        df = combine(stem_dir, stem, prism)
        left = int((df.outcome == "inconclusive").sum())
        print(f"{stem_dir / f'transfer_exact_{stem}.csv'}: {df.outcome.value_counts().to_dict()}"
              + (f" ({left} still inconclusive)" if left else ""), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="kind", required=True)
    v = sub.add_parser("reverdict", help="Written CSVs again, with the current bands")
    v.add_argument("csvs", type=Path, nargs="+")
    c = sub.add_parser("combine", help="transfer_exact_*.csv from the transfer_ii / transfer_lp checks")
    c.add_argument("paths", type=Path, nargs="+")
    r = sub.add_parser("run", help="A run's rule lists")
    r.add_argument("run_dir", type=Path)
    r.add_argument("--rules", choices=list(RULES), default="determinized")
    r.add_argument("--targets", default=None)
    r.add_argument("--data", default=None)
    b = sub.add_parser("baseline", help="A baseline, refitted from its config")
    b.add_argument("config", type=Path)
    b.add_argument("--rules", choices=["final", "open"], default="final")
    b.add_argument("--set", action="append", default=[], help="Baseline config override key=value")
    b.add_argument("--nets", default=None, help="Fitted networks: loaded if the file exists, else fitted and saved there")
    for p in (r, b):
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--prefix", default=None, help="Output file prefix (default transfer_exact; transfer_sealed "
                       "for the test pass)")
        p.add_argument("--unseal", action="store_true", help="Allow sealed test targets (the test pass only)")
        p.add_argument("--prism", action="append", default=[],
                       help="PRISM setting override key=value (e.g. max_concurrent=12, java_max_mem=12g, timeout_s=3600)")
        p.add_argument("--source-targets", default=None,
                       help="JSON, source -> its own target ids (e.g. each family's list on its family only)")
    args = parser.parse_args(argv)
    if args.kind == "reverdict":
        prism = load_config().prism
        for path in args.csvs:
            print(f"{path}: {reverdict(path, prism).outcome.value_counts().to_dict()}", flush=True)
        return
    if args.kind == "combine":
        combine_cmd(args.paths, load_config().prism)
        return
    (run_cmd if args.kind == "run" else baseline_cmd)(args)


if __name__ == "__main__":
    main()
