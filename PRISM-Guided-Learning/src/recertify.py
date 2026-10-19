"""Re-certify frozen controllers with the exact solvers on every check (the paper's certification rule).

`src/transfer.py` and `src/run_baseline.py` check held-out targets with the loop's solver and re-check only near a
threshold, so they cannot say when a verdict is inconclusive. This applies `sequential.certify`'s rule to every
(controller, target) pair: interval iteration at `prism.exact_epsilon`, falling back to Gauss-Seidel at
`prism.exact_fallback_epsilon`; certified iff interval iteration converged and the smallest margin over the
requirements is > epsilon; inconclusive when it is within +-epsilon, or when only the fallback finished and the margin
is >= -epsilon; failed otherwise. A check where neither solver finishes, or whose model cannot be built, is
"not certifiable within limits", with the reason. It changes nothing in the loop or the other checkers.

Usage (from PRISM-Guided-Learning/):
    python src/recertify.py run <run dir> --rules determinized [--targets 12-19] [--data D] [--workers 2]
        [--prefix transfer_ii] [--unseal]
    python src/recertify.py baseline configs/baselines/<name>.yaml [--rules final|open] [--set key=value ...]
        [--workers 2] [--prefix transfer_ii] [--unseal]
`run` writes <run dir>/<prefix>_<dataset stem><rules suffix>.csv (suffixes as src/transfer.py's); `baseline` refits
the baseline as src/run_baseline.py does and writes out/results/baselines/<name>/<prefix>_<dataset stem>[_open].csv.
Rows of an existing CSV for targets not checked again are kept. Columns: source, target, trained, outcome
(certified | failed | inconclusive | not_certifiable | unparsed), certified, solver, fallback, margin, binding (the
requirement with the smallest margin), worst_<requirement>, seconds, model_seconds, states, error.
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
from core.verifier import PolicyVerifier
from run_baseline import BASELINES_PATH, BaselineConfig
from transfer import RULES, guard_sealed, merge, parse_ids, training_instances

MODEL_TIME = re.compile(r"Time for model construction:\s*([\d.]+)")
II = "interval iteration"


def verdict(margins: Dict[str, float], solver: str, eps: float) -> str:
    """sequential.certify's rule on one check's margins."""
    smallest = min(margins.values())
    if solver != II or abs(smallest) <= eps:
        return "inconclusive" if smallest >= -eps else "failed"
    return "certified" if smallest > eps else "failed"


def summarize(row: dict, spec, worst: Dict[str, float], solver: str, eps: float) -> dict:
    margins = {r.name: r.margin(worst[r.name]) for r in spec.requirements}
    binding = min(margins, key=margins.get)
    outcome = verdict(margins, solver, eps)
    return {**row, "outcome": outcome, "certified": outcome == "certified", "solver": solver,
            "fallback": solver != II, "margin": margins[binding], "binding": binding,
            **{f"worst_{k}": worst[k] for k in margins}}


def certify_rules(domain, cfg, instance, sources: Dict[str, list]) -> List[dict]:
    """Every source rule list on one target, by the exact solvers through the rule-list model."""
    eps = float(cfg.prism.exact_epsilon)
    exact = replace(cfg.prism, method="", max_iters=cfg.prism.exact_max_iters)
    solvers = [(II, ["-intervaliter", "-topological", "-epsilon", cfg.prism.exact_epsilon]),
               (f"Gauss-Seidel {cfg.prism.exact_fallback_epsilon}",
                ["-gaussseidel", "-epsilon", cfg.prism.exact_fallback_epsilon])]
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
        rows.append({**summarize(row, spec, v.worst, solver, eps), "seconds": seconds,
                     "model_seconds": sum(float(x) for x in MODEL_TIME.findall(stdout)),
                     "states": len(v.result.states), "error": " | ".join(errors)})
    return rows


def certify_explicit(domain, cfg, instance, policies_of) -> List[dict]:
    """Per-state policies on one target, through the target's explicit model restricted to each policy (as
    ExplicitMDP.certify), with the exact solvers and the same verdict rule. `policies_of(mdp)` -> {source: policy}."""
    eps = float(cfg.prism.exact_epsilon)
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
            worst, solver, stdout, errors = {r.name: model.value(r, worst=True) for r in reqs}, II, "", []
        else:
            worst, solver, stdout, errors = prism_worst(mdp, model)
        seconds = time.time() - start
        if worst is None:
            rows.append({**row, "outcome": "not_certifiable", "certified": False, "seconds": seconds,
                         "states": model.S, "error": " | ".join(errors)})
            continue
        rows.append({**summarize(row, mdp.spec, worst, solver, eps), "seconds": seconds,
                     "model_seconds": sum(float(x) for x in MODEL_TIME.findall(stdout)), "states": model.S,
                     "error": " | ".join(errors)})
    del mdp
    return rows


def prism_worst(mdp: ExplicitMDP, model) -> Tuple[Optional[Dict[str, float]], Optional[str], str, List[str]]:
    """ExplicitMDP._prism_worst, also returning the solver used, PRISM's output and each solver's failure."""
    reqs = mdp.spec.requirements
    names = sorted({n for r in reqs for n in re.findall(r'"([^"]+)"', r.formula)})
    cfg = mdp.config.prism
    solvers = [(II, ["-intervaliter", "-topological", "-epsilon", cfg.exact_epsilon]),
               (f"Gauss-Seidel {cfg.exact_fallback_epsilon}", ["-gaussseidel", "-epsilon", cfg.exact_fallback_epsilon])]
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
    cfg = load_run_config(args.run_dir)
    if args.rules == "narrowed" or cfg.rules.early_narrow:   # narrowings name places with numbers (as transfer.py)
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
    write(df, args.run_dir / f"{args.prefix}_{Path(dataset).stem}{suffix}.csv")


def baseline_cmd(args) -> None:
    settings = load_file(args.config, BaselineConfig, args.set)
    target_dataset = settings.target_dataset or settings.dataset
    guard_sealed(target_dataset, parse_ids(settings.targets), args.unseal)
    cfg = load_config(settings.condition)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    instances = {int(i.id): i for i in domain.load_instances(settings.dataset)}
    train = parse_ids(settings.train) if settings.train else []
    baseline = load_method(settings.method)(domain, settings.params)
    baseline.config = cfg
    baseline.fit([instances[i] for i in train])
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
    write(df, out_dir / f"{args.prefix}_{Path(target_dataset).stem}{suffix}.csv")
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="kind", required=True)
    r = sub.add_parser("run", help="A run's rule lists")
    r.add_argument("run_dir", type=Path)
    r.add_argument("--rules", choices=list(RULES), default="determinized")
    r.add_argument("--targets", default=None)
    r.add_argument("--data", default=None)
    b = sub.add_parser("baseline", help="A baseline, refitted from its config")
    b.add_argument("config", type=Path)
    b.add_argument("--rules", choices=["final", "open"], default="final")
    b.add_argument("--set", action="append", default=[], help="Baseline config override key=value")
    for p in (r, b):
        p.add_argument("--workers", type=int, default=2)
        p.add_argument("--prefix", default="transfer_ii", help="Output file prefix (transfer_sealed for the test pass)")
        p.add_argument("--unseal", action="store_true", help="Allow sealed test targets (the test pass only)")
        p.add_argument("--source-targets", default=None,
                       help="JSON, source -> its own target ids (e.g. each family's list on its family only)")
    args = parser.parse_args(argv)
    (run_cmd if args.kind == "run" else baseline_cmd)(args)


if __name__ == "__main__":
    main()
