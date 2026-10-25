"""Exact re-check (item 3, night of Oct 21): every certification check that linear programming or Gauss-Seidel
decided, re-run with PRISM's exact engine (`-exact`: rational arithmetic, no tolerance), compared exactly with each
threshold. Gridworld's sequence requirements are LTL, which the exact engine does not take; there the model is
exported and checked through deterministic monitors (ltl_exact.py). Input: fallback_checks.csv (file, source,
target, episode, solver, outcome, margin, states). Output (appended per check, so a restart resumes).

Usage (from PRISM-Guided-Learning/): python exact_recheck.py <fallback csv> <out csv> [--workers 6] [--only PATTERN]
"""
import argparse, json, re, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
from baselines import load_method
from config import load_config, load_file, load_run_config
from core.domain import load_domain
from core.prism import PrismCall, PrismRunner, run_call
from core.rules import open_ties
from core.verifier import PolicyVerifier
from run_baseline import BaselineConfig
from transfer import RULES, parse_ids
import recertify
from ltl_exact import product_model, read_explicit

RAT = re.compile(r"^Result: (\S+) \(exact\)", re.M)
LOCK = threading.Lock()
COLUMNS = ["file", "source", "target", "episode", "old_solver", "old_outcome", "old_margin", "states",
           "exact_outcome", "exact_margin", "binding", "exact_values", "exact_seconds", "exact_how", "agrees", "error"]


def rules_for_run(run: Path, suffix: str, domain, inst):
    """{source: rules} of a run file's rule set, from the file name's suffix."""
    if suffix in ("_oneshot", "_unnarrowed"):
        data = json.loads((run / f"{suffix[1:]}.json").read_text(encoding="utf-8"))
        return {str(s): [tuple(r) for r in e["rules"]] for s, e in data.items()}
    key = {"": "final", "_determinized": "determinized", "_norepair_determinized": "norepair_determinized",
           "_norepair": "norepair", "_open": "open"}[suffix]
    lists = {str(s): r for s, r in RULES[key][0](run).items()}
    if key == "open":
        actions = list(domain.spec(next(iter(inst.values()))).actions)
        lists = {s: open_ties(r, actions) for s, r in lists.items()}
    return lists


def context(file: str):
    """(domain, cfg, instances by id, controller) for one result file: controller is a function target id ->
    {source: rules}, or ("explicit", baseline) for per-state policies."""
    p = Path(file)
    if p.name == "checks.csv":                       # sequential study: library lists on sequential_v0.csv
        cfg = load_config("port_fam_history", ["domain.dataset=sequential_v0.csv"])
        domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
        lib = {e["name"]: [tuple(r) for r in e["rules"]]
               for e in json.loads((p.parent / "library.json").read_text(encoding="utf-8"))}
        inst = {int(i.id): i for i in domain.load_instances("sequential_v0.csv")}
        return domain, cfg, inst, lambda t: lib
    m = re.match(r"transfer_(exact|sealed)_(.+)\.csv$", p.name)
    stem, suffix = m.group(2), ""
    for known in ("grid_20_large", "grid_test_large", "families_v0", "pacman_transfer", "mer_transfer",
                  "mer_n_transfer"):
        if stem.startswith(known):
            stem, suffix = known, stem[len(known):]
    dataset = f"{stem}.csv"
    if "baselines" in p.parts:
        settings = load_file(Path("configs/baselines") / f"{p.parent.name}.yaml", BaselineConfig, [])
        cfg = load_config(settings.condition)
        domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
        base = load_method(settings.method)(domain, settings.params)
        base.config = cfg
        nets = p.parent / "nets.pt"
        if nets.exists():
            for k, v in recertify.torch_load(nets).items():
                setattr(base, k, v)
        else:
            train_inst = {int(i.id): i for i in domain.load_instances(settings.dataset)}
            base.fit([train_inst[i] for i in parse_ids(settings.train)] if settings.train else [])
        inst = {int(i.id): i for i in domain.load_instances(dataset)}
        if base.explicit:       # per-state policies (networks): checked on the target's explicit model
            return domain, cfg, inst, ("explicit", base)
        actions = list(domain.spec(next(iter(inst.values()))).actions)
        opened = suffix == "_open"
        return domain, cfg, inst, lambda t: {str(s): (open_ties(r, actions) if opened else r)
                                             for s, r in base.rules(inst[t]).items()}
    run = p.parent
    cfg = load_run_config(run)
    if cfg.rules.early_narrow:
        cfg = replace(cfg, rules=replace(cfg.rules, shared_literals=True))
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    inst = {int(i.id): i for i in domain.load_instances(dataset)}
    lists = rules_for_run(run, suffix, domain, inst)
    return domain, cfg, inst, lambda t: lists


def run_exact(cmd_head, props_text, n, timeout, max_concurrent, tmp):
    """PRISM's exact engine; the n results as Fractions."""
    (Path(tmp) / "p.props").write_text(props_text, encoding="utf-8")
    cmd = (*cmd_head, str(Path(tmp) / "p.props"), "-exact", "-javamaxmem", "2g")
    out = run_call(PrismCall(cmd, timeout, max_concurrent))
    values = RAT.findall(out)
    if len(values) != n:
        raise RuntimeError((re.findall(r"(Error[^\n]*|OutOfMemory[^\n]*)", out) or [out[-300:]])[-1])
    return [Fraction(x) for x in values]


def product_check(prism_path, reqs, tra, lab, timeout, max_concurrent, tmp, order=None, stop_on_fail=False):
    """{requirement name: worst case}, exactly, through the monitor product of the explicit model (ltl_exact), one
    product per requirement, in `order` (names). `stop_on_fail`: stop at the first requirement exactly violated, which
    alone settles a failed verdict."""
    if any(r.reward is not None for r in reqs):
        raise RuntimeError("reward requirement: the product path covers probabilities only")
    choices, init, labels = read_explicit(Path(tra), Path(lab))
    by_name = {r.name: r for r in reqs}
    out = {}
    for name in order or list(by_name):
        r = by_name[name]
        text, props, finish, _ = product_model(choices, init, labels, [r])
        if not props[0]:
            out[name] = finish[0]
        else:
            (Path(tmp) / "prod.prism").write_text(text, encoding="utf-8")
            out[name] = finish[0](run_exact((prism_path, str(Path(tmp) / "prod.prism")), props[0] + ";\n", 1,
                                            timeout, max_concurrent, tmp)[0])
        if stop_on_fail and not exact_holds(r, out[name]):
            break
    return out


def exact_holds(r, value) -> bool:
    thr = Fraction(str(r.threshold))
    return value >= thr if r.maximize else value <= thr


def exact_check(domain, cfg, instance, rules, timeout, **how):
    """Each requirement's worst case of `rules` on `instance` by PRISM's exact engine, as Fractions, on the composed
    model `recertify.check_policy` checks (with its properties, where the exact engine takes them)."""
    runner = PrismRunner(cfg.prism)
    v = PolicyVerifier(domain, instance, cfg, runner)
    spec = v.spec
    policy = spec.policy(rules, spec.vocabulary(cfg.rules))
    with tempfile.TemporaryDirectory(prefix="exact_") as tmp:
        model = Path(tmp) / "m.prism"
        model.write_text(v.compose(policy), encoding="utf-8")
        if cfg.domain.name != "gridworld":
            props = "".join(q + ";\n" for q in v._properties([lambda r: r.worst_op()], vectors=False))
            values = run_exact((runner.prism_path, str(model)), props, len(spec.requirements), timeout,
                               cfg.prism.max_concurrent, tmp)
            return spec, {r.name: v for r, v in zip(spec.requirements, values)}, "exact engine"
        out = run_call(PrismCall((runner.prism_path, str(model), "-exportmodel", str(Path(tmp) / "out.tra,lab"),
                                  "-exportmodelprecision", "17", "-javamaxmem", "2g"), timeout,
                                 cfg.prism.max_concurrent))
        if not (Path(tmp) / "out.tra").exists():
            raise RuntimeError("export failed: " + out[-300:])
        return spec, product_check(runner.prism_path, spec.requirements, Path(tmp) / "out.tra",
                                   Path(tmp) / "out.lab", timeout, cfg.prism.max_concurrent, tmp, **how), "monitor product"


def exact_explicit(domain, cfg, instance, base, source, timeout, **how):
    """As exact_check for a per-state policy: the target's explicit model under it (as recertify.certify_explicit)."""
    from core.explicit import ExplicitMDP
    mdp = ExplicitMDP(domain, instance, cfg)
    reqs = mdp.spec.requirements
    model = mdp.under(base.policies(mdp)[source])
    model = model.part(model.reachable())
    if model.S == 1:
        raise RuntimeError("one-state model: decided exactly in numpy already")
    names = sorted({n for r in reqs for n in re.findall(r'"([^"]+)"', r.formula)})
    with tempfile.TemporaryDirectory(prefix="exact_") as tmp:
        model.write(Path(tmp) / "model", names)
        return mdp.spec, product_check(mdp.runner.prism_path, reqs, Path(tmp) / "model.tra",
                                       Path(tmp) / "model.lab", timeout, cfg.prism.max_concurrent, tmp, **how), \
            "explicit model, monitor product"


def verdict(spec, values):
    """Exact margins (Requirement.margin in Fractions) of the requirements computed and the verdict: failed if one is
    violated, certified if every requirement was computed and holds."""
    margins = {}
    for r in spec.requirements:
        if r.name not in values:
            continue
        thr, val = Fraction(str(r.threshold)), values[r.name]
        gap = val - thr if r.maximize else thr - val
        margins[r.name] = gap / (abs(thr) or 1) if r.reward is not None else gap
    binding = min(margins, key=margins.get)
    if margins[binding] < 0:
        return "failed", margins[binding], binding
    if len(margins) < len(spec.requirements):
        raise RuntimeError("not every requirement was computed")
    return "certified", margins[binding], binding


def old_worst(file: str) -> dict:
    """(source, target, episode) -> {requirement: worst case} of a result file's own checks (worst_* columns)."""
    df = pd.read_csv(file)
    src, tgt = ("rule_list", "instance") if "rule_list" in df else ("source", "target")
    ep = df["episode"] if "episode" in df else pd.Series(-1, index=df.index)
    cols = [c for c in df if c.startswith("worst_")]
    return {(str(s), int(t), int(e) if pd.notna(e) else -1): {c[6:]: row[c] for c in cols if pd.notna(row[c])}
            for s, t, e, (_, row) in zip(df[src], df[tgt], ep, df.iterrows())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fallback"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    todo = pd.read_csv(a.fallback)
    if a.only:
        todo = todo[todo.file.str.contains(a.only)]
    out = Path(a.out)
    key = lambda f, s, t, e: (f, str(s), int(t), int(e) if pd.notna(e) else -1)
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        prev = prev[prev.exact_outcome != "error"]
        done = {key(*r) for r in zip(prev.file, prev.source, prev.target, prev.episode)}
    todo = todo[[key(f, s, t, e) not in done for f, s, t, e in zip(todo.file, todo.source, todo.target, todo.episode)]]
    contexts, locks = {}, {f: threading.Lock() for f in todo.file.unique()}

    def ctx(file):
        with locks[file]:
            if file not in contexts:
                contexts[file] = (*context(file), old_worst(file))
            return contexts[file]

    def one(row):
        file = row.file
        res = dict.fromkeys(COLUMNS, "")
        res.update(file=file, source=str(row.source), target=int(row.target), episode=row.episode,
                   old_solver=row.solver, old_outcome=row.outcome, old_margin=row.margin, states=row.states)
        start = time.time()
        try:
            domain, cfg, inst, controller, worst = ctx(file)
            spec = domain.spec(inst[int(row.target)])
            prior = worst.get(key(file, row.source, row.target, row.episode)[1:], {})
            margin = lambda r: r.margin(prior[r.name]) if r.name in prior else 0.0
            how = {"order": [r.name for r in sorted(spec.requirements, key=margin)],
                   "stop_on_fail": row.outcome == "failed"}     # one exact violation settles a failed verdict
            if isinstance(controller, tuple):
                spec, values, kind = exact_explicit(domain, cfg, inst[int(row.target)], controller[1],
                                                    str(row.source), a.timeout, **how)
            else:
                spec, values, kind = exact_check(domain, cfg, inst[int(row.target)],
                                                 controller(int(row.target))[str(row.source)], a.timeout, **how)
            outcome, m, binding = verdict(spec, values)
            res.update(exact_outcome=outcome, exact_margin=float(m), binding=binding,
                       exact_values=json.dumps({k: float(v) for k, v in values.items()}),
                       exact_how=kind + ("" if len(values) == len(spec.requirements) else ", stopped at a violation"),
                       agrees=outcome == row.outcome)
        except Exception as e:     # recorded, never silently dropped
            res.update(exact_outcome="error", error=f"{type(e).__name__}: {str(e)[:300]}")
        res["exact_seconds"] = time.time() - start
        with LOCK:
            pd.DataFrame([res], columns=COLUMNS).to_csv(out, mode="a", header=not out.exists(), index=False)
            if res["exact_outcome"] == "error" or not res["agrees"]:
                print(f"  {file} {res['source']} on {res['target']}: was {res['old_outcome']}, exact "
                      f"{res['exact_outcome']} {res['exact_margin']} {res['error']}", flush=True)
        return res

    print(f"{len(todo)} checks to do", flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(one, todo.itertuples(index=False)))
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
