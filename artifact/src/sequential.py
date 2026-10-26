"""The sequential adaptation study: an adaptation manager meets a fixed sequence of instances, one episode each, and
before every episode decides which rule list controls it (the policy of the paper's Approach, Sec. IV-A):

1. retain the current rule list if it is certified on the instance;
2. otherwise switch to the certified rule list of the library with the largest smallest margin;
3. otherwise (mode `growing` only) generate a new rule list for the instance together with the training instances
   of the library's rule list that came closest to being certified on it, and deploy and store it if it is
   certified on the instance;
4. otherwise (study key `fallback: synthesize`, the default) synthesize a strategy for the instance alone
   (`synthesize`) and deploy it if it is certified: it serves this episode only, so the current rule list and the
   library stay as they were;
5. otherwise adaptation fails and the episode is deferred.

Mode `frozen` never generates, so it measures reuse alone (with synthesis as its only other way to serve an episode).
Synthesis needs a strategy to deploy. With one requirement it takes an optimal action of that requirement in every
decision state of the instance's explicit model (`ExplicitMDP.optimal_actions`, as the strategy-of-one-instance
baseline) and certifies that per-state strategy like any controller (`recertify.certify_explicit`: the exact solvers
and the certification rule below); it must depend only on the observable variables. With several requirements PRISM's
multi-objective query (`PolicyVerifier.joint_query`, exact LP) decides whether one strategy meets every threshold at
once, but PRISM cannot export that strategy: the episode records whether one exists and is deferred. A synthesized
strategy is not stored: the same computation yields it again. Certification uses the exact solvers
(`PolicyVerifier.verify_exact`) and the certification rule (`core.verifier.exact_verdict`): a rule list is certified
only if an exact solver (`prism.exact_methods`: interval iteration, then linear programming where it stalls)
finished and every worst case lies beyond its threshold by more than its band (`prism.exact_epsilon` for interval
iteration, `prism.lp_tolerance` for linear programming). A value within that band, or a value from the Gauss-Seidel fallback (no error bound), is inconclusive and
counts as not certified; failure, inconclusive results and timeouts are recorded separately. Generation runs the study's condition through src/run_ablation.py (LLM calls),
then src/determinize.py fixes the order of its `any` rules, as for the library's seed rule lists.

Usage (from artifact/, with the OpenRouter key in the environment: `set -a && . ../.env && set +a`):
    python src/sequential.py configs/sequential/<study>.yaml --mode frozen|growing [--only s1 s2] [--dry-run]
Writes out/results/sequential/<study>/<mode>/<sequence>/: episodes.csv (one row per episode: the decision, the
rule list deployed, certification and generation times, LLM tokens), checks.csv (one row per certification),
library.json (the library after the sequence), and gen_t<episode>/ (each generation run).
"""
import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from config import load_config
from core.domain import load_domain
from core.prism import PrismError, PrismRunner
from core.rules import RuleError
from core.verifier import PolicyVerifier, exact_verdict
from recertify import certify_explicit
from settings import PROJECT_ROOT
from transfer import determinized_rules, guard_sealed, training_instances

PYTHON = sys.executable
OUT_ROOT = PROJECT_ROOT / "out" / "results" / "sequential"
MODEL_TIME = re.compile(r"Time for model construction:\s*([\d.]+)")
FALLBACKS = ("synthesize", "defer")
Rules = List[Tuple[str, str]]


@dataclass
class Sequence:
    name: str
    ids: List[int]


@dataclass
class Study:
    """configs/sequential/<study>.yaml documents each key."""
    name: str
    condition: str                       # run condition for generation (configs/run/conditions/)
    dataset: str                         # every instance id below refers to this dataset
    seed_run: Union[str, List[str]]      # run dir(s) whose determinized rule lists seed the library
    sequences: List[Sequence]
    generation_set: List[str] = field(default_factory=list)   # extra overrides for generation runs
    max_training: int = 4                # training instances per generation run, the instance met included
    fallback: str = "synthesize"         # when no rule list serves the episode: synthesize | defer

    def __post_init__(self):
        if self.fallback not in FALLBACKS:
            raise ValueError(f"fallback must be one of {FALLBACKS}, not {self.fallback!r}")


@dataclass
class Entry:
    """A rule list in the library: its name, rules, and the instances it was written for."""
    name: str
    rules: Rules
    training: List[int]


def load_study(path: Path) -> Study:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["sequences"] = [Sequence(**s) for s in data["sequences"]]
    return Study(**data)


def seed_library(study: Study) -> List[Entry]:
    """The library's seed entries: every determinized rule list of the seed run, named seed<sample id>; with several
    seed runs (a list), every list of each, named seed<k>.<sample id> for the k-th run (1-based)."""
    runs = [study.seed_run] if isinstance(study.seed_run, str) else list(study.seed_run)
    entries = []
    for k, run in enumerate(runs, 1):
        run_dir = PROJECT_ROOT / run
        rules, training = determinized_rules(run_dir), training_instances(run_dir)
        prefix = "seed" if len(runs) == 1 else f"seed{k}."
        entries += [Entry(f"{prefix}{sid}", rules[sid], training[sid]) for sid in sorted(rules)]
    return entries


def certify(domain, cfg, runner, instance, entry: Entry) -> dict:
    """Certify one rule list on one instance with the exact solvers; one row of checks.csv."""
    row = {"instance": int(instance.id), "rule_list": entry.name}
    start = time.time()
    verifier = PolicyVerifier(domain, instance, cfg, runner)
    spec = verifier.spec
    try:
        policy = spec.policy(entry.rules, spec.vocabulary(cfg.rules))
    except RuleError as e:
        return {**row, "outcome": "unparsed", "error": str(e).splitlines()[0], "seconds": time.time() - start}
    try:
        v, solver = verifier.verify_exact(policy)
    except (PrismError, subprocess.TimeoutExpired) as e:
        v, solver = None, None
        row["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    seconds = time.time() - start
    if v is None:
        return {**row, "outcome": "timeout", "seconds": seconds}
    margins = {r.name: r.margin(v.worst[r.name]) for r in spec.requirements}
    smallest = min(margins.values())
    outcome = exact_verdict(margins, solver, cfg.prism)
    model_seconds = sum(float(x) for x in MODEL_TIME.findall(getattr(v.result, "stdout", "") or ""))
    return {**row, "outcome": outcome, "solver": solver, "margin": smallest, "seconds": seconds,
            "model_seconds": model_seconds, "states": len(v.result.states),
            **{f"worst_{k}": v.worst[k] for k in margins}}


def synthesize(domain, cfg, runner, instance) -> dict:
    """Synthesize a strategy for `instance` alone (module docstring). `synthesis`: `synthesized` (certified, deployed for
    this episode), `failed`, `inconclusive` or `not_certifiable` (the certification's verdicts), `not_observable` (the
    optimal strategy tells apart states the controller cannot), or, with several requirements, `exists` / `none` /
    `inconclusive` (the joint query's answer; nothing to deploy)."""
    reqs = domain.spec(instance).requirements
    start = time.time()
    if len(reqs) > 1:
        feasible = PolicyVerifier(domain, instance, cfg, runner).jointly_feasible()
        return {"synthesis": {True: "exists", False: "none", None: "inconclusive"}[feasible],
                "synthesis_method": "joint query", "synthesis_seconds": time.time() - start}
    observable = {}

    def optimal(mdp) -> Dict[str, np.ndarray]:
        _, best = mdp.optimal_actions(mdp.spec.requirements[0])
        states = np.flatnonzero(mdp.decision)
        policy = np.full(mdp.S, -1, dtype=np.int64)
        policy[states] = best[states].argmax(axis=1)
        pairs = np.unique(np.column_stack([mdp.obs[states], policy[states]]), axis=0)
        observable["ok"] = len(pairs) == len(np.unique(mdp.obs[states], axis=0))   # one action per observation
        return {"synthesis": policy}

    check = certify_explicit(domain, cfg, instance, optimal)[0]
    seconds = time.time() - start
    if not observable.get("ok", True):
        return {"synthesis": "not_observable", "synthesis_method": "optimal actions", "synthesis_seconds": seconds}
    outcome = check["outcome"]
    return {"synthesis": "synthesized" if outcome == "certified" else outcome, "synthesis_method": check.get("solver"),
            "synthesis_value": check.get(f"worst_{reqs[0].name}"), "synthesis_states": check.get("states"),
            "synthesis_seconds": seconds}


def generate(study: Study, training: List[int], run_name: str, out_dir: Path) -> Tuple[Optional[Entry], dict]:
    """Run the study's condition on one training set, fix its `any` order, and return the new library entry and
    the generation's costs."""
    sets = "[[" + ", ".join(str(i) for i in training) + "]]"
    overrides = [f"run.train_sets={sets}", f"domain.dataset={study.dataset}"] + list(study.generation_set)
    cmd = [PYTHON, "src/run_ablation.py", study.condition, "--unseeded", "1", "--out-root", str(out_dir),
           "--name", run_name] + [x for o in overrides for x in ("--set", o)]
    run_dir = out_dir / run_name / "seed_none_1"
    log = out_dir / f"{run_name}.log"
    start = time.time()
    with open(log, "a", encoding="utf-8") as out:
        ok = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=out, stderr=subprocess.STDOUT).returncode == 0
        ok = ok and subprocess.run([PYTHON, "src/determinize.py", str(run_dir)], cwd=PROJECT_ROOT, stdout=out,
                                   stderr=subprocess.STDOUT).returncode == 0
    costs = {"generation_seconds": time.time() - start, **llm_costs(run_dir), **loop_times(run_dir)}
    if not ok or not (run_dir / "determinized.json").exists():
        return None, {**costs, "generation_error": f"see {log}"}
    rules = determinized_rules(run_dir)
    return Entry(run_name, rules[min(rules)], training), costs


def llm_costs(run_dir: Path) -> dict:
    """Prompt and output tokens over every LLM task of a run (<run dir>/llm_tasks.jsonl)."""
    prompt = output = calls = 0
    path = run_dir / "llm_tasks.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        result = json.loads(line).get("result") or {}
        prompt += result.get("prompt_tokens") or 0
        output += result.get("output_tokens") or 0
        calls += 1
    return {"llm_calls": calls, "prompt_tokens": prompt, "output_tokens": output}


def loop_times(run_dir: Path) -> dict:
    """LLM and PRISM seconds summed over every round of every chain, and the run's total (outputs/sample_*.json)."""
    llm = prism = total = 0.0
    for path in sorted((run_dir / "outputs").glob("sample_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        total += record.get("total_time") or 0.0
        for it in record.get("iterations", []):        # every round of every chain
            llm += it.get("llm_time") or 0.0
            prism += it.get("prism_time") or 0.0
    return {"loop_llm_seconds": llm, "loop_prism_seconds": prism, "loop_total_seconds": total}


def closest(checks: List[dict], library: List[Entry]) -> Optional[Entry]:
    """The library entry whose check came closest to certified (largest smallest margin)."""
    scored = [(c["margin"], c["rule_list"]) for c in checks if c.get("margin") is not None]
    if not scored:
        return None
    best = max(scored)[1]
    return next(e for e in library if e.name == best)


def run_sequence(study: Study, seq: Sequence, mode: str, cfg, domain, instances, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = PrismRunner(cfg.prism)
    library = seed_library(study)
    current: Optional[Entry] = None
    episodes, checks = [], []
    for t, iid in enumerate(seq.ids):
        instance = instances[iid]
        start = time.time()
        row = {"episode": t, "instance": iid, "family": instance.data.get("family"), "n": instance.data.get("n")}
        mine = []

        def check(entry: Entry) -> dict:
            c = {"episode": t, **certify(domain, cfg, runner, instance, entry)}
            mine.append(c)
            checks.append(c)
            return c

        decision, deployed = None, None
        if current is not None and check(current)["outcome"] == "certified":
            decision, deployed = "retain", current
        if decision is None:
            others = [e for e in library if e is not current]
            passing = [(c["margin"], e) for e in others for c in [check(e)] if c["outcome"] == "certified"]
            if passing:
                decision, deployed = "switch", max(passing, key=lambda p: p[0])[1]
        row["certify_seconds"] = sum(c.get("seconds", 0.0) for c in mine)
        row["model_seconds"] = sum(c.get("model_seconds", 0.0) for c in mine)
        row["checks"] = len(mine)
        if decision is None and mode == "growing":
            near = closest(mine, library)          # (`n` is None where the dataset has no size column, e.g. Pac-Man)
            base = sorted(near.training, key=lambda i: instances[i].data.get("n") or 0)[:study.max_training - 1] \
                if near else []
            training = base + [iid]
            entry, costs = generate(study, training, f"gen_t{t}", out_dir)
            row.update(costs, generated_from=near.name if near else None, training=json.dumps(training))
            if entry is not None:
                c = check(entry)
                row["new_outcome"] = c["outcome"]
                if c["outcome"] == "certified":
                    library.append(entry)
                    decision, deployed = "generate", entry
        if decision is None and study.fallback == "synthesize":
            row.update(synthesize(domain, cfg, runner, instance))
            if row["synthesis"] == "synthesized":
                decision = "synthesize"
        row.update(decision=decision or "defer", deployed=deployed.name if deployed else
                   "synthesis" if decision == "synthesize" else None,
                   library_size=len(library), seconds=time.time() - start)
        episodes.append(row)
        current = deployed or current
        print(f"[{seq.name} {mode}] episode {t}: instance {iid} ({row['family']}, n={row['n']}) -> {row['decision']}"
              f"{' ' + row['deployed'] if row['deployed'] else ''} ({row['checks']} checks, "
              f"{row['seconds']:.0f} s)", flush=True)
        pd.DataFrame(episodes).to_csv(out_dir / "episodes.csv", index=False)
        pd.DataFrame(checks).to_csv(out_dir / "checks.csv", index=False)
    (out_dir / "library.json").write_text(json.dumps(
        [{"name": e.name, "training": e.training, "rules": e.rules} for e in library], indent=1), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", type=Path, help="configs/sequential/<study>.yaml")
    parser.add_argument("--mode", choices=["frozen", "growing"], required=True)
    parser.add_argument("--only", nargs="+", default=None, help="Only these sequences")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and the seed library only")
    args = parser.parse_args(argv)
    study = load_study(args.study)
    cfg = load_config(study.condition, [f"domain.dataset={study.dataset}"] + list(study.generation_set))
    if cfg.rules.early_narrow:   # narrowings name places with numbers (as transfer.py and recertify.py); this config only
        cfg = replace(cfg, rules=replace(cfg.rules, shared_literals=True))   # certifies: generation has its own process
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    instances = {int(i.id): i for i in domain.load_instances(study.dataset)}
    rows = pd.read_csv(PROJECT_ROOT / "domains" / cfg.domain.name / "data" / study.dataset)
    for iid, inst in instances.items():                  # the domain keeps only what its model needs
        inst.data.setdefault("family", rows.family[iid] if "family" in rows else None)
        inst.data.setdefault("n", int(rows.n[iid]) if "n" in rows else None)
    sequences = [s for s in study.sequences if not args.only or s.name in args.only]
    for seq in sequences:
        guard_sealed(study.dataset, seq.ids)
        missing = [i for i in seq.ids if i not in instances]
        if missing:
            raise SystemExit(f"sequence {seq.name}: unknown instance ids {missing}")
    library = seed_library(study)
    print(f"study {study.name}, mode {args.mode}: seed library "
          + ", ".join(f"{e.name} ({len(e.rules)} rules, trained on {e.training})" for e in library))
    for seq in sequences:
        print(f"  {seq.name}: " + ", ".join(f"{i} ({instances[i].data.get('family')} {instances[i].data.get('n')})"
                                           for i in seq.ids))
    if args.dry_run:
        return
    for seq in sequences:
        run_sequence(study, seq, args.mode, cfg, domain, instances, OUT_ROOT / study.name / args.mode / seq.name)


if __name__ == "__main__":
    main()
