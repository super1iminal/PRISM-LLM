"""Resolve the `any` rules of a run's frozen rule sets into a fixed action order, chosen by verification on each
rule set's own training instances.

An `any` rule allows every action whose condition holds, and the worst case lets the adversary pick among them. A
fixed order instead takes the first allowed action: `c -> any` becomes `c -> a1`, `c -> a2`, ... for an order
(a1, a2, ...) of the domain's actions. The result is still a rule list in the language, so it is certified like
any other; only the training instances are used to choose the order, never the targets.

Usage (from PRISM-Guided-Learning/):
    python src/determinize.py <run dir> [--orders all|rotations] [--workers N]
Writes <run dir>/determinized.json: per sample id, the chosen order, the rules, and each training member's
requirements met. Rule sets without `any` rules are kept as they are. Then
`python src/transfer.py <run dir> --rules determinized` certifies them on targets.
"""
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from config import load_run_config
from core.domain import load_domain
from core.prism import PrismError, PrismRunner
from core.rules import RuleError
from core.verifier import PolicyVerifier
from core.rules import action_orders, fixed_order
from transfer import final_rules, training_instances

OUT = "determinized.json"


def score(verifiers: List[PolicyVerifier], cfg, rules) -> Tuple[Tuple[int, int, float], List[int]]:
    """(members passing, requirements met, -shortfall) of `rules` on the training members, and met per member."""
    passed = met = 0
    shortfall = 0.0
    per_member = []
    for verifier in verifiers:
        spec = verifier.spec
        try:
            policy = spec.policy(rules, spec.vocabulary(cfg.rules))
        except RuleError:
            return (-1, -1, float("-inf")), []
        try:
            worst = verifier.verify(policy, analysis=False).worst
        except (PrismError, subprocess.TimeoutExpired):    # undecided: as bad as rules that do not parse
            return (-1, -1, float("-inf")), []
        ok = [r.satisfied(worst[r.name]) for r in spec.requirements]
        passed += all(ok)
        met += sum(ok)
        shortfall += sum(r.shortfall(worst[r.name]) for r in spec.requirements)
        per_member.append(sum(ok))
    return (passed, met, -round(shortfall, 9)), per_member


def determinize(run_dir: Path, kind: str, workers: int) -> Dict[int, dict]:
    """Per sample: the best order on its training instances and the rules it gives."""
    cfg = load_run_config(run_dir)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    instances = {int(i.id): i for i in domain.load_instances(cfg.domain.dataset)}
    runner = PrismRunner(cfg.prism)
    trained = training_instances(run_dir)
    out = {}
    for sid, rules in final_rules(run_dir).items():
        verifiers = [PolicyVerifier(domain, instances[i], cfg, runner) for i in trained[sid]]
        if not any(a == "any" for _, a in rules):
            out[sid] = {"order": None, "rules": [list(r) for r in rules], "train_met": score(verifiers, cfg, rules)[1]}
            continue
        candidates = action_orders(list(verifiers[0].spec.actions), kind)
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(lambda o: score(verifiers, cfg, fixed_order(rules, o)), candidates))
        best = max(range(len(candidates)), key=lambda k: results[k][0])   # first of the best, in candidate order
        out[sid] = {"order": list(candidates[best]), "rules": [list(r) for r in fixed_order(rules, candidates[best])],
                    "train_met": results[best][1], "train_score": list(results[best][0])}
        print(f"sample {sid}: order {'-'.join(candidates[best])}, training {results[best][0]}", flush=True)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--orders", choices=["all", "rotations"], default="all")
    parser.add_argument("--workers", type=int, default=4, help="Orders checked at once")
    args = parser.parse_args(argv)
    result = determinize(args.run_dir, args.orders, args.workers)
    (args.run_dir / OUT).write_text(json.dumps({str(k): v for k, v in result.items()}, indent=1), encoding="utf-8")
    print(args.run_dir / OUT)


if __name__ == "__main__":
    main()
