"""Narrow a run's frozen rule sets as `rules.early_narrow` does in the loop, by verification on their own training
instances: when a rule set passes every training member, keep the moves that lose most early in the training games
(core/early.py) away from the places where they lose in at least 3/4 of the checked situations (`& !(place)` on every
rule choosing the move), each only if every member still passes and the early loss drops; up to k narrowings. Only
the training instances are used, never the targets.

Usage (from PRISM-Guided-Learning/):
    python src/narrow.py <run dir> [--k 3] [--clock steps] [--group xP,yP] [--rules determinized]
(`--rules determinized`: the lists with `any` resolved, as transfer.py's; `any` rules choose no single move to narrow.)
Writes <run dir>/narrowed.json (per sample id: the rules, the narrowings and the early loss before and after); then
`python src/transfer.py <run dir> --rules narrowed` certifies them on targets (numbers allowed in comparisons with
the policy variables, as `rules.shared_literals`).
"""
import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from config import load_run_config
from core import early
from core.domain import load_domain
from core.planner import NARROW_SHARE
from core.prism import PrismError, PrismRunner
from core.rules import RuleError
from core.verifier import PolicyVerifier
from transfer import RULES, training_instances

OUT = "narrowed.json"
Rules = List[Tuple[str, str]]


def passes(verifiers: Sequence[PolicyVerifier], rules_cfg, shared, rules: Rules) -> Optional[list]:
    """Each member's policy for `rules` if they pass every member in the worst case, else None."""
    policies = []
    for v in verifiers:
        try:
            policy = v.spec.policy(rules, v.spec.vocabulary(rules_cfg, shared))
            worst = v.verify(policy, analysis=False).worst
        except (RuleError, PrismError, subprocess.TimeoutExpired):
            return None
        if any(not r.satisfied(worst[r.name]) for r in v.spec.requirements):
            return None
        policies.append(policy)
    return policies


def narrow(verifiers: Sequence[PolicyVerifier], rules_cfg, rules: Rules, k: int, clock: str, group: Sequence[str],
           log=print) -> Tuple[Rules, dict]:
    """The narrowed rules and what was done (`rules` unchanged if they fail a member)."""
    shared = [v.name for v in verifiers[0].spec.variables]
    policies = passes(verifiers, rules_cfg, shared, rules)
    if policies is None:
        return rules, {"passes": False, "narrowed": []}
    checks = [early.EarlyCheck(v, clock, f"instance {v.instance.id}") for v in verifiers]
    result = early.check(list(zip(checks, policies)), group=group)
    info = {"passes": True, "narrowed": [], "early_before": result.mean}
    for _ in range(k):
        best = None
        for g in [g for g in result.rules if g.move and g.place and g.losing >= NARROW_SHARE * g.decided][:k]:
            place = " & ".join(f"{name} = {value}" for name, value in g.place)
            variant = [(f"({c}) & !({place})" if a == g.move else c, a) for c, a in rules]
            ps = passes(verifiers, rules_cfg, shared, variant)
            if ps is None:
                continue
            r = early.check(list(zip(checks, ps)), group=group)
            if r.mean < (best[1].mean if best else result.mean) - 1e-12:
                best = (variant, r, f"{g.move} kept away from {place}")
        if best is None:
            break
        rules, result = best[0], best[1]
        info["narrowed"].append(best[2])
        log(f"  {best[2]}: early loss {result.mean:.4f}")
    info["early_after"] = result.mean
    return rules, info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--k", type=int, default=3, help="Narrowings at most")
    parser.add_argument("--clock", default="steps", help="The step counter (feedback.early)")
    parser.add_argument("--group", default="xP,yP", help="The variables naming a place (feedback.early_group)")
    parser.add_argument("--rules", default="final", choices=["final", "determinized"], help="The lists to narrow")
    args = parser.parse_args(argv)
    cfg = load_run_config(args.run_dir)
    rules_cfg = replace(cfg.rules, shared_literals=True)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    instances = {int(i.id): i for i in domain.load_instances(cfg.domain.dataset)}
    runner = PrismRunner(cfg.prism)
    trained = training_instances(args.run_dir)
    out = {}
    for sid, rules in RULES[args.rules][0](args.run_dir).items():
        verifiers = [PolicyVerifier(domain, instances[i], cfg, runner) for i in trained[sid]]
        print(f"sample {sid}: {len(rules)} rules", flush=True)
        narrowed, info = narrow(verifiers, rules_cfg, list(rules), args.k, args.clock, args.group.split(","),
                                log=lambda s: print(s, flush=True))
        out[str(sid)] = {"rules": [list(r) for r in narrowed], "from": args.rules, **info}
    (args.run_dir / OUT).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(args.run_dir / OUT)


if __name__ == "__main__":
    main()
