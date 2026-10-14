"""Polish a run's frozen rule sets by verification on their own training instances: greedily delete rules, and
conditions of rules that are plain conjunctions (`a & b & c`), keeping a deletion when the rules do no worse on the
training instances. Shorter rules that pass the same training instances may carry over better. Only the training
instances are used, never the targets.

Usage (from PRISM-Guided-Learning/):
    python src/polish.py <run dir> [--workers N]
Writes <run dir>/polished.json (per sample id: the rules, and how many rules and conditions were removed); then
`python src/transfer.py <run dir> --rules polished` certifies them on targets.
"""
import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

from config import load_run_config
from core.domain import load_domain
from core.prism import PrismRunner
from core.verifier import PolicyVerifier
from determinize import score
from transfer import final_rules, training_instances

OUT = "polished.json"
Rules = List[Tuple[str, str]]


def conjuncts(condition: str) -> Optional[List[str]]:
    """The parts of a condition that is a plain conjunction (no parentheses, `|` or `=>`); None otherwise."""
    if re.search(r"[()|]|=>", condition):
        return None
    parts = [p.strip() for p in condition.split("&")]
    return parts if len(parts) > 1 else None


def candidates(rules: Rules):
    """Every single deletion: a rule (never the last one), or one conjunct of a rule."""
    for i in range(len(rules) - 1):
        yield rules[:i] + rules[i + 1:], "rule"
    for i, (cond, action) in enumerate(rules):
        parts = conjuncts(cond)
        for j in range(len(parts or [])):
            yield rules[:i] + [(" & ".join(parts[:j] + parts[j + 1:]), action)] + rules[i + 1:], "condition"


def polish(verifiers: List[PolicyVerifier], cfg, rules: Rules, log=print) -> Tuple[Rules, dict]:
    """Greedy deletions until none keeps the training score (first improvement, repeated)."""
    best, _ = score(verifiers, cfg, rules)
    removed = {"rule": 0, "condition": 0}
    changed = True
    while changed:
        changed = False
        for cand, kind in candidates(rules):
            s, _ = score(verifiers, cfg, cand)
            if s >= best:                      # no worse on the training instances (higher is better)
                rules, best, changed = cand, s, True
                removed[kind] += 1
                log(f"  removed a {kind}: {len(rules)} rules, training {s}")
                break
    return rules, {**removed, "train_score": list(best)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    cfg = load_run_config(args.run_dir)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    instances = {int(i.id): i for i in domain.load_instances(cfg.domain.dataset)}
    runner = PrismRunner(cfg.prism)
    trained = training_instances(args.run_dir)
    out = {}
    for sid, rules in final_rules(args.run_dir).items():
        verifiers = [PolicyVerifier(domain, instances[i], cfg, runner) for i in trained[sid]]
        print(f"sample {sid}: {len(rules)} rules", flush=True)
        polished, info = polish(verifiers, cfg, list(rules), log=lambda s: print(s, flush=True))
        out[str(sid)] = {"rules": [list(r) for r in polished], **info}
    (args.run_dir / OUT).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(args.run_dir / OUT)


if __name__ == "__main__":
    main()
