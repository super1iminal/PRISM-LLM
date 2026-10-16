"""Per chain of every gridworld chains run: the kept list's margin (run's own) and its numpy certification on its
group's targets; then best-of-K by margin within and across runs."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from order_cache import ROOT, TARGETS, evaluate, mdp_of
from baselines.gridworld_mdp import ACTIONS, RuleScorer

import numpy as np

from config import load_run_config
from core.domain import load_domain
from results_io import sample_records
from tie_analysis import allowed
from transfer import parse_ids

sys.path.insert(0, str(ROOT / "src"))
from chain_transfer import chain_rules

GROUPS = {0: "rooms", 1: "crossing", 2: "adjacent"}


def scorer_allow(domain, cfg, inst, rules, mdp):
    """The allowed actions of a list whose conditions are not feature literals, by the core rule evaluator."""
    spec = domain.spec(inst)
    pol = spec.policy(rules, spec.vocabulary(cfg.rules))
    scorer = RuleScorer(inst.data, spec.variables)
    allow = np.zeros((mdp.S, 4), dtype=bool)
    for i, state in enumerate(scorer.valuations):
        a = pol.allowed(state)
        if a is None:
            allow[i] = True
        else:
            allow[i, [ACTIONS.index(x) for x in a]] = True
    return allow


def run_chains(run: Path):
    cfg = load_run_config(run)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    instances = {int(i.id): i for i in domain.load_instances(cfg.domain.dataset)}
    out = []
    for sid, rec in sample_records(run).items():
        if "chains" not in rec:
            continue
        group = TARGETS[cfg.domain.dataset]
        targets = parse_ids(group[sid] if sid in group else group[None])
        margins = {c["chain"]: c["margin"] for c in rec["chains"]}
        for chain, rules in chain_rules(rec).items():
            cert = 0
            for t in targets:
                mdp, mons = mdp_of(cfg.domain.dataset, instances[t])
                reqs = {r.name: r for r in domain.spec(instances[t]).requirements}
                try:
                    allow = allowed(mdp, rules)
                except KeyError:
                    allow = scorer_allow(domain, cfg, instances[t], rules, mdp)
                cert += bool(evaluate(mdp, mons, allow, reqs, False))
            out.append({"run": run.name, "sid": sid, "chain": chain, "margin": margins[chain], "cert": cert,
                        "chosen": chain == rec["chain"],
                        "bench": ("families/" + GROUPS[sid]) if "fam" in run.parent.name else run.parent.name})
    return out


def main(conditions):
    rows = []
    for cond in conditions.split(","):
        for run in sorted((ROOT / "out/results/ablations" / cond).glob("seed_none_*")):
            if (run / "outputs").exists():
                rows += run_chains(run)
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main(sys.argv[1])


def summary(path):
    """Per benchmark: the chosen chain, the mean chain (one chain's expectation) and the best chain, per set-run."""
    import pandas as pd
    df = pd.DataFrame([json.loads(l) for l in open(path)])
    g = df.groupby(["bench", "run", "sid"])
    per = pd.DataFrame({"chosen": g.apply(lambda x: x[x.chosen].cert.iloc[0]), "mean": g.cert.mean(),
                        "best": g.cert.max()}).reset_index()
    print(per.groupby("bench")[["chosen", "mean", "best"]].mean().round(2).to_string())
    print(per.to_string())
