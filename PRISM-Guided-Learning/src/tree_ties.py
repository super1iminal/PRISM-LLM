"""How much 1-2-3-Go!'s tree depends on the choice among optimal actions (plan 02's sanity check against the paper).

Trains trees over the raw variables on one instance's decision states with different training labels: every optimal
action with weight 1/k (our baseline, with and without the states where every action is optimal), and one optimal
action per state, the first in each of the orders of the actions (what a solver's tie-breaking gives, e.g. Storm's
scheduler, which 1-2-3-Go! learns from). Prints each tree's worst-case value on the targets (numpy, explicit model)
as a markdown table. A diagnostic, not a baseline.

Usage (from PRISM-Guided-Learning/):
    python src/tree_ties.py --condition pacman_baseline --data pacman_transfer.csv --train 8 --targets 0,2,7
"""
import argparse
import itertools

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from baselines.decision_tree import Column, tree_leaves
from baselines.explicit_mdp import ExplicitMDP
from config import load_config
from core.domain import load_domain
from transfer import guard_sealed, parse_ids


def label_sets(mdp: ExplicitMDP, optimal: np.ndarray):
    """Name -> (states, actions, weights) of each way of labelling the decision states with optimal actions."""
    dec = np.flatnonzero(mdp.decision)
    some = dec[~optimal[dec].all(axis=1)]
    out = {}
    for name, states in (("all optimal actions, 1/k each, all-optimal states left out (the baseline)", some),
                         ("all optimal actions, 1/k each, every decision state", dec)):
        rows, acts = np.nonzero(optimal[states])
        out[name] = (states[rows], acts, 1.0 / optimal[states].sum(axis=1)[rows])
    for order in itertools.permutations(range(len(mdp.actions))):
        first = np.array([next(a for a in order if optimal[s, a]) for s in dec])
        out["first optimal in order " + ", ".join(mdp.actions[a] for a in order)] = (dec, first, None)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--condition", default="pacman_baseline")
    parser.add_argument("--data", default="pacman_transfer.csv")
    parser.add_argument("--train", type=int, default=8, help="id of the training instance")
    parser.add_argument("--targets", default="0,2,7")
    args = parser.parse_args(argv)
    guard_sealed(args.data, parse_ids(args.targets))
    cfg = load_config(args.condition)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    instances = {int(i.id): i for i in domain.load_instances(args.data)}
    source = ExplicitMDP(domain, instances[args.train], cfg)
    req = source.spec.requirements[0]
    _, optimal = source.optimal_actions(req)
    columns = [Column(v.name, None, v.type == "bool") for v in source.spec.variables]
    targets = {i: ExplicitMDP(domain, instances[i], cfg) for i in parse_ids(args.targets)}
    names = [instances[i].data.get("name", i) for i in targets]
    print(f"| training labels | leaves | {' | '.join(f'{req.name} on {n}' for n in names)} | tree |")
    print("|---|---|" + "---|" * len(names) + "---|")
    for label, (states, actions, weights) in label_sets(source, optimal).items():
        tree = DecisionTreeClassifier(criterion="gini", random_state=0).fit(source.obs[states], actions,
                                                                            sample_weight=weights)
        values = []
        for mdp in targets.values():
            policy = np.full(mdp.S, -1)
            dec = np.flatnonzero(mdp.decision)
            policy[dec] = tree.predict(mdp.obs[dec])
            values.append(mdp.evaluate(policy, names=[req.name])[req.name])
        rules = tree_leaves(tree, columns, source.actions, source.spec)
        text = "; ".join(f"{c} -> {a}" for c, a in rules) if len(rules) <= 5 else f"{len(rules)} rules"
        print(f"| {label} | {tree.get_n_leaves()} | {' | '.join(f'{v:.4f}' for v in values)} | {text} |", flush=True)


if __name__ == "__main__":
    main()
