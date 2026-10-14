"""1-2-3-Go!-style decision trees (Azeem et al., VMCAI 2025), reimplemented on the gridworld: optimal strategies of
the training instances for one reachability objective, a decision tree learned from them, frozen, and applied to
every target. Gridworld only.

As in the paper: the objective is a single reachability goal (here: every goal in order, no later goal entered
early, no collision with the moving obstacle; `gridworld_mdp.optimal_actions`), the training data are the
reachable states of the training instances with an optimal action each (the first optimal one in the order
up, right, down, left; the paper keeps all optimal actions and takes a majority vote per leaf), and the tree is
CART with the Gini index, grown until its leaves are pure (`max_depth: null`). Not as in the paper: sklearn's CART,
not dtControl, and value iteration in numpy, not Storm.

On a target, the tree picks an action in every state; that policy is written as rules over x, y, the goal flags
and obs_idx (one per goal stage, obstacle phase and action, listing its cells) and certified like any other.

params:
  inputs: raw          raw: the state variables (x, y, goal flags, obs_idx), as in the paper;
                       features: the rule-language features of all four actions plus the goal flags
  max_depth: null      tree depth limit (null: grow until pure)
  sets: null           training sets as lists of ids, one tree each; null: one tree from all training ids
  features: null       with inputs: features, the features to use (null: all of gridworld_mdp.FEATURES)
"""
from typing import Dict, List, Sequence

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from baselines.base import Baseline, Rules
from baselines.gridworld_mdp import ACTIONS, FEATURES, GridMDP, optimal_actions
from core.domain import Instance


def reachable(mdp: GridMDP) -> np.ndarray:
    """States reachable from the initial state under some policy."""
    seen = np.zeros(mdp.S, dtype=bool)
    seen[mdp.initial] = True
    frontier = np.array([mdp.initial])
    while frontier.size:
        nxt = np.unique(mdp.succ[frontier].ravel())
        frontier = nxt[~seen[nxt]]
        seen[frontier] = True
    return seen


def inputs(mdp: GridMDP, kind: str, features: Sequence[str] = FEATURES) -> np.ndarray:
    """One row per state: the tree's input variables."""
    stage = np.array([s for (_, _, s, _) in mdp.states])
    flags = np.stack([stage > k for k in range(mdp.K)], axis=1).astype(int)
    if kind == "raw":
        pos = np.array([(r, c, o) for (r, c, _, o) in mdp.states])
        return np.hstack([pos[:, :2], flags, pos[:, 2:]])
    if kind == "features":
        return np.hstack([mdp.features[f].astype(int) for f in features] + [flags])
    raise ValueError(f"unknown tree inputs {kind!r} (raw | features)")


def policy_rules(mdp: GridMDP, actions: np.ndarray) -> Rules:
    """A deterministic policy as rules: one per goal stage, obstacle phase and action, over the cells."""
    groups: Dict[tuple, List[str]] = {}
    for (r, c, s, o), a in zip(mdp.states, actions):
        groups.setdefault((s, o, int(a)), []).append(f"(x = {r} & y = {c})")
    rules = []
    for (s, o, a), cells in sorted(groups.items()):
        stage = " & ".join([f"g{k + 1}" for k in range(s)] + ([f"!g{s + 1}"] if s < mdp.K else []))
        phase = f"obs_idx = {o}" if mdp.cycle else ""
        cond = " & ".join(p for p in (stage, phase, f"({' | '.join(cells)})") if p)
        rules.append((cond, ACTIONS[a]))
    return rules + [("true", ACTIONS[0])]


class DecisionTree(Baseline):

    def __init__(self, domain, params):
        super().__init__(domain, params)
        self.kind = params.get("inputs", "raw")
        self.max_depth = params.get("max_depth")
        self.sets = params.get("sets")
        self.features = params.get("features") or list(FEATURES)
        self.tree = None
        self.trees: Dict[str, DecisionTreeClassifier] = {}

    def fit(self, train: Sequence[Instance]) -> None:
        by_id = {int(i.id): i for i in train}
        for k, ids in enumerate(self.sets or [sorted(by_id)]):
            self.tree = self._fit([by_id[i] for i in ids])
            self.trees[f"tree_{self.kind}" + (f"_set{k}" if self.sets else "")] = self.tree

    def _fit(self, train: Sequence[Instance]) -> DecisionTreeClassifier:
        tree = DecisionTreeClassifier(criterion="gini", max_depth=self.max_depth, random_state=0)
        X, y = [], []
        for inst in train:
            mdp = GridMDP(inst.data)
            value, optimal = optimal_actions(mdp)
            keep = reachable(mdp) & (value > 0) & ~optimal.all(axis=1)
            X.append(inputs(mdp, self.kind, self.features)[keep])
            y.append(optimal[keep].argmax(axis=1))
        tree.fit(np.vstack(X), np.concatenate(y))
        print(f"decision tree ({self.kind}): {tree.tree_.node_count} nodes, depth {tree.get_depth()}, "
              f"{sum(len(v) for v in y)} training states", flush=True)
        return tree

    def actions(self, target: Instance) -> np.ndarray:
        """The last fitted tree's action in every state of `target` (for the fast model)."""
        mdp = GridMDP(target.data)
        return self.tree.predict(inputs(mdp, self.kind, self.features))

    def rules(self, target: Instance) -> Dict[str, Rules]:
        mdp = GridMDP(target.data)
        x = inputs(mdp, self.kind, self.features)
        return {name: policy_rules(mdp, tree.predict(x)) for name, tree in self.trees.items()}
