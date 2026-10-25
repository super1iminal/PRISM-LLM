"""1-2-3-Go!-style decision trees (Azeem et al., VMCAI 2025), reimplemented: optimal strategies of the training
instances for one reachability objective, a decision tree learned from them, frozen, and applied to every target.
This class is the gridworld version; `params.model: explicit` gives `ExplicitDecisionTree` (below), on any domain.

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
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from baselines.base import Baseline, Rules, named_sets
from baselines.gridworld_mdp import ACTIONS, FEATURES, GridMDP, optimal_actions
from config import load_config
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

    def __new__(cls, domain=None, params=None):
        if cls is DecisionTree and (params or {}).get("model") == "explicit":
            cls = ExplicitDecisionTree
        return super().__new__(cls)

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


MAX_LISTED = 1000       # fit.json lists a tree's rules (one per leaf) up to this many


class Column(NamedTuple):
    """A tree input: a policy variable (`action` None) or a rule-language feature (for `action`, if an action
    feature), and whether it is boolean."""
    name: str
    action: Optional[str]
    boolean: bool


class ExplicitDecisionTree(DecisionTree):
    """1-2-3-Go! on any domain (`params.model: explicit`), from the training instances' explicit models
    (core/explicit.py).

    Training data: every decision state of each training instance (PRISM's model holds only reachable states) with
    every optimal action for one requirement (`requirement`, default the first) over all schedulers, one sample
    per optimal action with weight 1/k: the paper's majority vote per leaf. States where every action is optimal
    constrain nothing and are left out, as in the gridworld version. CART with Gini, grown until pure.

    Inputs: `raw`, the policy-visible variables (as the paper: for Pac-Man xP, yP, dP, xG0, ..., steps); or
    `features`, the named rule-language features (`features`; null: all the spec's), an action feature once per
    action, computed by the core rule evaluator.

    The tree compiles exactly into rules: one per leaf, its path's bounds (`v <= t`, `v >= t + 1`, `f`, `!f`) and the
    leaf's action (`summary` lists them, up to `MAX_LISTED`). A feature of another action than the leaf's is written as the domain's
    expression of it for that action, since in a rule an action feature names the rule's own action. PRISM gets the
    same policy as one rule per action, the tree as an if-then-else (`nested`): with thousands of leaves, a rule per
    leaf would not compile, since the rule compiler negates every earlier rule in a rule's guard when the variable
    space is too large to enumerate. Numbers in conditions need a condition with `rules.extended: true,
    general: false` (configs/run/conditions/pacman_baseline.yaml).

    params:
      model: explicit
      inputs: raw | features
      features: null          with inputs: features, the feature names (null: every feature of the domain)
      sets: null              training sets, a mapping name -> ids or a list (set0, ...); one tree each
      max_depth: null
      requirement: null       the requirement whose optimal actions the trees learn (null: the first)
    """

    def __init__(self, domain, params):
        Baseline.__init__(self, domain, params)
        self.kind = params.get("inputs", "raw")
        if self.kind not in ("raw", "features"):
            raise ValueError(f"unknown tree inputs {self.kind!r} (raw | features)")
        self.max_depth = params.get("max_depth")
        self.sets = params.get("sets")
        self.feature_names = params.get("features")
        self.requirement = params.get("requirement")
        self.lists: Dict[str, Rules] = {}
        self.trees: Dict[str, DecisionTreeClassifier] = {}
        self.info: Dict[str, Dict[str, Any]] = {}

    def fit(self, train: Sequence[Instance]) -> None:
        from core.explicit import ExplicitMDP
        by_id = {int(i.id): i for i in train}
        config = self.config or load_config()
        for name, ids in named_sets(self.sets, list(by_id)).items():
            X, y, w, states = [], [], [], 0
            for i in ids:
                mdp = ExplicitMDP(self.domain, by_id[i], config)
                req = next(r for r in mdp.spec.requirements if self.requirement in (None, r.name))
                _, optimal = mdp.optimal_actions(req)
                keep = np.flatnonzero(mdp.decision & ~optimal.all(axis=1))
                x, columns = self.inputs(mdp, keep)
                rows, acts = np.nonzero(optimal[keep])
                X.append(x[rows]), y.append(acts), w.append(1.0 / optimal[keep].sum(axis=1)[rows])
                states += keep.size
                spec = mdp.spec
            tree = DecisionTreeClassifier(criterion="gini", max_depth=self.max_depth, random_state=0)
            tree.fit(np.vstack(X), np.concatenate(y), sample_weight=np.concatenate(w))
            source = f"tree_{self.kind}" + (f"_{name}" if name else "")
            leaves = tree_leaves(tree, columns, list(spec.actions), spec)
            self.trees[source], self.lists[source] = tree, nested(tree, columns, list(spec.actions), spec)
            self.info[source] = {"training": ids, "states": states, "samples": int(sum(len(v) for v in y)),
                                 "nodes": int(tree.tree_.node_count), "depth": int(tree.get_depth()),
                                 "leaves": len(leaves), "rules": leaves if len(leaves) <= MAX_LISTED else None}
            print(f"decision tree {source}: {len(leaves)} leaves, depth {tree.get_depth()}, {states} training states",
                  flush=True)

    def inputs(self, mdp, states: np.ndarray) -> Tuple[np.ndarray, List[Column]]:
        """The tree's inputs on `states` of `mdp` (an core.explicit.ExplicitMDP), and their columns."""
        if self.kind == "raw":
            return mdp.obs[states], [Column(v.name, None, v.type == "bool") for v in mdp.spec.variables]
        table, columns = mdp.features(states, self.feature_names or [f.name for f in mdp.spec.features])
        return table, [Column(c.name, c.action, c.boolean) for c in columns]

    def actions_on(self, mdp, source: str) -> np.ndarray:
        """The tree's action index in every state of `mdp` (-1 outside decision states)."""
        states = np.flatnonzero(mdp.decision)
        out = np.full(mdp.S, -1, dtype=np.int64)
        out[states] = self.trees[source].predict(self.inputs(mdp, states)[0])
        return out

    def rules(self, target: Instance) -> Dict[str, Rules]:
        return dict(self.lists)

    def training(self, source: str) -> Optional[List[int]]:
        return self.info[source]["training"]

    def summary(self) -> Dict[str, Any]:
        return self.info


def tree_leaves(tree: DecisionTreeClassifier, columns: Sequence[Column], actions: Sequence[str], spec) -> Rules:
    """The tree as rules, one per leaf: the bounds its path sets on the inputs, and its majority action."""
    t, out = tree.tree_, []
    stack = [(0, {})]
    while stack:
        node, bounds = stack.pop()
        if t.children_left[node] < 0:
            action = actions[int(tree.classes_[np.argmax(t.value[node][0])])]
            literals = [_literal(columns[j], lo, hi, action, spec) for j, (lo, hi) in sorted(bounds.items())]
            out.append((" & ".join(literals) or "true", action))
            continue
        j, cut = int(t.feature[node]), int(np.floor(t.threshold[node]))
        lo, hi = bounds.get(j, (None, None))
        stack.append((int(t.children_right[node]), {**bounds, j: (cut + 1 if lo is None else max(lo, cut + 1), hi)}))
        stack.append((int(t.children_left[node]), {**bounds, j: (lo, cut if hi is None else min(hi, cut))}))
    return out


def nested(tree: DecisionTreeClassifier, columns: Sequence[Column], actions: Sequence[str], spec) -> Rules:
    """The tree as one rule per action, an if-then-else over its tests (`(t & L) | (!t & R)`, without the branches
    that reach no leaf of the action): the same policy as `tree_leaves`, since the leaves partition the inputs, but
    nested only as deep as the tree."""
    t = tree.tree_
    leaf = {n: actions[int(tree.classes_[np.argmax(t.value[n][0])])] for n in range(t.node_count)
            if t.children_left[n] < 0}
    below: Dict[int, set] = {}
    for n in reversed(range(t.node_count)):           # children have larger ids than their parent
        below[n] = {leaf[n]} if n in leaf else below[t.children_left[n]] | below[t.children_right[n]]

    def cond(n: int, a: str) -> Optional[str]:         # None: false; a disjunction comes in parentheses
        if a not in below[n]:
            return None
        if below[n] == {a}:
            return "true"
        j, cut = int(t.feature[n]), int(np.floor(t.threshold[n]))
        parts = []
        for child, (lo, hi) in ((t.children_left[n], (None, cut)), (t.children_right[n], (cut + 1, None))):
            sub = cond(int(child), a)
            if sub is not None:
                test = _literal(columns[j], lo, hi, a, spec)
                parts.append(test if sub == "true" else f"{test} & {sub}")
        return parts[0] if len(parts) == 1 else f"(({parts[0]}) | ({parts[1]}))"

    return [(c, a) for a in actions for c in [cond(0, a)] if c is not None]


def _literal(col: Column, lo: Optional[int], hi: Optional[int], action: str, spec) -> str:
    """Rule text for `lo <= col <= hi` in a rule for `action`."""
    if col.action is None or col.action == action:
        term = col.name
    else:                       # another action's feature: its expression for that action
        feature = next(f for f in spec.features if f.name == col.name)
        term = f"({feature.per_action[col.action]})"
    if col.boolean:
        return term if lo is not None and lo >= 1 else f"!{term}"
    if lo is not None and lo == hi:
        return f"{term} = {lo}"
    return " & ".join(([f"{term} >= {lo}"] if lo is not None else []) + ([f"{term} <= {hi}"] if hi is not None else []))
