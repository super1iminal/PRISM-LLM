"""The cheap baselines of 1-2-3-Go!'s evaluation and the strategy of one instance applied to the others, on any domain's explicit model (explicit_mdp.py). Their policies are per-state, certified through each
target's explicit model.

- `Uniform`: every policy action with equal probability in every decision state (a randomized policy; other
  nondeterminism, such as Pac-Man's idle step, stays open, so the certificate is the worst case over it).
- `RandomPolicies`: the best of `count` random deterministic memoryless policies. Policy k is a seeded hash of the
  policy-visible valuation, so it is defined on every instance; ranked by the sum of its worst-case values on the
  training instances (numpy), the best one is certified on the targets.
- `OneInstance`: the optimal strategy of one training instance (`instance`; the first optimal action in the
  declared order) as a table from a key to an action, looked up on the targets. Each source names its key
  (`keys`): policy variables and rule-language state features (core evaluator), e.g. Pac-Man's variables, or the
  same with `steps` replaced by `moves_left` to align horizons by their end. Keys never seen in training are left
  undecided (the worst case covers them).

params:
  RandomPolicies: count (1000), seed (0), requirement (null: the first)
  OneInstance: instance (an id among the training ids), keys {source: [names]}, requirement (null: the first)
"""
from typing import Dict, List, Optional, Sequence

import numpy as np

from baselines.base import Baseline, Rules
from config import load_config
from core.domain import Instance


class _Explicit(Baseline):
    """Per-state policies on the explicit model; `rules` writes them one rule per state (small targets only)."""
    explicit = True

    def _mdp(self, instance: Instance):
        from baselines.explicit_mdp import ExplicitMDP
        return ExplicitMDP(self.domain, instance, self.config or load_config())

    def rules(self, target: Instance) -> Dict[str, Rules]:
        from baselines.explicit_mdp import per_state_rules
        mdp = self._mdp(target)
        return {s: per_state_rules(mdp, p) for s, p in self.policies(mdp).items()}

    def _requirement(self, mdp):
        name = self.params.get("requirement")
        return next(r for r in mdp.spec.requirements if name in (None, r.name))


class Uniform(_Explicit):

    def policies(self, mdp) -> Dict[str, np.ndarray]:
        return {"uniform": np.full((mdp.S, len(mdp.actions)), 1.0 / len(mdp.actions))}

    def rules(self, target: Instance) -> Dict[str, Rules]:
        raise NotImplementedError("a randomized policy has no rules")


def hashed_actions(obs: np.ndarray, seed: int, actions: int) -> np.ndarray:
    """A random function of each row of `obs` (integers) into range(actions), fixed by `seed` (splitmix64 over the
    row)."""
    def mix(x):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))
    with np.errstate(over="ignore"):
        h = mix(np.full(len(obs), seed, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15))
        for col in obs.astype(np.int64).T.view(np.uint64):
            h = mix(h ^ (col + np.uint64(0x9E3779B97F4A7C15)))
    return (h % np.uint64(actions)).astype(np.int64)


class RandomPolicies(_Explicit):

    def __init__(self, domain, params):
        super().__init__(domain, params)
        self.count, self.seed = params.get("count", 1000), params.get("seed", 0)
        self.best: Optional[int] = None
        self.scores: List[float] = []

    def _policy(self, mdp, k: int) -> np.ndarray:
        policy = np.full(mdp.S, -1, dtype=np.int64)
        states = np.flatnonzero(mdp.decision)
        policy[states] = hashed_actions(mdp.obs[states], self.seed * 1_000_003 + k, len(mdp.actions))
        return policy

    def fit(self, train: Sequence[Instance]) -> None:
        total = np.zeros(self.count)
        for inst in train:
            mdp = self._mdp(inst)
            req = self._requirement(mdp)
            for lo in range(0, self.count, 50):
                ks = range(lo, min(lo + 50, self.count))
                total[lo:lo + len(ks)] += mdp.evaluate_many(np.stack([self._policy(mdp, k) for k in ks]), req)
            maximize = req.bound == ">="
        self.best = int(np.argmax(total) if maximize else np.argmin(total))
        self.scores = total.tolist()
        print(f"random policies: best {self.best} of {self.count}, training sum {total[self.best]:.4f} "
              f"(median {np.median(total):.4f})", flush=True)

    def policies(self, mdp) -> Dict[str, np.ndarray]:
        return {f"random_best_of_{self.count}": self._policy(mdp, self.best)}

    def summary(self):
        return {f"random_best_of_{self.count}": {"best": self.best, "training_sum": self.scores[self.best],
                                                 "median_training_sum": float(np.median(self.scores))}}


class OneInstance(_Explicit):

    def __init__(self, domain, params):
        super().__init__(domain, params)
        self.instance = int(params["instance"])
        self.keys: Dict[str, List[str]] = params["keys"]
        self.tables: Dict[str, Dict[tuple, int]] = {}

    @staticmethod
    def key_rows(mdp, states: np.ndarray, names: Sequence[str]) -> np.ndarray:
        """Per state in `states`: the values of `names` (policy variables, or state features of the rule language)."""
        variables = [v.name for v in mdp.spec.variables]
        features = [n for n in names if n not in variables]
        table = mdp.features(states, features)[0] if features else np.zeros((len(states), 0), dtype=np.int64)
        cols = [mdp.obs[states, variables.index(n)] if n in variables else table[:, features.index(n)] for n in names]
        return np.stack(cols, axis=1)

    def fit(self, train: Sequence[Instance]) -> None:
        inst = next(i for i in train if int(i.id) == self.instance)
        mdp = self._mdp(inst)
        _, optimal = mdp.optimal_actions(self._requirement(mdp))
        states = np.flatnonzero(mdp.decision)
        actions = optimal[states].argmax(axis=1)
        for source, names in self.keys.items():
            rows = self.key_rows(mdp, states, names)
            self.tables[source] = {tuple(r): int(a) for r, a in zip(rows.tolist(), actions)}
            if len(self.tables[source]) != len(states):
                raise ValueError(f"key {names} does not tell the decision states of instance {self.instance} apart")

    def policies(self, mdp) -> Dict[str, np.ndarray]:
        states = np.flatnonzero(mdp.decision)
        out = {}
        for source, names in self.keys.items():
            table = self.tables[source]
            policy = np.full(mdp.S, -1, dtype=np.int64)
            policy[states] = [table.get(tuple(r), -1) for r in self.key_rows(mdp, states, names).tolist()]
            out[source] = policy
        return out

    def training(self, source: str) -> Optional[List[int]]:
        return [self.instance]
