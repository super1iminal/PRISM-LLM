"""Exact worst-case values of rule lists on one gridworld instance in numpy: the fast checker the loop uses where it
scores many candidate lists without PRISM (`planner.repair`, `planner.pick: candidates`). Ported from the
fresh-transfer session's checker (src/fresh/gw.py on branch fresh-transfer), on gridworld_mdp.GridMDP's states and for any number of goals.

The rules give each state the actions their deciding rule allows (core.vector, the PRISM encoding's semantics), and
each requirement's worst case over those actions is computed exactly on its reach / avoid form (goals k_1 < k_2 < ...,
stage = goals reached in order):
    goal{k}                F at_goal_k          reach the goal's cell
    seq_{k_i}_before_...   the domain's U form   reach stage > i before entering goal k_{i+1}'s cell at stage <= i
    complete_sequence      all of them           reach stage >= K - 1 avoiding every such entry
    avoid_moving_seg{k}    G (seg_k => !obs)     1 - the largest chance of meeting the obstacle in segment k
by policy iteration after the graph precomputations (Prob0A for max, Prob0E for min), so each linear system is regular.
tests/test_gridworld_check.py compares it with PRISM.
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve

from baselines.gridworld_mdp import GridMDP
from core.vector import allowed_actions


class GridChecker:
    """Worst-case values of rule lists (parsed with `vocabulary`) on one gridworld instance."""

    def __init__(self, data: dict, spec, vocabulary):
        self.spec, self.vocabulary = spec, vocabulary
        self.mdp = m = GridMDP(data)
        stage = np.array([s for (_, _, s, _) in m.states])
        names = {v.name for v in spec.variables}
        values = {"x": np.array([r for (r, _, _, _) in m.states]), "y": np.array([c for (_, c, _, _) in m.states])}
        values.update({f"g{k + 1}": stage > k for k in range(m.K)})
        if "obs_idx" in names:
            values["obs_idx"] = np.array([o for (_, _, _, o) in m.states])
        self.values = {k: v for k, v in values.items() if k in names}
        self.cache: Dict = {}
        self.succ, self.p = m.succ, m.prob[0]                       # S x 4 x 3, the outcome probabilities
        cells = [(r, c) for (r, c, _, _) in m.states]
        at = [np.array([cell == g for cell in cells]) for g in m.goals]
        on_obs = np.array([bool(m.cycle) and (r, c) == m.cycle[o] for (r, c, s, o) in m.states])
        ks = sorted(data["goals"])
        self.forms: Dict[str, Tuple[np.ndarray, Optional[np.ndarray], bool]] = {}
        for i, k in enumerate(ks):
            self.forms[f"goal{k}"] = (at[i], None, False)
        early = []
        for i in range(m.K - 1):
            avoid = at[i + 1] & (stage <= i)
            early.append(avoid)
            self.forms[f"seq_{ks[i]}_before_{ks[i + 1]}"] = (stage > i, avoid, False)
        if m.K > 1:
            self.forms["complete_sequence"] = (stage >= m.K - 1, np.logical_or.reduce(early), False)
        if m.cycle:
            for i, k in enumerate(ks):
                self.forms[f"avoid_moving_seg{k}"] = (on_obs & (stage == i), None, True)
        missing = [r.name for r in spec.requirements if r.name not in self.forms]
        if missing:
            raise ValueError(f"no reach/avoid form for requirements {missing}")

    def allowed(self, rules, order: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        policy = self.spec.policy(rules, self.vocabulary)
        return allowed_actions(policy, self.values, self.mdp.S, self.cache, order)

    def worst(self, rules, order: Optional[Sequence[str]] = None) -> Dict[str, float]:
        """Each requirement's worst-case value from the initial state (`core.rules.RuleError` if the rules do not
        parse)."""
        allowed, _ = self.allowed(rules, order)
        out = {}
        for r in self.spec.requirements:
            target, avoid, bad = self.forms[r.name]
            if bad:                                                 # G !bad: the adversary maximizes reaching bad
                out[r.name] = 1.0 - float(self._reach(allowed, target, None, maximize=True)[self.mdp.initial])
            else:
                out[r.name] = float(self._reach(allowed, target, avoid, maximize=False)[self.mdp.initial])
        return out

    def _reach(self, allowed: np.ndarray, target: np.ndarray, avoid: Optional[np.ndarray], maximize: bool) -> np.ndarray:
        """P{min,max} of reaching `target` before `avoid` (both absorbing) over the allowed actions: min removes every
        end component first (Prob0E), max starts from a proper attractor strategy and switches actions only on strict
        improvement, which keeps it proper."""
        succ, p, S = self.succ, self.p, self.mdp.S
        stop = target | (avoid if avoid is not None else np.zeros(S, dtype=bool))
        live = ~stop
        if maximize:
            dist, d = np.where(target, 0, -1), 0
            while True:
                hit = (dist[succ] == d).any(axis=2) & allowed
                new = live & (dist < 0) & hit.any(axis=1)
                if not new.any():
                    break
                d += 1
                dist[new] = d
            zero = live & (dist < 0)
            hit_prev = (dist[succ] == (dist - 1)[:, None, None]).any(axis=2) & allowed
            act = np.where(hit_prev.any(axis=1), np.argmax(hit_prev, axis=1), np.argmax(allowed, axis=1))
        else:
            Z = ~target
            while True:
                ok = (Z[succ].all(axis=2) & allowed).any(axis=1)
                nz = Z & (~live | ok)
                if (nz == Z).all():
                    break
                Z = nz
            zero = live & Z
            act = np.argmax(allowed, axis=1)
        solve = live & ~zero
        idx = np.flatnonzero(solve)
        v = target.astype(float)
        if not len(idx):
            return v
        pos = np.full(S, -1)
        pos[idx] = np.arange(len(idx))
        for _ in range(1000):
            nxt = succ[idx, act[idx]]
            rows, cols, pr = np.repeat(np.arange(len(idx)), 3), pos[nxt.ravel()], np.tile(p, len(idx))
            inside = cols >= 0
            A = identity(len(idx), format="csr") - csr_matrix((pr[inside], (rows[inside], cols[inside])),
                                                              shape=(len(idx), len(idx)))
            b = np.zeros(len(idx))
            np.add.at(b, rows[~inside], pr[~inside] * target[nxt.ravel()[~inside]])
            v = target.astype(float)
            v[idx] = spsolve(A.tocsc(), b)
            q = np.where(allowed, (v[succ] * p).sum(axis=2), -np.inf if maximize else np.inf)
            cur = q[np.arange(S), act]
            better = ((q.max(axis=1) > cur + 1e-12) if maximize else (q.min(axis=1) < cur - 1e-12)) & solve
            if not better.any():
                break
            act = np.where(better, q.argmax(axis=1) if maximize else q.argmin(axis=1), act)
        return v
