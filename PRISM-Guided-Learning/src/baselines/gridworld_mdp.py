"""A numpy model of one gridworld instance, for searching over policies: the rule-language features of every state
and action, and the value of every requirement under a deterministic policy, in milliseconds instead of a PRISM
run. It mirrors domains/gridworld/model.prism.j2, the domain's requirements and its feature definitions
(`rule_vocabulary`); tests/test_gridworld_mdp.py checks it against PRISM and the core rule evaluator. Search code
uses it to rank candidates; PRISM certifies what is kept.

States are (x, y, stage, phase): stage = number of goals reached in order (the goal flags are always a prefix),
phase = the moving obstacle's position in its cycle (obs_idx; 0 without one). Requirements are evaluated on the
product with a small automaton per requirement (`_Monitor`), as PRISM does for these path formulas.
"""
import importlib.util
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from settings import DOMAINS_PATH


def _domain_module():
    spec = importlib.util.spec_from_file_location("gridworld_domain_for_mdp", DOMAINS_PATH / "gridworld" / "domain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G = _domain_module()
ACTIONS = list(G.MOVES)                       # up, right, down, left
FEATURES = ["wall", "hazard", "risky", "toward", "closer", "obstacle"]
OUTCOMES = {a: [(a, G.PROB_FORWARD), (G.SLIP_LEFT[a], G.PROB_SLIP_LEFT), (G.SLIP_RIGHT[a], G.PROB_SLIP_RIGHT)]
            for a in ACTIONS}


@dataclass
class _Monitor:
    """A deterministic automaton over state labels: `step[q, s]` is the automaton state after entering grid state
    s from automaton state q; `accept` / `reject` are absorbing sets."""
    step: np.ndarray        # (Q, S) int
    accept: np.ndarray      # (Q,) bool
    reject: np.ndarray      # (Q,) bool


class GridMDP:
    """One gridworld instance (a dataset row's data) as arrays."""

    def __init__(self, data: dict):
        self.n, self.goals = data["n"], [tuple(data["goals"][k]) for k in sorted(data["goals"])]
        self.static = {tuple(c) for c in data["static"]}
        self.cycle = [tuple(c) for c in G.expand_cycle(data["moving"])]
        self.K, self.L = len(self.goals), max(len(self.cycle), 1)
        cells = [(r, c) for r in range(self.n) for c in range(self.n) if (r, c) not in self.static]
        self.states = [(r, c, s, o) for (r, c) in cells for s in range(self.K + 1) for o in range(self.L)]
        self.index = {st: i for i, st in enumerate(self.states)}
        self.S = len(self.states)
        self.initial = self.index[(0, 0, 0, 0)]
        self._transitions()
        self.features = self._features()

    # ------------------------------------------------------------ dynamics

    def _move(self, r: int, c: int, d: str) -> Tuple[int, int, bool]:
        """Where moving `d` from (r, c) ends, and whether it entered (or stayed on, at the border) its clamped
        target cell (PRISM's `!blocked`)."""
        dr, dc = G.MOVES[d]
        tr, tc = min(max(r + dr, 0), self.n - 1), min(max(c + dc, 0), self.n - 1)
        blocked = (tr, tc) in self.static
        return (r, c, False) if blocked else (tr, tc, True)

    def _transitions(self):
        """succ[i, a, m]: successor of state i under action a, outcome m; prob[a, m]."""
        succ = np.zeros((self.S, 4, 3), dtype=np.int64)
        for i, (r, c, s, o) in enumerate(self.states):
            for ai, a in enumerate(ACTIONS):
                for m, (d, _) in enumerate(OUTCOMES[a]):
                    nr, nc, entered = self._move(r, c, d)
                    ns = s + 1 if s < self.K and entered and (nr, nc) == self.goals[s] else s
                    succ[i, ai, m] = self.index[(nr, nc, ns, (o + 1) % self.L)]
        self.succ = succ
        self.prob = np.array([[p for _, p in OUTCOMES[a]] for a in ACTIONS])

    # ------------------------------------------------------------ features (domain.rule_vocabulary)

    def _features(self) -> Dict[str, np.ndarray]:
        """Feature name -> bool array (S, 4): its value for each state and action."""
        patrol = set(self.cycle)
        dists = []
        for k in range(self.K):
            later = set(self.goals[k + 1:])
            dists.append(G._distances(self.n, self.goals[k], self.static | later, costly=patrol))
        f = {name: np.zeros((self.S, 4), dtype=bool) for name in FEATURES}
        for i, (r, c, s, o) in enumerate(self.states):
            later = set(self.goals[s + 1:]) if s < self.K else set()
            hazards = patrol | later
            for ai, a in enumerate(ACTIONS):
                dr, dc = G.MOVES[a]
                t = (r + dr, c + dc)
                f["wall"][i, ai] = not (0 <= t[0] < self.n and 0 <= t[1] < self.n) or t in self.static
                f["hazard"][i, ai] = t in hazards
                f["risky"][i, ai] = any((r + G.MOVES[side][0], c + G.MOVES[side][1]) in hazards
                                        for side in (G.SLIP_LEFT[a], G.SLIP_RIGHT[a]))
                f["obstacle"][i, ai] = bool(self.cycle) and t == self.cycle[(o + 1) % self.L]
                if s < self.K:
                    gr, gc = self.goals[s]
                    f["toward"][i, ai] = {"up": gr < r, "down": gr > r, "left": gc < c, "right": gc > c}[a]
                    d = dists[s]
                    f["closer"][i, ai] = (r, c) in d and t in d and d[t] < d[(r, c)]
        return f

    # ------------------------------------------------------------ requirements

    def _labels(self):
        at_goal = np.array([[(r, c) == g for g in self.goals] for (r, c, s, o) in self.states])        # (S, K)
        collide = np.array([bool(self.cycle) and (r, c) == self.cycle[o] for (r, c, s, o) in self.states])
        stage = np.array([s for (_, _, s, _) in self.states])
        return at_goal, collide, stage

    def monitors(self) -> Dict[str, Tuple[_Monitor, bool]]:
        """Requirement name -> (its automaton, whether the requirement is `G !reject` rather than `F accept`)."""
        at_goal, collide, stage = self._labels()
        out = {}
        for k in range(self.K):
            step = np.stack([np.where(at_goal[:, k], 1, 0), np.ones(self.S, dtype=np.int64)])
            out[f"goal{k + 1}"] = (_Monitor(step, np.array([False, True]), np.array([False, False])), False)
        seqs = []
        for i in range(1, self.K):                       # goals 1..i in order before goal i+1
            q = i + 2                                    # progress 0..i-1, accept = i, reject = i+1
            step = np.zeros((q, self.S), dtype=np.int64)
            for p in range(i):
                step[p] = np.where(at_goal[:, p], p + 1, np.where(at_goal[:, i], i + 1, p))
            step[i], step[i + 1] = i, i + 1
            mon = _Monitor(step, np.arange(q) == i, np.arange(q) == i + 1)
            seqs.append(mon)
            out[f"seq_{i}_before_{i + 1}"] = (mon, False)
        if self.K > 1:
            out["complete_sequence"] = (_product(seqs), False)
        if self.cycle:
            for k in range(self.K):
                bad = collide & (stage == k)
                step = np.stack([np.where(bad, 1, 0), np.ones(self.S, dtype=np.int64)])
                out[f"avoid_moving_seg{k + 1}"] = (_Monitor(step, np.array([False, False]), np.array([False, True])),
                                                   True)
        return out

    def evaluate(self, actions: np.ndarray, monitors=None) -> Dict[str, float]:
        """Value of every requirement in the initial state under the deterministic policy `actions` (S,) of action
        indices: the probability of `F accept` (or of `G !reject`)."""
        monitors = monitors or self.monitors()
        nxt = self.succ[np.arange(self.S), actions]                         # (S, 3)
        p = self.prob[actions]                                              # (S, 3)
        return {name: _reach(mon, nxt, p, self.initial, safety) for name, (mon, safety) in monitors.items()}


def _product(monitors: List[_Monitor]) -> _Monitor:
    """Automaton accepting when all of `monitors` accept, rejecting as soon as one rejects."""
    sizes = [m.step.shape[0] for m in monitors]
    combos = np.array(np.meshgrid(*[np.arange(q) for q in sizes], indexing="ij")).reshape(len(sizes), -1).T
    code = {tuple(c): i for i, c in enumerate(combos)}
    S = monitors[0].step.shape[1]
    step = np.zeros((len(combos), S), dtype=np.int64)
    for i, combo in enumerate(combos):
        parts = np.stack([m.step[q] for m, q in zip(monitors, combo)])     # (M, S)
        step[i] = [code[tuple(col)] for col in parts.T]
    accept = np.array([all(m.accept[q] for m, q in zip(monitors, c)) for c in combos])
    reject = np.array([any(m.reject[q] for m, q in zip(monitors, c)) for c in combos])
    return _Monitor(step, accept, reject)


def _reach(mon: _Monitor, nxt: np.ndarray, p: np.ndarray, initial: int, safety: bool) -> float:
    """P(F accept), or P(G !reject) for `safety`, from the initial state of the chain (nxt, p) times `mon`."""
    Q, S = mon.step.shape
    target = mon.reject if safety else mon.accept
    # product state (q, i) -> index q * S + i; successor of (q, i) via outcome m: (step[q, j], j), j = nxt[i, m]
    src = np.repeat(np.arange(Q * S), 3)
    j = np.tile(nxt.reshape(-1), Q)
    q = np.repeat(np.arange(Q), S * 3)
    dst = mon.step[q, j] * S + j
    w = np.tile(p.reshape(-1), Q)
    absorbing = np.repeat(mon.accept | mon.reject, S)
    keep = ~absorbing[src]
    P = sp.csr_matrix((w[keep], (src[keep], dst[keep])), shape=(Q * S, Q * S))
    goal = np.repeat(target, S)
    # states that can reach the target (backward search), the rest have probability 0
    can = goal.copy()
    Pt = P.T.tocsr()
    frontier = np.flatnonzero(goal)
    while frontier.size:
        pred = Pt[frontier].indices
        new = pred[~can[pred]]
        can[new] = True
        frontier = np.unique(new)
    solve = can & ~goal
    x = goal.astype(float)
    if solve.any():
        idx = np.flatnonzero(solve)
        A = sp.identity(idx.size, format="csr") - P[idx][:, idx]
        b = np.asarray(P[idx][:, np.flatnonzero(goal)].sum(axis=1)).ravel()
        x[idx] = spsolve(A.tocsc(), b)
    start = mon.step[0, initial] * S + initial
    value = float(x[start])
    return 1.0 - value if safety else value


def first_allowed(mdp: GridMDP, rules: List[Tuple[Dict[str, bool], Optional[List[int]]]],
                  fallback: int = 0) -> np.ndarray:
    """A deterministic policy from a decision list over feature literals: each rule is ({feature: wanted value},
    an action order or None for the default order); the first rule that allows some action picks the first allowed
    action in its order. States no rule covers take `fallback`."""
    chosen = np.full(mdp.S, -1, dtype=np.int64)
    for literals, order in rules:
        ok = np.ones((mdp.S, 4), dtype=bool)
        for name, want in literals.items():
            ok &= mdp.features[name] == want
        for a in (order or range(4)):
            pick = (chosen < 0) & ok[:, a]
            chosen[pick] = a
    chosen[chosen < 0] = fallback
    return chosen


def optimal_actions(mdp: GridMDP, tol: float = 1e-9, max_iters: int = 100000, discount: float = 0.99) -> Tuple[np.ndarray, np.ndarray]:
    """Optimal actions for one reachability objective, 1-2-3-Go!'s kind: reach every goal in order without entering
    a later goal early and without meeting the moving obstacle. Returns (value per state, bool (S, 4) of optimal
    actions) by value iteration; ties between probability-optimal actions go to those that reach the target soonest."""
    cell_goal = {g: k for k, g in enumerate(mdp.goals)}
    success = np.array([s == mdp.K for (_, _, s, _) in mdp.states])
    violation = np.array([(cell_goal.get((r, c), -1) > s) or (bool(mdp.cycle) and s < mdp.K and (r, c) == mdp.cycle[o])
                          for (r, c, s, o) in mdp.states])
    fixed = success | violation
    V = success.astype(float)
    for _ in range(max_iters):
        Q = (mdp.prob[None, :, :] * V[mdp.succ]).sum(axis=2)            # (S, 4)
        new = np.where(fixed, V, Q.max(axis=1))
        if np.abs(new - V).max() < tol:
            V = new
            break
        V = new
    Q = (mdp.prob[None, :, :] * V[mdp.succ]).sum(axis=2)
    optimal = Q >= Q.max(axis=1, keepdims=True) - 1e-9
    # Among the probability-optimal actions, prefer those that reach the target soonest (discounted value over
    # optimal actions only): an arbitrary choice among them can cycle forever where the probability is 1 anyway.
    D = success.astype(float)
    for _ in range(max_iters):
        Qd = np.where(optimal, discount * (mdp.prob[None, :, :] * D[mdp.succ]).sum(axis=2), -1.0)
        new = np.where(fixed, D, Qd.max(axis=1))
        if np.abs(new - D).max() < tol:
            D = new
            break
        D = new
    Qd = np.where(optimal, (mdp.prob[None, :, :] * D[mdp.succ]).sum(axis=2), -1.0)
    return V, Qd >= Qd.max(axis=1, keepdims=True) - 1e-12
