"""Locate where a policy loses probability, as feedback for the refinement loop.

The *mass* of a state s is  d_pi(s) * gap(s):  the expected number of visits to s within a horizon
under strategy pi, times the probability still at stake in s. Mass is high where trajectories
spend time while the outcome is still open, which also catches stalls (the agent circling
through states) that purely local, one-step measures miss.

* extend (best case passes, worst case fails): pi resolves uncovered states adversarially
  (greedy on the worst-case values), and gap = best - worst. Only uncovered states count; these
  are the situations that most need a rule.
* refine (best case fails): pi resolves uncovered states favourably, and gap = optimum - best,
  where the optimum is the best value on the bare MDP. Each state's mass is charged to the
  rule that decides it.

Forced states (every choice has the same successor distribution) are never ranked: no rule can
change what happens there. This is a ranking heuristic, and masses are reported as shares of the total.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.domain import Requirement
from core.prism import PrismResult
from core.verifier import PolicyVerifier, Verification


@dataclass
class Hotspot:
    valuation: Dict                 # policy-variable valuation
    mass: float                     # share of the total mass (0..1)
    requirements: Dict[str, float] = field(default_factory=dict)


@dataclass
class RuleBlame:
    rule: Optional[int]             # rule index (0-based) or None for uncovered states
    mass: float                     # share of the total mass (0..1)
    requirements: Dict[str, float] = field(default_factory=dict)
    hotspots: List[Hotspot] = field(default_factory=list)


def _q_values(result: PrismResult, values: np.ndarray) -> List[np.ndarray]:
    """Per state, the expected next-state value of each available choice."""
    q = []
    for choices in result.choices:
        q.append(np.array([sum(p * values[t] for t, p in c.successors) for c in choices]))
    return q


def _greedy(q: List[np.ndarray], maximize: bool) -> List[int]:
    return [int(np.argmax(v) if maximize else np.argmin(v)) if len(v) else -1 for v in q]


def _occupancy(result: PrismResult, strategy: List[int], horizon: int) -> np.ndarray:
    """Expected number of visits to each state within `horizon` steps under `strategy`."""
    n = len(result.states)
    src, dst, prob = [], [], []
    for s, c in enumerate(strategy):
        if c < 0:
            continue
        for t, p in result.choices[s][c].successors:
            src.append(s)
            dst.append(t)
            prob.append(p)
    src, dst, prob = np.array(src, dtype=int), np.array(dst, dtype=int), np.array(prob)
    mu = np.zeros(n)
    mu[result.initial_state] = 1.0
    visits = np.zeros(n)
    for _ in range(horizon):
        visits += mu
        mu = np.bincount(dst, weights=mu[src] * prob, minlength=n)
        if mu.sum() < 1e-12:
            break
    return visits


def _signed(req: Requirement, vector: List[float]) -> np.ndarray:
    """Values oriented so that larger is always better."""
    v = np.asarray(vector, dtype=float)
    return v if req.maximize else -v


def _normalize(items, total: float) -> None:
    for item in items:
        item.mass = float(item.mass / total)
        item.requirements = {k: float(m / total) for k, m in item.requirements.items()}


class MassAnalyzer:
    def __init__(self, verifier: PolicyVerifier, horizon: int = 100, top_k: int = 10):
        self.verifier = verifier
        self.horizon = horizon
        self.top_k = top_k

    def _state_masses(self, v: Verification, strategy_values: np.ndarray, gap: np.ndarray,
                      maximize_strategy: bool) -> np.ndarray:
        strategy = _greedy(_q_values(v.result, strategy_values), maximize_strategy)
        return _occupancy(v.result, strategy, self.horizon) * np.clip(gap, 0.0, None)

    def uncovered_hotspots(self, v: Verification, failing: List[Requirement]) -> List[Hotspot]:
        """Extend mode: uncovered situations ranked by the worst-case probability lost there."""
        by_valuation: Dict[Tuple, Hotspot] = {}
        for req in failing:
            worst, best = _signed(req, v.worst_vectors[req.name]), _signed(req, v.best_vectors[req.name])
            masses = self._state_masses(v, worst, best - worst, maximize_strategy=False)
            for s in np.nonzero(masses > 0)[0]:
                if v.state_rules[s] is not None or not v.decisions[s]:
                    continue
                valuation = self.verifier.policy_valuation(v.result, s)
                key = tuple(valuation.values())
                spot = by_valuation.setdefault(key, Hotspot(valuation, 0.0))
                spot.mass += masses[s]
                spot.requirements[req.name] = spot.requirements.get(req.name, 0.0) + masses[s]
        total = sum(h.mass for h in by_valuation.values())
        if total > 0:
            _normalize(by_valuation.values(), total)
        return sorted(by_valuation.values(), key=lambda h: -h.mass)[:self.top_k]

    def rule_blame(self, v: Verification, failing: List[Requirement]) -> List[RuleBlame]:
        """Refine mode: probability lost versus the unconstrained optimum, charged to rules."""
        _, opt_vectors, opt_result = self.verifier.optimum()
        if opt_result.variables != v.result.variables:
            raise ValueError("policy module must not add state variables")
        opt_index = opt_result.index_of()
        blames: Dict[Optional[int], RuleBlame] = {}
        spots: Dict[Optional[int], Dict[Tuple, Hotspot]] = defaultdict(dict)
        for req in failing:
            opt = _signed(req, opt_vectors[req.name])
            optimum = np.array([opt[opt_index[state]] for state in v.result.states])
            best = _signed(req, v.best_vectors[req.name])
            masses = self._state_masses(v, best, optimum - best, maximize_strategy=True)
            for s in np.nonzero(masses > 0)[0]:
                if not v.decisions[s]:
                    continue
                rule = v.state_rules[s]
                blame = blames.setdefault(rule, RuleBlame(rule, 0.0))
                blame.mass += masses[s]
                blame.requirements[req.name] = blame.requirements.get(req.name, 0.0) + masses[s]
                valuation = self.verifier.policy_valuation(v.result, s)
                spot = spots[rule].setdefault(tuple(valuation.values()), Hotspot(valuation, 0.0))
                spot.mass += masses[s]
        total = sum(b.mass for b in blames.values())
        for rule, blame in blames.items():
            if total > 0:
                _normalize(spots[rule].values(), total)
            blame.hotspots = sorted(spots[rule].values(), key=lambda h: -h.mass)[:3]
        if total > 0:
            _normalize(blames.values(), total)
        return sorted(blames.values(), key=lambda b: -b.mass)[:self.top_k]
