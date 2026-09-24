"""Verify a (partial) symbolic policy on a domain instance: best and worst case per requirement."""
from dataclasses import dataclass, field
from time import time
from typing import Dict, List, Optional, Tuple

from core.domain import Domain, Instance, Spec
from core.prism import PrismResult, PrismRunner
from core.rules import SymbolicPolicy, Value


@dataclass
class Verification:
    best: Dict[str, float]                  # requirement -> best-case probability (uncovered states resolved favourably)
    worst: Dict[str, float]                 # requirement -> worst-case probability (resolved adversarially)
    result: PrismResult                     # reachable states, transitions and per-state value vectors
    best_vectors: Dict[str, List[float]] = field(default_factory=dict)
    worst_vectors: Dict[str, List[float]] = field(default_factory=dict)
    state_rules: List[Optional[int]] = field(default_factory=list)   # per reachable state: deciding rule or None
    reachable_situations: int = 0          # distinct policy-variable valuations among reachable states
    uncovered_situations: int = 0
    seconds: float = 0.0


class PolicyVerifier:
    """Composes the domain's MDP with a policy module and model checks every requirement twice.

    For a requirement with bound >=, the best case is Pmax and the worst case Pmin (swapped for <=),
    taken over every way of choosing actions in states no rule covers. A fully covering policy
    gives best == worst.
    """

    def __init__(self, domain: Domain, instance: Instance, runner: Optional[PrismRunner] = None):
        self.domain = domain
        self.instance = instance
        self.spec: Spec = domain.spec(instance)
        self.model = domain.model(instance)
        self.runner = runner or PrismRunner()
        self._optimum: Optional[Tuple[Dict[str, float], Dict[str, List[float]], PrismResult]] = None

    def empty_policy(self) -> SymbolicPolicy:
        return SymbolicPolicy(self.spec.variables, list(self.spec.actions))

    def _properties(self, ops: List[str], vectors: bool) -> List[str]:
        props = []
        for req in self.spec.requirements:
            for op in ops:
                prop = f"{op(req) if callable(op) else op}=? [ {req.formula} ]"
                props.append(f"filter(printall, {prop})" if vectors else prop)
        return props

    def verify(self, policy: SymbolicPolicy, analysis: bool = True) -> Verification:
        """Model check `policy`. With `analysis`, also export per-state values and transitions."""
        start = time()
        model = self.model + "\n" + policy.to_prism_module() + "\n"
        props = self._properties([lambda r: r.best_op(), lambda r: r.worst_op()], vectors=analysis)
        result = self.runner.run(model, props, export_transitions=analysis)

        v = Verification(best={}, worst={}, result=result)
        for i, req in enumerate(self.spec.requirements):
            v.best[req.name] = result.initial_values[2 * i]
            v.worst[req.name] = result.initial_values[2 * i + 1]
            if analysis:
                v.best_vectors[req.name] = result.state_values[2 * i]
                v.worst_vectors[req.name] = result.state_values[2 * i + 1]
        self._assign_rules(v, policy)
        v.seconds = time() - start
        return v

    def optimum(self) -> Tuple[Dict[str, float], Dict[str, List[float]], PrismResult]:
        """Best achievable value of each requirement on the bare MDP (no policy), cached.

        Each requirement is optimized separately, so these are upper bounds, not one joint policy.
        """
        if self._optimum is None:
            props = self._properties([lambda r: r.best_op()], vectors=True)
            result = self.runner.run(self.model, props, export_transitions=True)
            values = {r.name: result.initial_values[i] for i, r in enumerate(self.spec.requirements)}
            vectors = {r.name: result.state_values[i] for i, r in enumerate(self.spec.requirements)}
            self._optimum = (values, vectors, result)
        return self._optimum

    def policy_valuation(self, result: PrismResult, state_index: int) -> Dict[str, Value]:
        state = result.states[state_index]
        pos = {name: i for i, name in enumerate(result.variables)}
        return {var.name: state[pos[var.name]] for var in self.spec.variables}

    def _assign_rules(self, v: Verification, policy: SymbolicPolicy) -> None:
        cache: Dict[Tuple, Optional[int]] = {}
        for s in range(len(v.result.states)):
            valuation = self.policy_valuation(v.result, s)
            key = tuple(valuation.values())
            if key not in cache:
                cache[key] = policy.first_match(valuation)
            v.state_rules.append(cache[key])
        v.reachable_situations = len(cache)
        v.uncovered_situations = sum(1 for r in cache.values() if r is None)
