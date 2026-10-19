"""Verify a (partial) symbolic policy on a domain instance: best and worst case per requirement."""
import subprocess
from dataclasses import dataclass, field, replace
from time import time
from typing import Dict, List, Optional, Set, Tuple

from config import Config
from core.domain import Domain, Instance, Spec
from core.prism import Choice, PrismError, PrismResult, PrismRunner, PrismSteps, StateKey, run_now
from core.rules import SymbolicPolicy, Value


SOUND = {"lp": "linear programming", "intervaliter": "interval iteration"}   # prism.exact_method -> solver name


def exact_solvers(cfg) -> List[Tuple[str, List[str]]]:
    """The exact check's solvers in order, as (name, PRISM arguments), for PRISM config `cfg`: `exact_method`
    (linear programming, or interval iteration at `exact_epsilon`), then Gauss-Seidel at `exact_fallback_epsilon`
    (no error bound) where it fails."""
    first = ["-lp"] if cfg.exact_method == "lp" else ["-intervaliter", "-topological", "-epsilon", cfg.exact_epsilon]
    return [(SOUND[cfg.exact_method], first),
            (f"Gauss-Seidel {cfg.exact_fallback_epsilon}", ["-gaussseidel", "-epsilon", cfg.exact_fallback_epsilon])]


def exact_band(cfg, solver: Optional[str] = None) -> float:
    """How close to a threshold a value from `solver` (default: `exact_method`'s) decides nothing: `lp_tolerance`
    for linear programming, which has no error bound, and interval iteration's precision for it."""
    method = {name: m for m, name in SOUND.items()}.get(solver, cfg.exact_method)
    return float(cfg.lp_tolerance) if method == "lp" else float(cfg.exact_epsilon)


def exact_verdict(margins: Dict[str, float], solver: Optional[str], cfg) -> str:
    """The certification rule on one check's requirement margins: certified iff a sound solver (`SOUND`) gave
    every margin beyond its band (`exact_band`); inconclusive within the band, or from the fallback with no margin
    below minus the band; failed otherwise."""
    band, smallest = exact_band(cfg, solver), min(margins.values())
    if solver not in SOUND.values() or abs(smallest) <= band:
        return "inconclusive" if smallest >= -band else "failed"
    return "certified" if smallest > band else "failed"


def forced(choices: List[Choice], actions: Set[str]) -> bool:
    """Whether the policy cannot change anything in a state with these choices: among the choices labelled with a
    policy action (in `actions`), fewer than two distinct successor distributions."""
    controlled = {tuple(sorted((t, round(p, 12)) for t, p in c.successors)) for c in choices if c.action in actions}
    return len(controlled) <= 1


@dataclass
class Verification:
    best: Dict[str, float]                  # requirement -> best-case probability (uncovered states resolved favourably)
    worst: Dict[str, float]                 # requirement -> worst-case probability (resolved adversarially)
    result: PrismResult                     # reachable states, transitions and per-state value vectors
    best_vectors: Dict[str, List[float]] = field(default_factory=dict)
    worst_vectors: Dict[str, List[float]] = field(default_factory=dict)
    state_rules: List[Optional[int]] = field(default_factory=list)   # per reachable state: deciding rule or None
    decisions: List[bool] = field(default_factory=list)   # per reachable state: can the action change anything?
    reachable_situations: int = 0          # distinct policy-variable valuations among reachable decision states
    uncovered_situations: int = 0
    seconds: float = 0.0


class PolicyVerifier:
    """Composes the domain's MDP with a policy module and model checks every requirement twice.

    For a requirement with bound >=, the best case is Pmax and the worst case Pmin (swapped for <=),
    taken over every way of choosing actions in states no rule covers. A fully covering policy
    gives best == worst.

    Each method that runs PRISM comes in two forms: `x_steps(...)`, a generator that yields every PRISM
    process as a `PrismCall` (the planner's loop uses these, so schedulers can run PRISM for many
    instances at once), and `x(...)`, which runs those processes here and returns the answer.
    """

    def __init__(self, domain: Domain, instance: Instance, config: Config, runner: Optional[PrismRunner] = None):
        """PRISM and rule settings come from `config`; `runner` replaces the one it would build
        (e.g. to share it, or to use other solver arguments)."""
        self.domain = domain
        self.instance = instance
        self.spec: Spec = domain.spec(instance)
        self.model = domain.model(instance)
        self.runner = runner or PrismRunner(config.prism)
        self.prism_config = config.prism
        self.max_enumeration = config.rules.max_enumeration
        self._optimum: Optional[Tuple[Dict[str, float], Dict[str, List[float]], PrismResult]] = None
        self._forced: Optional[Set[StateKey]] = None

    def empty_policy(self) -> SymbolicPolicy:
        return SymbolicPolicy(self.spec.variables, list(self.spec.actions))

    def _properties(self, ops: List[str], vectors: bool) -> List[str]:
        props = []
        for req in self.spec.requirements:
            for op in ops:
                prop = f"{op(req) if callable(op) else op}=? [ {req.formula} ]"
                props.append(f"filter(printall, {prop})" if vectors else prop)
        return props

    def verify(self, policy: SymbolicPolicy, analysis: bool = True,
               runner: Optional[PrismRunner] = None) -> Verification:
        return run_now(self.verify_steps(policy, analysis, runner))

    def verify_steps(self, policy: SymbolicPolicy, analysis: bool = True,
                     runner: Optional[PrismRunner] = None) -> PrismSteps[Verification]:
        """Model check `policy` (with `runner`, default the verifier's own). With `analysis`, also export
        per-state values and transitions."""
        start = time()
        model = self.compose(policy)
        props = self._properties([lambda r: r.best_op(), lambda r: r.worst_op()], vectors=analysis)
        result = yield from (runner or self.runner).run_steps(model, props, export_transitions=analysis)

        v = Verification(best={}, worst={}, result=result)
        for i, req in enumerate(self.spec.requirements):
            v.best[req.name] = result.initial_values[2 * i]
            v.worst[req.name] = result.initial_values[2 * i + 1]
            if analysis:
                v.best_vectors[req.name] = result.state_values[2 * i]
                v.worst_vectors[req.name] = result.state_values[2 * i + 1]
        yield from self.forced_states_steps()
        self._assign_rules(v, policy)
        v.seconds = time() - start
        return v

    def verify_exact(self, policy: SymbolicPolicy) -> Tuple[Optional[Verification], Optional[str]]:
        return run_now(self.verify_exact_steps(policy))

    def verify_exact_steps(self, policy: SymbolicPolicy) -> PrismSteps[Tuple[Optional[Verification], Optional[str]]]:
        """Model check `policy` with the exact solver (`prism.exact_method`: linear programming, or interval
        iteration), or Gauss-Seidel at `prism.exact_fallback_epsilon` where it fails (`exact_solvers`). Returns
        the verification and the solver used, or (None, None) if neither finishes.

        The loop's solver stops when an iteration changes little, which bounds nothing: on chains that
        leak probability slowly it stops early, and values PRISM derives as 1 - another probability
        (`G` properties, the worst case of LTL properties) then come out too high. Interval iteration bounds the
        error but can stall on such chains (the crossing family's moving obstacle: no convergence in 10^8
        iterations); linear programming solves the model directly.
        """
        cfg = self.prism_config
        exact = replace(cfg, method="", max_iters=cfg.exact_max_iters)
        for name, args in exact_solvers(cfg):
            try:
                runner = PrismRunner(exact, extra_args=args, prism_path=self.runner.prism_path)
                return (yield from self.verify_steps(policy, analysis=False, runner=runner)), name
            except (PrismError, subprocess.TimeoutExpired):
                continue
        return None, None

    def compose(self, policy: Optional[SymbolicPolicy]) -> str:
        """The domain's MDP restricted by `policy` (the bare MDP when `policy` is None)."""
        if policy is None:
            return self.model
        return self.model + "\n" + policy.to_prism_module(self.max_enumeration) + "\n"

    def joint_query(self) -> str:
        """PRISM multi-objective query: can one scheduler meet every threshold at once?"""
        objectives = ", ".join(r.bounded() for r in self.spec.requirements)
        return f"multi({objectives})"

    def jointly_feasible(self, policy: Optional[SymbolicPolicy] = None) -> Optional[bool]:
        return run_now(self.jointly_feasible_steps(policy))

    def jointly_feasible_steps(self, policy: Optional[SymbolicPolicy] = None) -> PrismSteps[Optional[bool]]:
        """Whether a single completion of `policy` (any scheduler, possibly randomized and with memory)
        meets all thresholds simultaneously. `None` asks the question of the bare MDP (the ceiling).

        "No" is exact; "yes" is optimistic for memoryless, observation-based completions. None means
        undecided (PRISM's exact LP method does not support e.g. step-bounded requirements, or the LP did not
        finish within `prism.timeout_s`).
        """
        try:
            return (yield from self.runner.check_steps(self.compose(policy), self.joint_query()))
        except (PrismError, subprocess.TimeoutExpired):
            return None   # objective kinds PRISM's multi-objective engine rejects, or an LP that runs too long

    def optimum(self) -> Tuple[Dict[str, float], Dict[str, List[float]], PrismResult]:
        return run_now(self.optimum_steps())

    def optimum_steps(self) -> PrismSteps[Tuple[Dict[str, float], Dict[str, List[float]], PrismResult]]:
        """Best achievable value of each requirement on the bare MDP (no policy), cached.

        Each requirement is optimized separately, so these are upper bounds, not one joint policy.
        """
        if self._optimum is None:
            props = self._properties([lambda r: r.best_op()], vectors=True)
            result = yield from self.runner.run_steps(self.model, props, export_transitions=True)
            values = {r.name: result.initial_values[i] for i, r in enumerate(self.spec.requirements)}
            vectors = {r.name: result.state_values[i] for i, r in enumerate(self.spec.requirements)}
            self._optimum = (values, vectors, result)
        return self._optimum

    def forced_states(self) -> Set[StateKey]:
        return run_now(self.forced_states_steps())

    def forced_states_steps(self) -> PrismSteps[Set[StateKey]]:
        """States of the bare MDP where the policy cannot change anything, cached: among the choices
        labelled with a policy action, fewer than two distinct successor distributions.

        Other choices are not the policy's to make: forced moves under other labels, or unlabelled
        nondeterminism such as Pac-Man's idle step once both ghosts are gone. These states are not
        decision points: they do not count as situations and are left out of the feedback.
        """
        if self._forced is None:
            if self._optimum:
                result = self._optimum[2]
            else:
                result = yield from self.runner.run_steps(self.model, [], export_transitions=True)
            actions = set(self.spec.actions)
            self._forced = {state for state, choices in zip(result.states, result.choices) if forced(choices, actions)}
        return self._forced

    def policy_valuation(self, result: PrismResult, state_index: int) -> Dict[str, Value]:
        """The policy-visible variables of one reachable state, in spec order."""
        state, pos = result.states[state_index], result.positions
        return {var.name: state[pos[var.name]] for var in self.spec.variables}

    def _assign_rules(self, v: Verification, policy: SymbolicPolicy) -> None:
        forced = self._forced   # computed by verify_steps
        cache: Dict[Tuple, Optional[int]] = {}
        situations = set()
        for s, state in enumerate(v.result.states):
            valuation = self.policy_valuation(v.result, s)
            key = tuple(valuation.values())
            if key not in cache:
                cache[key] = policy.first_match(valuation)
            v.state_rules.append(cache[key])
            v.decisions.append(state not in forced)
            if v.decisions[-1]:
                situations.add(key)
        v.reachable_situations = len(situations)
        v.uncovered_situations = sum(1 for key in situations if cache[key] is None)
