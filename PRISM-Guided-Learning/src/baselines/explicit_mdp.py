"""Any domain's instance as an explicit MDP in numpy arrays, for the baselines that need the whole model: optimal
actions (1-2-3-Go!'s training data), simulation (PPO), the value of a per-state policy, and the PRISM certificate of a
policy that has no compact rules.

The model is the domain's PRISM model of the instance as PRISM builds and exports it (explicit engine): every
reachable state's valuation, its choices with their action labels and successors, and the state labels. Nothing here
is domain-specific: decision states are the verifier's (`core.verifier.forced`), the policy-visible variables, actions
and requirements come from the domain's spec, and features are computed by the core rule evaluator.

Objectives: the probability of `F ψ` or `φ U ψ`, where φ and ψ are boolean combinations of the model's labels,
minimized or maximized over every scheduler (all choices, including those under labels the policy does not control,
such as Pac-Man's idle step), or over the schedulers a policy leaves. Values come from value iteration from below,
stopped when no state changes by more than `TOL` (exact on models whose only cycles are self-loops, such as Pac-Man).
Anything else (rewards, step bounds, `G`, conditions on variables) raises NotImplementedError; a domain whose
requirements are of that kind can give one objective of the supported form on a model of its own (`Domain.objective`,
`ExplicitMDP(..., objective=True)`).

A per-state policy is an array over the states: an action index per state (-1: undecided), or a probability per state
and action (a row of zeros: undecided). It applies at decision states only. Undecided decision states keep every
choice, so the worst case covers them, as for states no rule covers. `certify` writes the model restricted to the
policy (its reachable part) as PRISM explicit files and checks them with `-importmodel`, so PRISM certifies.
"""
import re
import subprocess
import tempfile
from functools import cached_property
from io import StringIO
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order

from config import Config
from core.domain import Domain, Instance, Requirement
from core.prism import Choice, PrismCall, PrismError, PrismRunner, run_call
from core.rules import Var, Vocabulary, _Scope, atomic_rules, evaluate
from core.verifier import PolicyVerifier, exact_solvers, forced

TOL = 1e-12
MAX_ITERS = 1_000_000


class FeatureColumn(NamedTuple):
    """One input column from a rule-language feature: its name, the action it is for (None: a state feature), its
    compiled expression and whether it is boolean."""
    name: str
    action: Optional[str]
    expr: object
    boolean: bool

    @property
    def label(self) -> str:
        return self.name if self.action is None else f"{self.name}[{self.action}]"


class Model:
    """An MDP as arrays: choices grouped by state (`choice_start`, CSR), each with its policy action (`choice_action`:
    an index into the actions, -1 for any other label or none) and its transitions (`trans_start`, `target`, `prob`);
    `labels` (name -> bool per state) and the initial state."""

    def __init__(self, choice_start, choice_action, trans_start, target, prob, labels: Dict[str, np.ndarray],
                 initial: int):
        self.choice_start, self.choice_action = choice_start, choice_action
        self.trans_start, self.target, self.prob = trans_start, target, prob
        self.labels, self.initial = labels, int(initial)

    @property
    def S(self) -> int:
        return len(self.choice_start) - 1

    @property
    def C(self) -> int:
        return len(self.choice_action)

    @cached_property
    def choice_state(self) -> np.ndarray:
        return np.repeat(np.arange(self.S, dtype=np.int64), np.diff(self.choice_start))

    @cached_property
    def matrix(self) -> sp.csr_matrix:
        """Choices x states: each choice's successor distribution."""
        return sp.csr_matrix((self.prob, self.target, self.trans_start), shape=(self.C, self.S))

    @cached_property
    def absorbing(self) -> np.ndarray:
        """Per state: every transition of every choice is a self-loop."""
        loops = self.target == np.repeat(self.choice_state, np.diff(self.trans_start))
        per_choice = np.logical_and.reduceat(loops, self.trans_start[:-1])
        return np.logical_and.reduceat(per_choice, self.choice_start[:-1])

    # ------------------------------------------------------------ objectives

    def path_sets(self, formula: str) -> Tuple[np.ndarray, np.ndarray]:
        """(φ, ψ) of a path formula `F ψ` or `φ U ψ` over labels, as bool arrays over the states."""
        return _path_sets(formula, self.labels, self.S)

    def reach(self, formula: str, maximize: bool, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Per state: the maximum (or minimum) over schedulers of the probability of `formula`; with `mask` (per
        choice), over the schedulers that only take the choices it keeps (a state left without one keeps its value,
        as a self-loop would)."""
        stay, goal = self.path_sets(formula)
        solve = stay & ~goal
        V = goal.astype(float)
        reduce = np.maximum.reduceat if maximize else np.minimum.reduceat
        blocked = -np.inf if maximize else np.inf
        for _ in range(MAX_ITERS):
            q = self.matrix @ V
            if mask is not None:
                q = np.where(mask, q, blocked)
            best = reduce(q, self.choice_start[:-1])
            new = np.where(solve & np.isfinite(best), best, V)
            if np.abs(new - V).max(initial=0.0) <= TOL:
                return new
            V = new
        raise RuntimeError(f"value iteration did not converge in {MAX_ITERS} iterations")

    state_rewards: Dict[str, np.ndarray] = {}     # reward structure -> state reward per state (ExplicitMDP, if needed)

    def bounded_reach(self, goal: np.ndarray, steps: int, maximize: bool,
                      mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Per state: the maximum (or minimum) over schedulers of reaching `goal` within `steps` transitions, by
        backward induction over exactly `steps` steps (PRISM's `F<=k`). A state left without a choice keeps its value,
        as a self-loop would."""
        V = goal.astype(float)
        reduce = np.maximum.reduceat if maximize else np.minimum.reduceat
        blocked = -np.inf if maximize else np.inf
        for _ in range(steps):
            q = self.matrix @ V
            if mask is not None:
                q = np.where(mask, q, blocked)
            best = reduce(q, self.choice_start[:-1])
            V = np.where(goal, 1.0, np.where(np.isfinite(best), best, V))
        return V

    def reward_until(self, reward: str, formula: str, maximize: bool, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Per state: the maximum (or minimum) over schedulers of the expected state reward `reward` collected until
        `formula` (`F ψ`) holds, each state's reward counted when the state is left (PRISM's `R{r}[F ψ]`). Infinite
        where the optimizing scheduler can miss ψ with positive probability, as PRISM reports it. Value iteration from
        zero, stopped when no state changes by more than `TOL`."""
        stay, goal = self.path_sets(formula)
        if not stay.all():
            raise NotImplementedError(f"reward until {formula!r}: only F")
        r = self.state_rewards[reward]
        sure = self.almost_surely(goal, every=maximize, mask=mask)     # ψ with probability 1 (graph algorithm)
        V = np.where(sure | goal, 0.0, np.inf)
        reduce = np.maximum.reduceat if maximize else np.minimum.reduceat
        blocked = -np.inf if maximize else np.inf
        solve = sure & ~goal
        to_infinite = (self.matrix @ np.isinf(V).astype(float)) > 0      # choices that may reach an infinite state
        for _ in range(MAX_ITERS):
            q = np.where(to_infinite, np.inf, self.matrix @ np.where(np.isinf(V), 0.0, V))
            if mask is not None:
                q = np.where(mask, q, blocked)
            best = reduce(q, self.choice_start[:-1])
            new = np.where(solve, r + np.where(np.isfinite(best), best, 0.0), V)
            if np.abs(new[solve] - V[solve]).max(initial=0.0) <= TOL:
                return new
            V = new
        raise RuntimeError(f"value iteration did not converge in {MAX_ITERS} iterations")

    def almost_surely(self, goal: np.ndarray, every: bool, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Per state: is `goal` reached with probability 1 under every scheduler (`every`, PRISM's Prob1A) or under
        some scheduler (Prob1E)? Graph algorithms on the choices `mask` keeps (a state left without one keeps a
        self-loop), exact where value iteration only approaches 1."""
        keep = np.ones(self.C, dtype=bool) if mask is None else mask
        state = self.choice_state
        succ = sp.csr_matrix((np.ones(self.target.size), (np.repeat(np.arange(self.C), np.diff(self.trans_start)),
                                                         self.target)), shape=(self.C, self.S))
        has_choice = np.zeros(self.S, dtype=bool)
        has_choice[state[keep]] = True

        def some_choice(ok_choice: np.ndarray) -> np.ndarray:
            out = np.zeros(self.S, dtype=bool)
            out[state[ok_choice & keep]] = True
            return out

        if every:
            # avoid: states where some scheduler avoids `goal` forever (greatest fixpoint: some choice stays inside)
            avoid = ~goal
            while True:
                inside = (succ @ (~avoid).astype(float)) == 0                  # every successor in avoid
                nxt = avoid & (some_choice(inside) | ~has_choice)              # no choice: a self-loop stays
                if (nxt == avoid).all():
                    break
                avoid = nxt
            # can reach `avoid` with positive probability while not in `goal`: then not almost sure
            bad = avoid.copy()
            while True:
                hits = (succ @ bad.astype(float)) > 0
                nxt = bad | (~goal & some_choice(hits))
                if (nxt == bad).all():
                    break
                bad = nxt
            return ~bad
        U = np.ones(self.S, dtype=bool)
        while True:                                                            # Prob1E: nested fixpoint
            stays = (succ @ (~U).astype(float)) == 0
            R = goal.copy()
            while True:
                hits = (succ @ R.astype(float)) > 0
                nxt = R | (U & some_choice(stays & hits))
                if (nxt == R).all():
                    break
                R = nxt
            if (R == U).all():
                return U
            U = R

    def requirement_value(self, requirement: Requirement, worst: bool, mask: Optional[np.ndarray] = None) -> float:
        """The requirement's worst (or best) value in the initial state, for the kinds `value` leaves out: `G φ`
        (1 minus the opposite optimum of `F !φ`), `F<=k ψ` (`bounded_reach`) and expected rewards until `F ψ`
        (`reward_until`); `F` / `U` as `reach`. Labels only, as everywhere here."""
        op = requirement.worst_op() if worst else requirement.best_op()
        maximize = op.endswith("max")
        f = requirement.formula.strip()
        if requirement.reward is not None:
            if requirement.reward not in self.state_rewards:
                raise NotImplementedError(f"reward structure {requirement.reward!r}: state rewards only")
            return float(self.reward_until(requirement.reward, f, maximize, mask)[self.initial])
        bounded = _BOUNDED.match(f)
        if bounded:
            _, goal = self.path_sets("F " + bounded.group(2))
            return float(self.bounded_reach(goal, int(bounded.group(1)), maximize, mask)[self.initial])
        globally = _GLOBALLY.match(f)
        if globally:
            return 1.0 - float(self.reach(f"F !({globally.group(1)})", maximize=not maximize, mask=mask)[self.initial])
        return float(self.reach(f, maximize=maximize, mask=mask)[self.initial])

    def value(self, requirement: Requirement, worst: bool) -> float:
        """The requirement's worst (or best) value in the initial state."""
        if requirement.reward is not None:
            raise NotImplementedError("reward requirements")
        op = requirement.worst_op() if worst else requirement.best_op()
        return float(self.reach(requirement.formula, maximize=op == "Pmax")[self.initial])

    # ------------------------------------------------------------ restriction and certification

    def restrict(self, policy: np.ndarray, decide: np.ndarray) -> "Model":
        """The model under `policy` at the states where `decide` holds and the policy decides: other choices under a
        policy action are removed (deterministic policy) or replaced by their mixture (randomized). Choices under
        other labels stay. A state left without a choice gets a self-loop, as PRISM fixes deadlocks."""
        soc, act = self.choice_state, self.choice_action
        if policy.ndim == 1:
            decided = decide & (policy >= 0)
            keep = (act < 0) | ~decided[soc] | (act == policy[soc])
            mixed_state = np.zeros(0, dtype=np.int64)
            mixed_t = (np.zeros(0, dtype=np.int64),) * 2 + (np.zeros(0),)
        else:
            decided = decide & (policy.sum(axis=1) > 0)
            policy_choice = (act >= 0) & decided[soc]
            pairs = soc[policy_choice] * policy.shape[1] + act[policy_choice]
            if np.unique(pairs).size != pairs.size:
                raise NotImplementedError("a randomized policy where one action labels several choices of a state")
            keep = ~policy_choice
            mixed_state = np.flatnonzero(decided)
            owner = np.searchsorted(mixed_state, soc[policy_choice])          # which mixed choice each one joins
            weight = policy[soc[policy_choice], act[policy_choice]]
            tr = _ranges(self.trans_start[:-1][policy_choice], self.trans_start[1:][policy_choice])
            lengths = np.diff(self.trans_start)[policy_choice]
            mixed_t = (np.repeat(owner, lengths), self.target[tr], self.prob[tr] * np.repeat(weight, lengths))
        kept = np.flatnonzero(keep)
        lengths = np.diff(self.trans_start)[kept]
        tr = _ranges(self.trans_start[:-1][kept], self.trans_start[1:][kept])
        states = np.r_[soc[kept], mixed_state]
        t_choice = np.r_[np.repeat(np.arange(kept.size), lengths), kept.size + mixed_t[0]]
        actions = np.r_[act[kept], np.full(mixed_state.size, -1)]
        empty = np.setdiff1d(np.arange(self.S), states)                       # deadlocks: one self-loop each
        t_choice = np.r_[t_choice, states.size + np.arange(empty.size)]
        states, actions = np.r_[states, empty], np.r_[actions, np.full(empty.size, -1)]
        return _build(self.S, states, actions, t_choice, np.r_[self.target[tr], mixed_t[1], empty],
                      np.r_[self.prob[tr], mixed_t[2], np.ones(empty.size)], self.labels, self.initial)

    def reachable(self) -> np.ndarray:
        """Per state: reachable from the initial state."""
        graph = sp.csr_matrix((np.ones(self.target.size), (np.repeat(self.choice_state, np.diff(self.trans_start)),
                                                           self.target)), shape=(self.S, self.S))
        mask = np.zeros(self.S, dtype=bool)
        mask[breadth_first_order(graph, self.initial, directed=True, return_predecessors=False)] = True
        return mask

    def part(self, keep: np.ndarray) -> "Model":
        """The model on the states `keep` (closed under successors), renumbered in their order."""
        new = np.cumsum(keep) - 1
        choices = np.flatnonzero(keep[self.choice_state])
        tr = _ranges(self.trans_start[:-1][choices], self.trans_start[1:][choices])
        lengths = np.diff(self.trans_start)[choices]
        return _build(int(keep.sum()), new[self.choice_state[choices]], self.choice_action[choices],
                      np.repeat(np.arange(choices.size), lengths), new[self.target[tr]], self.prob[tr],
                      {k: v[keep] for k, v in self.labels.items()}, new[self.initial])

    def write(self, base: Path, labels: Sequence[str]) -> str:
        """Write `<base>.tra` and `<base>.lab` (PRISM's explicit format, with `labels`); returns the -importmodel
        argument."""
        local = np.arange(self.C) - self.choice_start[self.choice_state]
        lengths = np.diff(self.trans_start)
        rows = pd.DataFrame({"s": np.repeat(self.choice_state, lengths), "c": np.repeat(local, lengths),
                             "t": self.target, "p": self.prob})
        with open(f"{base}.tra", "w", encoding="utf-8", newline="\n") as f:
            f.write(f"{self.S} {self.C} {self.target.size}\n")
            rows.to_csv(f, sep=" ", header=False, index=False, lineterminator="\n")
        names = ["init"] + list(labels)
        marks = np.stack([np.arange(self.S) == self.initial] + [self.labels[n] for n in labels], axis=1)
        code = marks.astype(np.int64) @ (1 << np.arange(len(names), dtype=np.int64))    # which labels, as bits
        has = np.flatnonzero(code)
        ids = {c: " " + " ".join(str(k) for k in range(len(names)) if c >> k & 1) for c in np.unique(code[has])}
        with open(f"{base}.lab", "w", encoding="utf-8", newline="\n") as f:
            f.write(" ".join(f'{k}="{n}"' for k, n in enumerate(names)) + "\n")
            pd.DataFrame({"s": has, "ids": pd.Series(code[has]).map(ids)}).to_csv(
                f, sep=":", header=False, index=False, lineterminator="\n")
        return f"{base}.tra,lab"


class ExplicitMDP(Model):
    """One instance of a domain, as PRISM exports its model: `values` holds every state's valuation (columns
    `variables`, PRISM's order; booleans as 0 and 1), `actions` the spec's policy actions."""

    def __init__(self, domain: Domain, instance: Instance, config: Config, runner: Optional[PrismRunner] = None,
                 objective: bool = False):
        """`objective`: the domain's objective model (`Domain.objective`) instead of its model, for learning that
        objective's optimal actions; its requirement is `objective_requirement`."""
        verifier = PolicyVerifier(domain, instance, config, runner)
        self.domain, self.instance, self.config = domain, instance, config
        self.spec, self.runner = verifier.spec, verifier.runner
        self.actions = list(self.spec.actions)
        model_text, self.objective_requirement = verifier.model, None
        if objective:
            found = domain.objective(instance)
            if found is None:
                raise ValueError(f"{type(domain).__name__} has no objective model (Domain.objective)")
            model_text, self.objective_requirement = found
        with tempfile.TemporaryDirectory(prefix="explicit_") as tmp:
            base = Path(tmp) / "model"
            base.with_suffix(".prism").write_text(model_text, encoding="utf-8")
            cmd = [self.runner.prism_path, str(base.with_suffix(".prism")), "-explicit",
                   "-exportstates", f"{base}.sta", "-exporttrans", f"{base}.tra", "-exportlabels", f"{base}.lab",
                   "-javamaxmem", self.runner.java_max_mem]
            rewards = not objective and any(r.reward is not None for r in self.spec.requirements)
            if rewards:     # state rewards per structure (the fast checker's expected-reward requirements)
                cmd += ["-exportstaterewards", f"{base}.srew"]
            out = run_call(PrismCall(tuple(cmd), self.runner.timeout, self.runner.max_concurrent))
            if "Error" in out or not Path(f"{base}.tra").exists():
                raise PrismError("\n".join(l for l in out.splitlines() if "rror" in l) or out[-2000:])
            self.variables, self.values = _read_states(f"{base}.sta")
            labels, initial = _read_labels(f"{base}.lab", len(self.values))
            parts = _read_transitions(f"{base}.tra", self.actions)
            state_rewards = _read_state_rewards(base, len(self.values)) if rewards else {}
        super().__init__(*parts, labels=labels, initial=initial)
        self.state_rewards = state_rewards

    # ------------------------------------------------------------ what the policy sees

    @cached_property
    def decision(self) -> np.ndarray:
        """Per state: is it a decision state? Exactly the verifier's definition (`core.verifier.forced`); states with
        fewer than two choices under policy actions are forced without asking."""
        controlled = np.add.reduceat((self.choice_action >= 0).astype(np.int64), self.choice_start[:-1])
        out = np.zeros(self.S, dtype=bool)
        actions = set(self.actions)
        for s in np.flatnonzero(controlled >= 2):
            choices = []
            for c in range(self.choice_start[s], self.choice_start[s + 1]):
                a, lo, hi = self.choice_action[c], self.trans_start[c], self.trans_start[c + 1]
                choices.append(Choice(self.actions[a] if a >= 0 else None,
                                      list(zip(self.target[lo:hi].tolist(), self.prob[lo:hi].tolist()))))
            out[s] = not forced(choices, actions)
        return out

    @cached_property
    def obs(self) -> np.ndarray:
        """Per state: the policy-visible variables (spec order)."""
        return self.values[:, [self.variables.index(v.name) for v in self.spec.variables]]

    def valuation(self, row: np.ndarray) -> Dict[str, object]:
        """One `obs` row as the rule evaluator takes it."""
        return {v.name: (bool(x) if v.type == "bool" else int(x)) for v, x in zip(self.spec.variables, row)}

    def feature_columns(self, names: Sequence[str]) -> List[FeatureColumn]:
        """The columns of the named rule-language features: one per state feature, one per action per action
        feature, each with its compiled expression and type (from the core rule language)."""
        scope = _Scope.build(self.spec.variables, self.actions,
                             Vocabulary(list(self.spec.constants), list(self.spec.features)))
        sample = self.valuation(self.obs[self.initial])
        columns = []
        for name in names:
            if name in scope.state_features:
                parts = [(None, scope.state_features[name])]
            elif name in scope.action_features:
                parts = [(a, scope.action_features[name][a]) for a in self.actions]
            elif name in scope.variables:       # a policy variable itself (what rules.shared_literals lets rules name)
                parts = [(None, Var(name))]
            else:
                raise ValueError(f"unknown feature {name!r}; known: {', '.join(scope.state_features)}, "
                                 f"{', '.join(scope.action_features)}")
            columns += [FeatureColumn(name, a, e, isinstance(evaluate(e, sample), bool)) for a, e in parts]
        return columns

    def features(self, states: np.ndarray, names: Sequence[str]) -> Tuple[np.ndarray, List[FeatureColumn]]:
        """The named rule-language features on `states` (rows; booleans as 0 and 1) and their columns
        (`feature_columns`). Computed by the core rule evaluator, once per distinct observation."""
        columns = self.feature_columns(names)
        rows, inverse = np.unique(self.obs[states], axis=0, return_inverse=True)
        table = np.array([[int(evaluate(c.expr, self.valuation(r))) for c in columns] for r in rows], dtype=np.int64)
        return table.reshape(len(rows), len(columns))[inverse.ravel()], columns

    # ------------------------------------------------------------ optimal actions

    def optimal_actions(self, requirement: Requirement, tol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
        """(per state: the optimum over all schedulers, per state and action: is the action optimal?). An action is
        optimal where its best choice is within `tol` of the best choice under a policy action. When maximizing, it
        must also make progress where the optimum is positive: some optimal choice of it reaches a state closer to ψ
        in the attractor of the optimal choices (as Storm's schedulers do). Otherwise a self-loop, such as an action
        that leaves the state unchanged, would count as optimal, since its value equals the state's."""
        maximize = requirement.best_op() == "Pmax"
        V = self.reach(requirement.formula, maximize)
        Q = self.matrix @ V
        worst = -np.inf if maximize else np.inf
        QA = np.full((self.S, len(self.actions)), worst)
        mine = self.choice_action >= 0
        (np.maximum if maximize else np.minimum).at(QA, (self.choice_state[mine], self.choice_action[mine]), Q[mine])
        best = QA.max(axis=1) if maximize else QA.min(axis=1)
        with np.errstate(invalid="ignore"):            # states without policy actions: inf - inf
            optimal = np.isfinite(QA) & (np.abs(QA - best[:, None]) <= tol)
        if maximize:
            optimal &= self._progress(requirement, V, Q, tol)
        return V, optimal

    def _progress(self, requirement: Requirement, V: np.ndarray, Q: np.ndarray, tol: float) -> np.ndarray:
        """Per state and action: True unless the state's optimum is positive and none of the action's optimal choices
        has a successor closer to ψ (fewer optimal choices away) than the state."""
        _, goal = self.path_sets(requirement.formula)
        state = self.choice_state
        optimal = (Q >= V[state] - tol) & (V[state] > tol)
        rank = np.where(goal, 0, -1)
        moves = self.matrix[optimal]
        for r in range(self.S):
            hits = (moves @ (rank >= 0).astype(float)) > 0
            new = np.zeros(self.S, dtype=bool)
            new[state[optimal][hits]] = True
            new &= rank < 0
            if not new.any():
                break
            rank[new] = r + 1
        succ_rank = np.where(rank[self.target] >= 0, rank[self.target], self.S + 1)
        closest = np.minimum.reduceat(succ_rank, self.trans_start[:-1])
        progress = optimal & (closest < rank[state])
        keep = np.zeros((self.S, len(self.actions)), dtype=bool)
        mine = progress & (self.choice_action >= 0)
        keep[state[mine], self.choice_action[mine]] = True
        return keep | ~(rank > 0)[:, None]

    # ------------------------------------------------------------ policies

    def under(self, policy: np.ndarray) -> Model:
        """The model under a per-state policy (module docstring), applied at decision states."""
        return self.restrict(policy, self.decision)

    def evaluate(self, policy: np.ndarray, worst: bool = True, names: Optional[Sequence[str]] = None) -> Dict[str, float]:
        """Requirement name -> the policy's worst (or best) value over the choices it leaves, in numpy (`names`: only
        these requirements)."""
        model = self.under(policy)
        return {r.name: model.value(r, worst) for r in self.spec.requirements if names is None or r.name in names}

    def evaluate_many(self, policies: np.ndarray, requirement: Requirement, worst: bool = True) -> np.ndarray:
        """The worst (or best) value in the initial state of each deterministic policy in `policies` (k, S), by one
        value iteration over all of them at once."""
        if requirement.reward is not None:
            raise NotImplementedError("reward requirements")
        maximize = (requirement.worst_op() if worst else requirement.best_op()) == "Pmax"
        stay, goal = self.path_sets(requirement.formula)
        soc, act = self.choice_state, self.choice_action
        decided = self.decision[None, :] & (policies >= 0)
        allowed = ((act < 0)[:, None] | ~decided[:, soc].T | (act[:, None] == policies[:, soc].T))     # (C, k)
        solve = (stay & ~goal)[:, None]
        V = np.repeat(goal[:, None].astype(float), len(policies), axis=1)
        blocked = -np.inf if maximize else np.inf
        reduce = np.maximum.reduceat if maximize else np.minimum.reduceat
        for _ in range(MAX_ITERS):
            best = reduce(np.where(allowed, self.matrix @ V, blocked), self.choice_start[:-1], axis=0)
            new = np.where(solve, np.where(np.isinf(best), 0.0, best), V)        # no choice left: a self-loop
            if np.abs(new - V).max(initial=0.0) <= TOL:
                return new[self.initial]
            V = new
        raise RuntimeError(f"value iteration did not converge in {MAX_ITERS} iterations")

    def certify(self, policies: Dict[str, np.ndarray]) -> List[dict]:
        """Each named policy certified by PRISM on this instance: rows in src/transfer.py's format. The model under
        the policy (its reachable part) goes to PRISM as explicit files, checked with the exact solvers
        (`prism.exact_methods`), or Gauss-Seidel at `prism.exact_fallback_epsilon` where they all fail."""
        rows = []
        reqs = self.spec.requirements
        for source, policy in policies.items():
            under = self.under(policy)
            keep = under.reachable()
            model = under.part(keep)
            undecided = keep & self.decision & ~_decides(policy)
            uncovered = len(np.unique(self.obs[undecided], axis=0)) if undecided.any() else 0
            row = {"source": source, "target": int(self.instance.id), "parsed": True}
            try:                # PRISM cannot import a one-state model (its variable's range would be [0..0])
                worst = {r.name: model.value(r, worst=True) for r in reqs} if model.S == 1 else self._prism_worst(model)
            except (PrismError, subprocess.TimeoutExpired) as e:    # undecided: counts as not certified
                rows.append({**row, "certified": False, "requirements": len(reqs),
                             "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"})
                continue
            numpy = {r.name: model.value(r, worst=True) for r in reqs}
            off = [r.name for r in reqs if abs(numpy[r.name] - worst[r.name]) > 1e-6]
            if off:
                print(f"  warning: {source} on {self.instance.id}: numpy and PRISM differ on {off}: "
                      f"{numpy} against {worst}", flush=True)
            met = [r.name for r in reqs if r.satisfied(worst[r.name])]
            rows.append({**row, "met": len(met), "requirements": len(reqs), "certified": len(met) == len(reqs),
                         "uncovered": uncovered, "exact": True, **{f"worst_{r.name}": worst[r.name] for r in reqs}})
        return rows

    def _prism_worst(self, model: Model) -> Dict[str, float]:
        """Requirement name -> its worst case on `model`, by PRISM on the model's explicit files."""
        reqs = self.spec.requirements
        if any(r.reward is not None for r in reqs):
            raise NotImplementedError("reward requirements")
        names = sorted({n for r in reqs for n in re.findall(r'"([^"]+)"', r.formula)})
        cfg = self.config.prism
        solvers = [args for _, args in exact_solvers(cfg)]
        with tempfile.TemporaryDirectory(prefix="certify_") as tmp:
            files = model.write(Path(tmp) / "model", names)
            props = Path(tmp) / "props.props"
            props.write_text("".join(f"{r.worst_op()}=? [ {r.formula} ];\n" for r in reqs), encoding="utf-8")
            for args in solvers:
                cmd = [self.runner.prism_path, "-importmodel", files, str(props), "-explicit", "-javamaxmem",
                       self.runner.java_max_mem, "-maxiters", str(cfg.exact_max_iters), *args]
                out = run_call(PrismCall(tuple(cmd), self.runner.timeout, self.runner.max_concurrent))
                results = re.findall(r"^Result: (\S+)", out, re.M)
                if "Error" not in out and "did not converge" not in out and len(results) == len(reqs):
                    return {r.name: float(v) for r, v in zip(reqs, results)}
        raise PrismError("\n".join(l for l in out.splitlines() if "rror" in l or "converge" in l) or out[-2000:])


class RuleChecker:
    """Worst-case values of rule lists on one instance's explicit model, in numpy: the fast checker the loop uses where
    it scores many candidate lists without PRISM (`planner.repair`, `planner.pick: candidates`), for domains whose
    requirements are `F` / `U` over labels (Pac-Man, MER). The rules' allowed actions per state come from
    core.vector (the PRISM encoding's semantics); the policy's choices they exclude are masked, and each requirement's
    worst case is the opposite optimum over the rest (`Model.reach`)."""

    def __init__(self, domain: Domain, instance: Instance, config: Config, vocabulary):
        from core.vector import allowed_actions
        self._allowed_actions = allowed_actions
        self.mdp = ExplicitMDP(domain, instance, config)
        self.spec, self.vocabulary = self.mdp.spec, vocabulary
        self.values = {v.name: (self.mdp.obs[:, i] != 0) if v.type == "bool" else self.mdp.obs[:, i]
                       for i, v in enumerate(self.spec.variables)}
        self.cache: Dict = {}
        self.mine = self.mdp.choice_action >= 0

    def allowed(self, rules, order: Optional[Sequence[str]] = None):
        policy = self.spec.policy(rules, self.vocabulary)
        return self._allowed_actions(policy, self.values, self.mdp.S, self.cache, order)

    def worst(self, rules, order: Optional[Sequence[str]] = None) -> Dict[str, float]:
        """Each requirement's worst-case value from the initial state (`core.rules.RuleError` if the rules do not
        parse), on the part of the model the rules' choices reach."""
        allowed, _ = self.allowed(rules, order)
        m = self.mdp
        mask = np.ones(m.C, dtype=bool)
        mask[self.mine] = allowed[m.choice_state[self.mine], m.choice_action[self.mine]]
        sub = masked_part(m, mask)
        return {r.name: (float(sub.reach(r.formula, maximize=r.worst_op() == "Pmax")[sub.initial])
                         if r.reward is None and not (_BOUNDED.match(r.formula) or _GLOBALLY.match(r.formula))
                         else sub.requirement_value(r, worst=True))
                for r in self.spec.requirements}


def masked_part(model: Model, mask: np.ndarray) -> Model:
    """The part of `model` its initial state reaches through the choices `mask` keeps, with only those choices (a state
    left without one gets a self-loop, as PRISM resolves deadlocks), renumbered in state order."""
    lengths = np.diff(model.trans_start)
    kept = np.repeat(mask, lengths)
    graph = sp.csr_matrix((np.ones(int(kept.sum())), (np.repeat(model.choice_state, lengths)[kept], model.target[kept])),
                          shape=(model.S, model.S))
    keep = np.zeros(model.S, dtype=bool)
    keep[breadth_first_order(graph, model.initial, directed=True, return_predecessors=False)] = True
    new = np.cumsum(keep) - 1
    choices = np.flatnonzero(keep[model.choice_state] & mask)
    tr = _ranges(model.trans_start[:-1][choices], model.trans_start[1:][choices])
    states = new[model.choice_state[choices]]
    S = int(keep.sum())
    empty = np.setdiff1d(np.arange(S), states)
    part = _build(S, np.r_[states, empty], np.r_[model.choice_action[choices], np.full(empty.size, -1)],
                  np.r_[np.repeat(np.arange(choices.size), lengths[choices]), choices.size + np.arange(empty.size)],
                  np.r_[new[model.target[tr]], empty], np.r_[model.prob[tr], np.ones(empty.size)],
                  {k: v[keep] for k, v in model.labels.items()}, new[model.initial])
    if model.state_rewards:
        part.state_rewards = {k: v[keep] for k, v in model.state_rewards.items()}
    return part


def optimal_strategy(domain: Domain, instance: Instance, config: Config, requirement: Optional[str] = None):
    """`ExplicitMDP.optimal_actions` of one requirement (default: the spec's only one) as a core.optimal strategy: a
    state (every model variable) -> its chance of meeting the requirement's optimum direction (the probability when
    maximizing, one minus it when minimizing) and optimal actions; settled where every scheduler gets the same."""
    from core.optimal import Optimal
    mdp = ExplicitMDP(domain, instance, config)
    reqs = [r for r in mdp.spec.requirements if requirement in (None, r.name)]
    if len(reqs) != 1:
        raise ValueError(f"name one of the requirements {[r.name for r in mdp.spec.requirements]} for its optimum")
    req = reqs[0]
    maximize = req.best_op() == "Pmax"
    best, optimal = mdp.optimal_actions(req)
    worst = mdp.reach(req.formula, maximize=not maximize)
    success = best if maximize else 1.0 - best
    index = {row.tobytes(): i for i, row in enumerate(mdp.values)}
    dtype = mdp.values.dtype

    def strategy(state: dict):
        i = index.get(np.array([int(state[v]) for v in mdp.variables], dtype=dtype).tobytes())
        if i is None:
            return None
        return Optimal(float(success[i]), tuple(a for a, ok in zip(mdp.actions, optimal[i]) if ok),
                       settled=abs(best[i] - worst[i]) <= 1e-9)
    return strategy


def _decides(policy: np.ndarray) -> np.ndarray:
    return policy >= 0 if policy.ndim == 1 else policy.sum(axis=1) > 0


def per_state_rules(mdp: ExplicitMDP, policy: np.ndarray) -> List[Tuple[str, str]]:
    """A deterministic per-state policy as rules, one per decided observation of a decision state (core.rules
    .atomic_rules): the rule path of certification, for small instances only (a list of n rules compiles into
    about n^2 / 2 negated conditions)."""
    if policy.ndim != 1:
        raise NotImplementedError("a randomized policy has no rules")
    states = np.flatnonzero(mdp.decision & (policy >= 0))
    rows, first = np.unique(mdp.obs[states], axis=0, return_index=True)
    return atomic_rules((mdp.valuation(r), mdp.actions[policy[states[i]]]) for r, i in zip(rows, first))


# ---------------------------------------------------------------- arrays

def _ranges(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """The concatenated index ranges [starts[i], ends[i])."""
    lengths = ends - starts
    if not lengths.size:
        return np.zeros(0, dtype=np.int64)
    offsets = np.repeat(starts - np.r_[0, np.cumsum(lengths)[:-1]], lengths)
    return offsets + np.arange(lengths.sum())


def _build(S: int, choice_state, choice_action, t_choice, target, prob, labels, initial) -> Model:
    """A Model from choices (their state and action) and transitions (their choice, target, probability) in any
    order: choices sorted by state (stably), transitions by choice and target, duplicate targets merged."""
    order = np.argsort(choice_state, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    t_choice = rank[t_choice]
    t_order = np.lexsort((target, t_choice))
    t_choice, target, prob = t_choice[t_order], target[t_order], prob[t_order]
    first = np.r_[True, (t_choice[1:] != t_choice[:-1]) | (target[1:] != target[:-1])]
    starts = np.flatnonzero(first)
    prob = np.add.reduceat(prob, starts) if starts.size else prob
    t_choice, target = t_choice[first], target[first]
    choice_state = np.asarray(choice_state)[order]
    return Model(np.searchsorted(choice_state, np.arange(S + 1)).astype(np.int64),
                 np.asarray(choice_action)[order].astype(np.int64),
                 np.searchsorted(t_choice, np.arange(order.size + 1)).astype(np.int64),
                 target.astype(np.int64), prob.astype(np.float64), labels, initial)


# ---------------------------------------------------------------- PRISM's explicit files

def _read_states(path: str) -> Tuple[List[str], np.ndarray]:
    """Variable names (PRISM order) and one row of values per state (booleans as 0 and 1)."""
    text = Path(path).read_text(encoding="utf-8")
    header, _, body = text.partition("\n")
    variables = header.strip().strip("()").split(",")
    body = body.replace("true", "1").replace("false", "0").replace(":(", ",").replace(")", "")
    del text
    table = pd.read_csv(StringIO(body), header=None, dtype=np.int64).to_numpy()
    if not (table[:, 0] == np.arange(len(table))).all():
        raise PrismError(f"unexpected state numbering in {path}")
    return variables, table[:, 1:].astype(np.int32)


def _read_labels(path: str, S: int) -> Tuple[Dict[str, np.ndarray], int]:
    """Label name -> bool per state, and the initial state."""
    with open(path, encoding="utf-8") as f:
        names = {int(k): v for k, v in re.findall(r'(\d+)="([^"]*)"', f.readline())}
        labels = {n: np.zeros(S, dtype=bool) for n in names.values()}
        for line in f:
            s, _, ids = line.partition(":")
            for k in ids.split():
                labels[names[int(k)]][int(s)] = True
    initial = np.flatnonzero(labels.pop("init"))
    if initial.size != 1:
        raise PrismError(f"models must have exactly one initial state (found {initial.size})")
    return labels, int(initial[0])


def _read_transitions(path: str, actions: Sequence[str]):
    """choice_start, choice_action, trans_start, target, prob from PRISM's .tra file (rows by state and choice)."""
    with open(path, encoding="utf-8") as f:
        S, C, T = (int(x) for x in f.readline().split())
    df = pd.read_csv(path, sep=" ", skiprows=1, header=None, names=["s", "c", "t", "p", "a"],
                     dtype={"s": np.int64, "c": np.int64, "t": np.int64, "p": np.float64, "a": "category"})
    s, c = df.s.to_numpy(), df.c.to_numpy()
    first = np.r_[True, (s[1:] != s[:-1]) | (c[1:] != c[:-1])]
    if len(df) != T or first.sum() != C or (np.diff(s) < 0).any():
        raise PrismError(f"unexpected layout of {path}")
    code = {a: i for i, a in enumerate(actions)}
    label_action = np.array([code.get(a, -1) for a in df.a.cat.categories] + [-1], dtype=np.int64)
    choice_state = s[first]
    choice_start = np.searchsorted(choice_state, np.arange(S + 1)).astype(np.int64)
    if (np.diff(choice_start) == 0).any():
        raise PrismError(f"{path} has a state without choices")
    return (choice_start, label_action[df.a.cat.codes.to_numpy()[first]], np.r_[np.flatnonzero(first), T],
            df.t.to_numpy(), df.p.to_numpy())


# ---------------------------------------------------------------- path formulas over labels

_TOKEN = re.compile(r'\s*("[^"]*"|[()!&|]|\w+)')
_BOUNDED = re.compile(r"^\s*F\s*<=\s*(\d+)\s+(.+)$")       # F<=k ψ
_GLOBALLY = re.compile(r"^\s*G\s+(.+)$")                     # G φ


def _read_state_rewards(base: Path, S: int) -> Dict[str, np.ndarray]:
    """Reward structure name -> state reward per state, from PRISM's `-exportstaterewards <base>.srew` (one file, or
    one per structure as <base>1.srew, <base>2.srew, ..., each headed `# Reward structure "<name>"`)."""
    out = {}
    for path in sorted(base.parent.glob(base.name + "*.srew")):
        lines = path.read_text(encoding="utf-8").splitlines()
        name = next((re.search(r'"([^"]+)"', l).group(1) for l in lines if l.startswith("# Reward structure")), None)
        rows = [l for l in lines if l.strip() and not l.startswith("#")]
        r = np.zeros(S)
        for line in rows[1:]:                         # rows[0]: states and non-zero entries
            i, w = line.split()
            r[int(i)] = float(w)
        out[name or path.stem] = r
    return out


def _path_sets(formula: str, labels: Dict[str, np.ndarray], S: int) -> Tuple[np.ndarray, np.ndarray]:
    tokens, pos = [], 0
    while pos < len(formula.rstrip()):
        m = _TOKEN.match(formula, pos)
        if not m:
            raise NotImplementedError(f"path formula {formula!r}: only F and U over labels")
        tokens.append(m.group(1))
        pos = m.end()
    if tokens and tokens[0] == "F":
        return np.ones(S, dtype=bool), _StateFormula(tokens[1:], labels, S, formula).parse()
    depth = [0]
    for t in tokens:
        depth.append(depth[-1] + (t == "(") - (t == ")"))
    split = [i for i, t in enumerate(tokens) if t == "U" and depth[i] == 0]
    if len(split) != 1:
        raise NotImplementedError(f"path formula {formula!r}: only F and U over labels")
    i = split[0]
    return (_StateFormula(tokens[:i], labels, S, formula).parse(),
            _StateFormula(tokens[i + 1:], labels, S, formula).parse())


class _StateFormula:
    """A boolean combination of labels (`"name"`, true, false, !, &, |, parentheses) as a bool array."""

    def __init__(self, tokens: List[str], labels: Dict[str, np.ndarray], S: int, formula: str):
        self.tokens, self.labels, self.S, self.formula, self.i = tokens, labels, S, formula, 0

    def _fail(self):
        raise NotImplementedError(f"path formula {self.formula!r}: only F and U over labels")

    def parse(self) -> np.ndarray:
        out = self.or_()
        if self.i != len(self.tokens):
            self._fail()
        return out

    def _peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def or_(self):
        out = self.and_()
        while self._peek() == "|":
            self.i += 1
            out = out | self.and_()
        return out

    def and_(self):
        out = self.not_()
        while self._peek() == "&":
            self.i += 1
            out = out & self.not_()
        return out

    def not_(self):
        if self._peek() == "!":
            self.i += 1
            return ~self.not_()
        return self.atom()

    def atom(self):
        tok = self._peek()
        self.i += 1
        if tok == "(":
            out = self.or_()
            if self._peek() != ")":
                self._fail()
            self.i += 1
            return out
        if tok in ("true", "false"):
            return np.full(self.S, tok == "true")
        if tok and tok.startswith('"') and tok[1:-1] in self.labels:
            return self.labels[tok[1:-1]].copy()
        self._fail()
