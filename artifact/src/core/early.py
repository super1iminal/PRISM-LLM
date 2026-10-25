"""The rules' moves in the first half of the training games, against the best moves there (`feedback.early`).

Training games are short, so a rule list can pass them by running out the clock: a move into a place it cannot leave
in time (in Pac-Man, the dead end) is safe late in a short game and fatal in a longer one. This check scores the move
the rules choose in every decision state of a member's bare model whose step counter (the variable `feedback.early`
names) lies in the first half of its range. Each such state is reachable under some play, as in the data 1-2-3-Go!
learns from (`baselines/decision_tree.py`), so the check uses nothing beyond the training members. A move's loss is how
much worse a requirement's optimal value is after it than in the state, |Q*(s, a) - V*(s)| (one-step regret against
the optimal strategy, summed over the probability requirements; V* is the loop's own `PolicyVerifier.optimum`), and a
state's loss is that of the worst move the rules allow there. States where every move is equally good are left out.

The loop (`core/planner.py`) uses the mean loss over the members' checked states: once the members pass, it keeps
refining while the mean exceeds `feedback.early_tolerance`, and keep-best prefers the lower mean after the training
score.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.rules import Binary, Const, FeatureCall, SymbolicPolicy, Unary, Var
from core.verifier import PolicyVerifier

TOL = 1e-9

_OPS = {"+": np.add, "-": np.subtract, "=": np.equal, "!=": np.not_equal, "<": np.less, "<=": np.less_equal,
        ">": np.greater, ">=": np.greater_equal}


def evaluate_columns(expr, columns: Dict[str, np.ndarray], n: int, cache: Dict) -> np.ndarray:
    """`core.rules.evaluate` on many states at once: `columns` holds each variable's values (booleans as 0/1)."""
    if isinstance(expr, Const):
        return np.full(n, int(expr.value))
    if isinstance(expr, Var):
        return columns[expr.name]
    if isinstance(expr, FeatureCall):
        key = (expr.name, expr.action)
        if key not in cache:
            cache[key] = evaluate_columns(expr.expr, columns, n, cache)
        return cache[key]
    if isinstance(expr, Unary):
        v = evaluate_columns(expr.arg, columns, n, cache)
        return (v == 0).astype(np.int64) if expr.op == "!" else -v
    left, right = evaluate_columns(expr.left, columns, n, cache), evaluate_columns(expr.right, columns, n, cache)
    if expr.op == "&":
        return ((left != 0) & (right != 0)).astype(np.int64)
    if expr.op == "|":
        return ((left != 0) | (right != 0)).astype(np.int64)
    if expr.op == "=>":
        return ((left == 0) | (right != 0)).astype(np.int64)
    return _OPS[expr.op](left, right).astype(np.int64)


@dataclass
class Example:
    member: str                     # the member's label
    valuation: Dict                 # policy variables
    moves: Tuple[str, ...]          # what the rules allow
    worst: str                      # the allowed move that loses most (the one the loss is about)
    best_moves: Tuple[str, ...]     # the optimal actions
    loss: float
    values: Dict[str, Tuple[float, float]]   # requirement -> (value after the rules' worst move, optimal value)


@dataclass
class RuleLoss:
    rule: Optional[int]             # the deciding rule (0-based; None: uncovered)
    place: Tuple[Tuple[str, int], ...] = ()   # the values of the grouping variables (`feedback.early_group`)
    move: Optional[str] = None      # with grouping, the action taken there (None: the rule allows several)
    losing: int = 0                 # checked states it decides (there) where its move loses
    decided: int = 0                # checked states it decides (there)
    total: float = 0.0              # summed loss
    best: Dict[str, int] = field(default_factory=dict)    # action -> losing states where it is a best move
    examples: List[Example] = field(default_factory=list)


@dataclass
class EarlyResult:
    mean: float                     # mean loss over the checked states
    losing: int                     # checked states where the rules' move loses
    states: int                     # checked states
    rules: List[RuleLoss]           # by summed loss, largest first (per rule, or per rule, place and move)


class EarlyCheck:
    """One member's checked states: their policy valuations and, per action, the loss of choosing it."""

    def __init__(self, verifier: PolicyVerifier, clock: str, label: str):
        spec = verifier.spec
        clock_var = next((v for v in spec.variables if v.name == clock), None)
        if clock_var is None:
            raise ValueError(f"feedback.early: {clock!r} is not a policy variable of {verifier.instance.id}")
        _, vectors, result = verifier.optimum()          # cached by the loop
        forced = verifier.forced_states()
        self.label, self.actions = label, list(spec.actions)
        self.reqs = [r for r in spec.requirements if r.reward is None]
        half, at = (clock_var.low + clock_var.high) / 2, result.positions[clock]
        index = {a: i for i, a in enumerate(self.actions)}
        valuations, losses, after, optimal = [], [], [], []
        for s, state in enumerate(result.states):
            if state in forced or state[at] > half:
                continue
            q = np.full((len(self.reqs), len(self.actions)), np.nan)
            for c in result.choices[s]:
                if c.action in index:
                    a = index[c.action]
                    for k, r in enumerate(self.reqs):
                        value = sum(p * vectors[r.name][t] for t, p in c.successors)
                        q[k, a] = value if np.isnan(q[k, a]) else (max if r.maximize else min)(q[k, a], value)
            v = np.array([vectors[r.name][s] for r in self.reqs])
            sign = np.array([1.0 if r.maximize else -1.0 for r in self.reqs])
            loss = np.nansum(np.maximum(sign[:, None] * (v[:, None] - q), 0.0), axis=0)
            loss[np.isnan(q).all(axis=0)] = np.inf                # an action without a choice here: never chosen
            finite = loss[np.isfinite(loss)]
            if finite.size and finite.max() > TOL:               # the move matters
                valuations.append(verifier.policy_valuation(result, s))
                losses.append(loss)
                after.append(q)
                optimal.append(v)
        self.valuations = valuations
        self.loss = np.array(losses).reshape(len(losses), len(self.actions))
        self.after = np.array(after).reshape(len(after), len(self.reqs), len(self.actions))
        self.optimal = np.array(optimal).reshape(len(optimal), len(self.reqs))
        names = [v.name for v in spec.variables]
        self.columns = {n: np.array([int(val[n]) for val in valuations], dtype=np.int64) for n in names}

    def allowed(self, policy: SymbolicPolicy) -> Tuple[np.ndarray, np.ndarray]:
        """(the deciding rule per checked state, -1 if none; the actions it allows, all where none decides)."""
        n = len(self.valuations)
        first, mask, todo, cache = np.full(n, -1), np.ones((n, len(self.actions)), dtype=bool), np.ones(n, bool), {}
        for i, rule in enumerate(policy.rules):
            if not todo.any():
                break
            allows = np.zeros((n, len(self.actions)), dtype=bool)
            for a, e in rule.allows.items():
                allows[:, self.actions.index(a)] = evaluate_columns(e, self.columns, n, cache) != 0
            hit = todo & allows.any(axis=1)
            first[hit], mask[hit] = i, allows[hit]
            todo &= ~hit
        return first, mask


def check(members: Sequence[Tuple[EarlyCheck, SymbolicPolicy]], examples: int = 3,
          group: Sequence[str] = ()) -> EarlyResult:
    """The loss of each member's rules over its checked states, pooled; per deciding rule (with `group`: per rule,
    values of the `group` variables and move taken), the summed loss, how often each action is a best move where the
    rules' move loses, and the `examples` states where it loses most."""
    rules: Dict[tuple, RuleLoss] = {}
    total, losing, states, pool = 0.0, 0, 0, []
    for early, policy in members:
        first, mask = early.allowed(policy)
        worst = np.where(mask, early.loss, -np.inf)
        move = worst.argmax(axis=1)
        loss = worst[np.arange(len(move)), move]
        total, states, losing = total + loss.sum(), states + len(loss), losing + int((loss > TOL).sum())
        for s in range(len(loss)):
            rule = None if first[s] < 0 else int(first[s])
            place = tuple((g, int(early.columns[g][s])) for g in group)
            key = (rule, place, early.actions[move[s]] if group and mask[s].sum() == 1 else None)
            r = rules.get(key) or rules.setdefault(key, RuleLoss(*key))
            r.decided += 1
            r.total += float(loss[s])
            if loss[s] > TOL:
                r.losing += 1
                for a, l in zip(early.actions, early.loss[s]):
                    if l <= TOL:
                        r.best[a] = r.best.get(a, 0) + 1
                pool.append((float(loss[s]), early, s, key, mask[s], int(move[s])))
    for loss, early, s, key, allowed, move in sorted(pool, key=lambda t: -t[0]):
        r = rules[key]
        if len(r.examples) >= examples:
            continue
        best = tuple(a for a, l in zip(early.actions, early.loss[s]) if l <= TOL)
        r.examples.append(Example(early.label, early.valuations[s],
                                  tuple(a for a, ok in zip(early.actions, allowed) if ok), early.actions[move], best, loss,
                                  {q.name: (float(early.after[s, k, move]), float(early.optimal[s, k]))
                                   for k, q in enumerate(early.reqs)}))
    ranked = sorted((r for r in rules.values() if r.total > TOL), key=lambda r: -r.total)
    return EarlyResult(total / states if states else 0.0, losing, states, ranked)
