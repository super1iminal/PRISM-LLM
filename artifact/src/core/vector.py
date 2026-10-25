"""Rule lists evaluated on every state of a model at once (numpy), for fast checkers that score many candidate lists
without PRISM (`planner.repair`, `planner.pick: candidates`). The semantics are those of the PRISM encoding:
the first rule that allows some action decides a state, an `any` rule allows each action whose
condition holds for it, and an undecided state allows every action.
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from core.rules import ANY, Binary, Const, FeatureCall, SymbolicPolicy, Unary, Var


def vector_value(expr, values: Dict[str, np.ndarray], size: int, cache: Dict):
    """Value of a rule-language expression in every state: `values` maps each policy variable to its array over the
    states; features are computed once per (feature, action) and kept in `cache`."""
    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, Var):
        return values[expr.name]
    if isinstance(expr, FeatureCall):
        key = (expr.name, expr.action)
        if key not in cache:
            cache[key] = np.broadcast_to(vector_value(expr.expr, values, size, cache), (size,))
        return cache[key]
    if isinstance(expr, Unary):
        v = vector_value(expr.arg, values, size, cache)
        return np.logical_not(v) if expr.op == "!" else -v
    op = expr.op
    if op in ("&", "|"):
        items, stack = [], [expr]          # long chains (features are big disjunctions), flattened without recursion
        while stack:
            e = stack.pop()
            if isinstance(e, Binary) and e.op == op:
                stack += [e.right, e.left]
            else:
                items.append(e)
        acc = None
        for e in items:
            v = vector_value(e, values, size, cache)
            acc = v if acc is None else (np.logical_and(acc, v) if op == "&" else np.logical_or(acc, v))
        return acc
    a, b = vector_value(expr.left, values, size, cache), vector_value(expr.right, values, size, cache)
    if op == "=>":
        return np.logical_or(np.logical_not(a), b)
    return {"+": np.add, "-": np.subtract, "=": np.equal, "!=": np.not_equal, "<": np.less, "<=": np.less_equal,
            ">": np.greater, ">=": np.greater_equal}[op](a, b)


def allowed_actions(policy: SymbolicPolicy, values: Dict[str, np.ndarray], size: int, cache: Dict,
                    order: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(states x actions bool: the actions the deciding rule allows, every action where no rule decides; per state the
    deciding rule's index or -1). With `order`, an `any` rule takes only the first action in `order` that it allows
    (`core.rules.fixed_order`)."""
    actions = list(policy.actions)
    allowed = np.zeros((size, len(actions)), dtype=bool)
    decider = np.full(size, -1, dtype=np.int64)
    undecided = np.ones(size, dtype=bool)
    for i, rule in enumerate(policy.rules):
        per = np.zeros((size, len(actions)), dtype=bool)
        for a, e in rule.allows.items():
            per[:, actions.index(a)] = np.broadcast_to(vector_value(e, values, size, cache), (size,))
        if order is not None and rule.action == ANY:
            first, taken = np.zeros_like(per), np.zeros(size, dtype=bool)
            for a in order:
                j = actions.index(a)
                first[:, j] = per[:, j] & ~taken
                taken |= per[:, j]
            per = first
        fires = undecided & per.any(axis=1)
        allowed[fires] = per[fires]
        decider[fires] = i
        undecided &= ~fires
    allowed[undecided] = True
    return allowed, decider
