"""Vectorized evaluation of rule-language expressions over every state of an explicit model at once."""
from typing import Dict

import numpy as np

from core.rules import Binary, Const, FeatureCall, Unary, Var


class VecEval:
    """Mixin: needs `state_vars` (name -> array over states) and `S` (number of states)."""

    def veval(self, expr, cache: Dict):
        """Value of `expr` in every state (numpy array, or a scalar for constants)."""
        if isinstance(expr, Const):
            return expr.value
        if isinstance(expr, Var):
            return self.state_vars[expr.name]
        if isinstance(expr, FeatureCall):
            key = (expr.name, expr.action)
            if key not in cache:
                cache[key] = np.broadcast_to(self.veval(expr.expr, cache), (self.S,))
            return cache[key]
        if isinstance(expr, Unary):
            v = self.veval(expr.arg, cache)
            return np.logical_not(v) if expr.op == "!" else -v
        op = expr.op
        if op in ("&", "|"):
            # flatten long chains iteratively (features are big disjunctions)
            items, stack = [], [expr]
            while stack:
                e = stack.pop()
                if isinstance(e, Binary) and e.op == op:
                    stack.append(e.right)
                    stack.append(e.left)
                else:
                    items.append(e)
            acc = None
            for e in items:
                v = self.veval(e, cache)
                acc = v if acc is None else (np.logical_and(acc, v) if op == "&" else np.logical_or(acc, v))
            return acc
        a, b = self.veval(expr.left, cache), self.veval(expr.right, cache)
        if op == "=>":
            return np.logical_or(np.logical_not(a), b)
        return {"+": np.add, "-": np.subtract, "=": np.equal, "!=": np.not_equal, "<": np.less,
                "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}[op](a, b)

