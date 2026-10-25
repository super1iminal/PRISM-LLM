"""The fast checker (`Domain.fast_checker`): exact worst-case values of rule lists on one instance, in numpy, for the
places where the loop scores many candidate lists without PRISM (`planner.repair`, `planner.pick: candidates`, the
pick's narrowing). The same for every domain: the instance's model is the one PRISM builds and exports
(core/explicit.py), and the rules' allowed actions per state come from core/vector.py (the PRISM encoding's
semantics). Lists it picks are verified by PRISM.

Requirements are `P` bounds on path formulas over the model's labels. Each formula becomes a deterministic automaton by
formula progression, and its worst case is a reachability probability on the product of the model with the
automaton:
    co-safe formulas (F, U, X, &, | over state formulas, e.g. gridworld's nested `U` and their conjunction): the
        probability of reaching the automaton's `true`;
    safety formulas (G, X, &, | over state formulas, e.g. `G (seg => !obs)`): one minus that of reaching `false`.
State formulas are boolean combinations of labels (`!`, `&`, `|`, `=>`). Rewards, step bounds, conditions on variables
and formulas that are neither co-safe nor safe are not supported (`supported`); a domain with such a requirement gets
no fast checker, and the loop falls back to PRISM.

Each probability is solved exactly: graph precomputation (Prob0A when maximizing, Prob0E when minimizing) and then
policy iteration with sparse linear solves, starting from an attractor strategy when maximizing so that every system
is regular.
"""
import re
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from core.explicit import ExplicitMDP, Model, _build, masked_part

MAX_AUTOMATON = 1000          # automaton states before a formula counts as unsupported
MAX_POLICY_ITERS = 1000
IMPROVE = 1e-12               # policy iteration switches an action only on a strict improvement by more than this

# ---------------------------------------------------------------- formulas

_TOKEN = re.compile(r'\s*("[^"]*"|=>|[()!&|]|\w+)')

TRUE, FALSE = ("true",), ("false",)
Formula = Tuple


def parse(formula: str) -> Formula:
    """A PRISM path formula as a tree: ("label", name), ("const", bool), ("!", f), ("&" | "|" | "=>", f, g),
    ("F" | "G" | "X", f), ("U", f, g). PRISM's precedence: `U` binds loosest, then the unary temporal operators, then
    `=>`, `|`, `&` and `!` (so `F a & b` is `F (a & b)`). NotImplementedError for anything else."""
    tokens, pos = [], 0
    while pos < len(formula.rstrip()):
        m = _TOKEN.match(formula, pos)
        if not m:
            raise NotImplementedError(f"path formula {formula!r}: only labels, !, &, |, =>, F, G, X and U")
        tokens.append(m.group(1))
        pos = m.end()
    p = _Parser(tokens, formula)
    out = p.until()
    if p.i != len(tokens):
        p.fail()
    return out


class _Parser:
    def __init__(self, tokens: List[str], formula: str):
        self.tokens, self.formula, self.i = tokens, formula, 0

    def fail(self):
        raise NotImplementedError(f"path formula {self.formula!r}: only labels, !, &, |, =>, F, G, X and U")

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def until(self):
        left = self.unary()
        if self.peek() == "U":
            self.i += 1
            return ("U", left, self.unary())
        return left

    def unary(self):
        if self.peek() in ("F", "G", "X"):
            op = self.tokens[self.i]
            self.i += 1
            return (op, self.unary())
        return self.implies()

    def implies(self):
        left = self.or_()
        if self.peek() == "=>":
            self.i += 1
            return ("=>", left, self.implies())
        return left

    def or_(self):
        out = self.and_()
        while self.peek() == "|":
            self.i += 1
            out = ("|", out, self.and_())
        return out

    def and_(self):
        out = self.not_()
        while self.peek() == "&":
            self.i += 1
            out = ("&", out, self.not_())
        return out

    def not_(self):
        if self.peek() == "!":
            self.i += 1
            return ("!", self.not_())
        return self.atom()

    def atom(self):
        tok = self.peek()
        self.i += 1
        if tok == "(":
            out = self.until()
            if self.peek() != ")":
                self.fail()
            self.i += 1
            return out
        if tok in ("true", "false"):
            return ("const", tok == "true")
        if tok and tok.startswith('"'):
            return ("label", tok[1:-1])
        self.fail()


def _temporal(f: Formula) -> bool:
    return f[0] in ("F", "G", "X", "U") or any(_temporal(g) for g in f[1:] if isinstance(g, tuple))


def _state_value(f: Formula, labels: Dict[str, np.ndarray], S: int) -> np.ndarray:
    """A state formula as a bool array over the states."""
    op = f[0]
    if op == "const":
        return np.full(S, f[1])
    if op == "label":
        if f[1] not in labels:
            raise ValueError(f"unknown label {f[1]!r}; the model has {sorted(labels)}")
        return labels[f[1]].copy()
    if op == "!":
        return ~_state_value(f[1], labels, S)
    a, b = _state_value(f[1], labels, S), _state_value(f[2], labels, S)
    return a & b if op == "&" else a | b if op == "|" else ~a | b


# ---------------------------------------------------------------- automata by formula progression
# Normal form: TRUE, FALSE, ("atom", i) for the i-th maximal state subformula, ("and" | "or", frozenset),
# ("F" | "G" | "X", f), ("U", f, g).

def _and(parts) -> Formula:
    out = set()
    for p in parts:
        if p == FALSE:
            return FALSE
        if p == TRUE:
            continue
        out |= p[1] if p[0] == "and" else {p}
    return TRUE if not out else next(iter(out)) if len(out) == 1 else ("and", frozenset(out))


def _or(parts) -> Formula:
    out = set()
    for p in parts:
        if p == TRUE:
            return TRUE
        if p == FALSE:
            continue
        out |= p[1] if p[0] == "or" else {p}
    return FALSE if not out else next(iter(out)) if len(out) == 1 else ("or", frozenset(out))


def _normal(f: Formula, atoms: List[Formula]) -> Formula:
    """`f` in normal form, its maximal state subformulas collected in `atoms`."""
    if not _temporal(f):
        if f not in atoms:
            atoms.append(f)
        return ("atom", atoms.index(f))
    op = f[0]
    if op in ("F", "G", "X"):
        return (op, _normal(f[1], atoms))
    if op == "U":
        return ("U", _normal(f[1], atoms), _normal(f[2], atoms))
    if op == "&":
        return _and([_normal(f[1], atoms), _normal(f[2], atoms)])
    if op == "|":
        return _or([_normal(f[1], atoms), _normal(f[2], atoms)])
    raise NotImplementedError("negation or implication of a temporal formula")


def _progress(f: Formula, letter: int) -> Formula:
    """What remains of `f` to hold from the next state on, after a state whose atoms are the bits of `letter`."""
    op = f[0]
    if f in (TRUE, FALSE):
        return f
    if op == "atom":
        return TRUE if letter >> f[1] & 1 else FALSE
    if op == "and":
        return _and(_progress(g, letter) for g in f[1])
    if op == "or":
        return _or(_progress(g, letter) for g in f[1])
    if op == "X":
        return f[1]
    if op == "F":
        return _or([_progress(f[1], letter), f])
    if op == "G":
        return _and([_progress(f[1], letter), f])
    return _or([_progress(f[2], letter), _and([_progress(f[1], letter), f])])      # U


def _ops(f: Formula) -> set:
    if f[0] in ("and", "or"):
        return set().union(*(_ops(g) for g in f[1]))
    if f[0] in ("F", "G", "X", "U"):
        return {f[0]}.union(*(_ops(g) for g in f[1:]))
    return set()


class Objective:
    """A requirement's path formula ready for a model: its maximal state subformulas, its normal form and whether it
    is co-safe (it holds once the automaton reaches `true`) or safe (it fails once it reaches `false`)."""

    def __init__(self, formula: str):
        self.formula = formula
        self.atoms: List[Formula] = []
        self.start = _normal(parse(formula), self.atoms)
        ops = _ops(self.start)
        if ops <= {"F", "U", "X"}:
            self.cosafe = True
        elif ops <= {"G", "X"}:
            self.cosafe = False
        else:
            raise NotImplementedError(f"path formula {formula!r}: neither co-safe nor safe")

    def automaton(self, letters: np.ndarray) -> Tuple[List[Formula], np.ndarray, np.ndarray]:
        """(the live automaton states, the first one being `start`; per live state and letter, the next state's index
        among them, or -1 for `true`, -2 for `false`; per letter, the state after reading it first) over `letters`,
        the letters that occur in the model."""
        live, index, delta = [self.start], {self.start: 0}, []

        def code(g: Formula) -> int:
            if g == TRUE:
                return -1
            if g == FALSE:
                return -2
            if g not in index:
                if len(live) >= MAX_AUTOMATON:
                    raise NotImplementedError(f"path formula {self.formula!r}: automaton too large")
                index[g] = len(live)
                live.append(g)
            return index[g]

        first = np.array([code(_progress(self.start, int(l))) for l in letters], dtype=np.int64)
        k = 0
        while k < len(live):
            delta.append([code(_progress(live[k], int(l))) for l in letters])
            k += 1
        return live, np.array(delta, dtype=np.int64).reshape(len(live), len(letters)), first


def supported(requirements) -> bool:
    """Whether `fast_checker` handles every one of `requirements` (probability bounds on supported formulas)."""
    try:
        for r in requirements:
            if r.reward is not None:
                return False
            Objective(r.formula)
    except NotImplementedError:
        return False
    return True


class Product:
    """One requirement's objective on a model: the product of the model with its automaton, the product states
    (automaton state, model state) for live automaton states and two sinks, `true` and `false`, each with a self-loop.
    A product choice (q, c) copies the model's choice c; `choices(mask)` maps a mask over the model's choices to the
    product's."""

    def __init__(self, model: Model, objective: Objective):
        S, C = model.S, model.C
        values = [_state_value(a, model.labels, S) for a in objective.atoms]
        codes = sum((v.astype(np.int64) << i for i, v in enumerate(values)), np.zeros(S, dtype=np.int64))
        letters, letter = np.unique(codes, return_inverse=True)
        letter = letter.ravel()
        live, delta, first = objective.automaton(letters)
        self.cosafe, self.live = objective.cosafe, len(live)
        L = len(live)
        yes, no = L * S, L * S + 1

        def state(q: np.ndarray, s: np.ndarray) -> np.ndarray:
            return np.where(q >= 0, q * S + s, np.where(q == -1, yes, no))

        lengths = np.diff(model.trans_start)
        t_choice = np.repeat(np.arange(C, dtype=np.int64), lengths)
        choice_state, t_choices, targets = [], [], []
        for q in range(L):
            choice_state.append(q * S + model.choice_state)
            t_choices.append(q * C + t_choice)
            targets.append(state(delta[q][letter[model.target]], model.target))
        sinks = np.array([yes, no], dtype=np.int64)
        initial = state(first[letter[[model.initial]]], np.array([model.initial]))[0]
        self.model = _build(L * S + 2, np.concatenate(choice_state + [sinks]),
                            np.r_[np.tile(model.choice_action, L), -1, -1],
                            np.concatenate(t_choices + [L * C + np.arange(2)]), np.concatenate(targets + [sinks]),
                            np.r_[np.tile(model.prob, L), 1.0, 1.0],
                            {"true": np.arange(L * S + 2) == yes, "false": np.arange(L * S + 2) == no}, int(initial))

    def choices(self, mask: np.ndarray) -> np.ndarray:
        return np.r_[np.tile(mask, self.live), True, True]

    def worst(self, mask: np.ndarray, worst_op: str) -> float:
        """The formula's worst-case probability from the initial state over the model's choices that `mask` keeps
        (`worst_op`: Pmin or Pmax), on the part of the product they reach."""
        sub = masked_part(self.model, self.choices(mask))
        if self.cosafe:
            return float(reach(sub, sub.labels["true"], maximize=worst_op == "Pmax")[sub.initial])
        return 1.0 - float(reach(sub, sub.labels["false"], maximize=worst_op == "Pmin")[sub.initial])


# ---------------------------------------------------------------- exact reachability

def _best(model: Model, Q: np.ndarray, maximize: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Per state: the best value of its choices' `Q` and the first choice reaching it."""
    best = (np.maximum if maximize else np.minimum).reduceat(Q, model.choice_start[:-1])
    hit = np.flatnonzero(Q == best[model.choice_state])
    _, first = np.unique(model.choice_state[hit], return_index=True)
    return best, hit[first]


def reach(model: Model, target: np.ndarray, maximize: bool) -> np.ndarray:
    """Per state: the maximum (or minimum) over schedulers of the probability of reaching `target`, exactly (up to the
    linear solver's rounding)."""
    S, M = model.S, model.matrix
    if maximize:            # Prob0A, and an attractor strategy: a choice one step closer to `target`
        act = np.full(S, -1, dtype=np.int64)
        done, frontier = target.copy(), target.copy()
        while frontier.any():
            hits = np.flatnonzero((M @ frontier.astype(float) > 0) & ~done[model.choice_state])
            states, first = np.unique(model.choice_state[hits], return_index=True)
            act[states] = hits[first]
            frontier = np.zeros(S, dtype=bool)
            frontier[states] = True
            done |= frontier
        zero = ~done
    else:                   # Prob0E: the states from which some scheduler avoids `target` forever
        zero = ~target
        while True:
            leaves = M @ (~zero).astype(float) > 0              # per choice: may leave `zero`
            keep = zero & np.logical_or.reduceat(~leaves, model.choice_start[:-1])
            if (keep == zero).all():
                break
            zero = keep
        act = model.choice_start[:-1].copy()
    V = target.astype(float)
    solve = np.flatnonzero(~target & ~zero)
    if not solve.size:
        return V
    for _ in range(MAX_POLICY_ITERS):
        P = M[act[solve]]
        fixed = V.copy()
        fixed[solve] = 0.0
        A = sp.identity(solve.size, format="csc") - P[:, solve].tocsc()
        V[solve] = np.atleast_1d(spsolve(A, P @ fixed))
        Q = M @ V
        best, arg = _best(model, Q, maximize)
        cur = Q[act]
        better = (best > cur + IMPROVE) if maximize else (best < cur - IMPROVE)
        better[target | zero] = False
        if not better.any():
            return V
        act[better] = arg[better]
    raise RuntimeError(f"policy iteration did not converge in {MAX_POLICY_ITERS} iterations")


# ---------------------------------------------------------------- the checker

class RuleChecker:
    """Worst-case values of rule lists (parsed with `vocabulary`) on one instance's explicit model. The policy's
    choices the rules exclude are masked; every other choice stays open to the worst case, as in PRISM."""

    def __init__(self, domain, instance, config, vocabulary):
        from core.vector import allowed_actions
        objectives = {r.name: Objective(r.formula) for r in domain.spec(instance).requirements}
        self._allowed_actions = allowed_actions
        self.mdp = ExplicitMDP(domain, instance, config)
        self.spec, self.vocabulary = self.mdp.spec, vocabulary
        self.values = {v.name: (self.mdp.obs[:, i] != 0) if v.type == "bool" else self.mdp.obs[:, i]
                       for i, v in enumerate(self.spec.variables)}
        self.cache: Dict = {}
        self.mine = self.mdp.choice_action >= 0
        self.products = {name: Product(self.mdp, objective) for name, objective in objectives.items()}

    def allowed(self, rules, order: Optional[Sequence[str]] = None):
        policy = self.spec.policy(rules, self.vocabulary)
        return self._allowed_actions(policy, self.values, self.mdp.S, self.cache, order)

    def worst(self, rules, order: Optional[Sequence[str]] = None) -> Dict[str, float]:
        """Each requirement's worst-case value from the initial state, with `any` taking the first allowed action of
        `order` if given (`core.rules.RuleError` if the rules do not parse)."""
        allowed, _ = self.allowed(rules, order)
        m = self.mdp
        mask = np.ones(m.C, dtype=bool)
        mask[self.mine] = allowed[m.choice_state[self.mine], m.choice_action[self.mine]]
        return {r.name: self.products[r.name].worst(mask, r.worst_op()) for r in self.spec.requirements}
