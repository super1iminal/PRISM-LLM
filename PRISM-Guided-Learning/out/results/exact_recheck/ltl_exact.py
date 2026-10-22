"""Exact worst cases of LTL requirements PRISM's exact engine cannot take directly ("Exact engine does not yet handle
LTL-style path formulas"): the model is exported explicitly (PRISM's .tra/.lab), each requirement's formula becomes a
deterministic monitor by formula progression, and the product is written back as a PRISM-language model with
rational probabilities, on which every requirement is a plain reachability query the exact engine answers.

Fragment: atoms "label", !, &, |, =>, U, F, G. A formula with G must be a safety formula (G, &, |, propositional):
it holds iff its monitor never reaches false, so its value is 1 - P[F false]. Any other is co-safe (U, F, &, |,
propositional): it holds iff its monitor reaches true.
"""
import re
from fractions import Fraction
from pathlib import Path

TOKEN = re.compile(r'\s*("[^"]+"|=>|[!&|()]|U\b|F\b|G\b|true\b|false\b)')


def parse(text: str):
    toks, pos = [], 0
    text = text.strip()
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m:
            raise ValueError(f"cannot parse {text!r} at {pos}")
        toks.append(m.group(1)); pos = m.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
    toks.append(None)
    i = 0

    def peek():
        return toks[i]

    def take(t=None):
        nonlocal i
        tok = toks[i]
        if t is not None and tok != t:
            raise ValueError(f"expected {t} in {text!r}, got {tok}")
        i += 1
        return tok

    def until():                      # lowest: right-associative U
        left = imp()
        if peek() == "U":
            take(); return ("U", left, until())
        return left

    def imp():
        left = disj()
        if peek() == "=>":
            take(); return ("or", ("not", left), imp())
        return left

    def disj():
        left = conj()
        while peek() == "|":
            take(); left = ("or", left, conj())
        return left

    def conj():
        left = unary()
        while peek() == "&":
            take(); left = ("and", left, unary())
        return left

    def unary():
        t = peek()
        if t == "!":
            take(); return ("not", unary())
        if t in ("F", "G"):
            take(); return (t, unary())
        if t == "(":
            take(); f = until(); take(")"); return f
        if t in ("true", "false"):
            take(); return t == "true"
        if t and t.startswith('"'):
            take(); return ("ap", t[1:-1])
        raise ValueError(f"unexpected {t} in {text!r}")

    f = until()
    take(None)
    return f


def temporal(f) -> set:
    if isinstance(f, bool) or f[0] == "ap":
        return set()
    return ({f[0]} if f[0] in ("U", "F", "G") else set()) | set().union(*(temporal(g) for g in f[1:]))


def labels_of(f) -> set:
    if isinstance(f, bool):
        return set()
    if f[0] == "ap":
        return {f[1]}
    return set().union(*(labels_of(g) for g in f[1:]))


# Residual obligations in disjunctive normal form over the formula's temporal subformulas (U, F, G nodes): a
# frozenset of clauses, each a frozenset of nodes that must all hold from the next position on. Finitely many exist.
TRUE, FALSE = frozenset({frozenset()}), frozenset()


def d_or(a, b):
    clauses = a | b
    return frozenset(c for c in clauses if not any(o < c for o in clauses))     # drop absorbed clauses


def d_and(a, b):
    return d_or(frozenset(), frozenset(x | y for x in a for y in b))


def d_node(node):
    return frozenset({frozenset({node})})


def prog(f, letter: frozenset):
    """Formula progression: what the rest of the word must satisfy after reading `letter` (the labels true now)."""
    if isinstance(f, bool):
        return TRUE if f else FALSE
    op = f[0]
    if op == "ap":
        return TRUE if f[1] in letter else FALSE
    if op == "not":
        g = prog(f[1], letter)
        if g not in (TRUE, FALSE) or temporal(f[1]):
            raise ValueError("negation of a temporal formula is outside the fragment")
        return FALSE if g == TRUE else TRUE
    if op == "and":
        return d_and(prog(f[1], letter), prog(f[2], letter))
    if op == "or":
        return d_or(prog(f[1], letter), prog(f[2], letter))
    if op == "U":
        return d_or(prog(f[2], letter), d_and(prog(f[1], letter), d_node(f)))
    if op == "F":
        return d_or(prog(f[1], letter), d_node(f))
    if op == "G":
        return d_and(prog(f[1], letter), d_node(f))
    raise ValueError(op)


def step(residual, letter: frozenset):
    """Progression of a residual (DNF over nodes) through `letter`."""
    out = FALSE
    for clause in residual:
        c = TRUE
        for node in clause:
            c = d_and(c, prog(node, letter))
        out = d_or(out, c)
    return out


class Monitor:
    """Deterministic monitor of `formula` over the letters that occur: states are residual formulas; state 0 is
    the residual after the initial state's letter."""

    def __init__(self, formula: str, letters, initial_letter):
        self.f = parse(formula)
        kinds = temporal(self.f)
        self.safety = "G" in kinds
        if self.safety and kinds - {"G"}:
            raise ValueError(f"mixed safety/co-safe formula outside the fragment: {formula}")
        start = prog(self.f, initial_letter)
        self.states, self.index, self.delta = [start], {start: 0}, {}
        k = 0
        while k < len(self.states):
            for L in letters:
                nxt = step(self.states[k], L)
                if nxt not in self.index:
                    self.index[nxt] = len(self.states); self.states.append(nxt)
                    if len(self.states) > 1000:
                        raise ValueError(f"monitor of {formula} exceeds 1000 states")
                self.delta[k, L] = self.index[nxt]
            k += 1
        self.target = self.index.get(FALSE if self.safety else TRUE)


def read_explicit(tra: Path, lab: Path):
    """States, per state its choices as [(target, Fraction)], the initial state and each state's labels."""
    names = dict((int(k), v.strip('"')) for k, v in re.findall(r'(\d+)=("[^"]*")', lab.read_text().splitlines()[0]))
    labels, init = {}, None
    for line in lab.read_text().splitlines()[1:]:
        s, _, ids = line.partition(":")
        got = {names[int(k)] for k in ids.split()}
        if "init" in got:
            init = int(s)
        labels[int(s)] = frozenset(got - {"init", "deadlock"})
    lines = tra.read_text().splitlines()
    S = int(lines[0].split()[0])
    choices = [dict() for _ in range(S)]
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split()
        s, c, t, p = int(parts[0]), int(parts[1]), int(parts[2]), parts[3]
        exact = Fraction(p)
        q = exact.limit_denominator(10 ** 9)
        if abs(q - exact) > Fraction(1, 10 ** 13):
            raise ValueError(f"probability {p} has no small rational")
        choices[s].setdefault(c, []).append((t, q))
    out = []
    for s in range(S):
        cs = [choices[s][c] for c in sorted(choices[s])]
        for c in cs:
            if sum(p for _, p in c) != 1:
                raise ValueError(f"state {s}: choice probabilities sum to {sum(p for _, p in c)}")
        out.append(cs)
    return out, init, [labels.get(s, frozenset()) for s in range(S)]


def product_model(choices, init, labels, reqs):
    """A PRISM-language MDP of the model with one monitor variable per requirement, and per requirement the
    property and how its value comes out of PRISM's result ((value) -> worst case)."""
    used = set().union(*(labels_of(parse(r.formula)) for r in reqs))
    lets = [frozenset(L & used) for L in labels]
    letters = sorted(set(lets), key=lambda L: sorted(L))
    lid = {L: k for k, L in enumerate(letters)}
    mons = [Monitor(r.formula, letters, lets[init]) for r in reqs]
    out = ["mdp", ""]
    for i, m in enumerate(mons):
        for L in letters:
            expr = str(m.delta[len(m.states) - 1, L])
            for k in range(len(m.states) - 2, -1, -1):
                expr = f"(q{i}={k} ? {m.delta[k, L]} : {expr})"
            out.append(f"formula d{i}_{lid[L]} = {expr};")
    out += ["", "module product", f"  s : [0..{len(choices) - 1}] init {init};"]
    out += [f"  q{i} : [0..{len(m.states) - 1}] init 0;" for i, m in enumerate(mons)]
    for s, cs in enumerate(choices):
        for c in cs:
            branches = " + ".join(
                f"{p.numerator}/{p.denominator} : (s'={t})" + "".join(f" & (q{i}'=d{i}_{lid[lets[t]]})"
                                                                       for i in range(len(mons)))
                for t, p in c)
            out.append(f"  [] s={s} -> {branches};")
    out += ["endmodule", ""]
    props, finish = [], []
    for i, (r, m) in enumerate(zip(reqs, mons)):
        if m.target is None:      # the monitor cannot decide it: never satisfied (co-safe) or never violated (safety)
            props.append(None)
            finish.append(Fraction(1) if m.safety else Fraction(0))
            continue
        worst_min = r.maximize        # worst case of a lower bound: the least probability
        if m.safety:                  # value = 1 - P[F violated]; least value = 1 - most violation
            props.append(f'{"Pmax" if worst_min else "Pmin"}=? [ F q{i}={m.target} ]')
            finish.append(lambda v: 1 - v)
        else:
            props.append(f'{"Pmin" if worst_min else "Pmax"}=? [ F q{i}={m.target} ]')
            finish.append(lambda v: v)
    return "\n".join(out), props, finish, [len(m.states) for m in mons]
