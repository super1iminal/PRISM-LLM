"""Feature definitions: rule vocabulary built from what a PRISM model exposes, instead of domain Python.

A definition names a condition rules may use, like a hand-written feature, but it is written in a small
language over the model, and the same text is evaluated on each instance's own model:

    predicates    labels ("at_goal1"), state variables, the named constants rules see, numbers,
                  = != < <= > >= + -  ! & | =>, and  exists v: P / forall v: P  (P for some / every value
                  the state variable v takes in the model's reachable states)
    outcomes      next(P)        the action's most likely outcome satisfies P
                  slip(P)        some other possible outcome of the action satisfies P
                  stays(v, ...)  the most likely outcome leaves the variables v, ... unchanged
                  old(v)         inside next(...) and slip(...): v's value before the move
    distances     closer_to(T, avoid A)    the most likely outcome is strictly closer to a state where T
                                           holds, counted in most-likely moves that never pass through A
                                           (both states must be able to reach T that way)
                  progress_to(T, avoid A)  the minimum expected number of steps to T, every step spent in A
                                           costing a fixed penalty more, is lower after the move (computed
                                           by PRISM on the real MDP, slips included)

A definition with an outcome or a distance is an action feature: in a rule it is evaluated for the rule's
action. Otherwise it is a state feature. A definition can also give one expression per action
({action: text}), for features whose meaning differs by action.

Power levels cap what a definition may compute:
    0  predicates and outcomes (one step ahead)
    1  + closer_to (a deterministic shortest-path gradient, the power of gridworld's hand-written `closer`)
    2  + progress_to (expected steps on the real MDP)
Nothing a definition can say reads a requirement or another feature, so a level-1 feature solves at most one
deterministic routing sub-problem, as `closer` does.

Evaluation (`Evaluator`): the reachable states and transitions come from PRISM's export of the bare MDP
(`PolicyVerifier.optimum_steps`, which the planner runs anyway). Labels are read from the model text and
evaluated symbolically, so `exists` can look at any value of a variable. A definition becomes the table of
policy-visible states (per action) where it holds, compiled into a rule-language expression: a `Feature` the
rest of the pipeline uses exactly like a hand-written one. Where several reachable states share their
policy-visible values, the feature holds only if it holds in all of them.
"""
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple, Union

from core.domain import parse_model_variables
from core.prism import PrismResult, PrismSteps, run_now
from core.rules import Feature, Variable

LEVELS = (0, 1, 2)
DISTANCES = {"closer_to": 1, "progress_to": 2}      # construct -> the level that allows it
OUTCOMES = ("next", "slip")
KEYWORDS = {"exists", "forall", "next", "slip", "stays", "old", "avoid", "true", "false", "any", *DISTANCES}
COST_REWARD = "definition_cost"


class DefinitionError(ValueError):
    """A definition that cannot be parsed, is above the configured level, or cannot be evaluated."""


@dataclass(frozen=True)
class Definition:
    name: str
    description: str
    text: Union[str, Dict[str, str]]     # one expression, or one per action

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "definition": self.text}

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Definition":
        text = raw.get("definition")
        if isinstance(text, dict):
            text = {str(a): str(e) for a, e in text.items()}
        elif not isinstance(text, str):
            raise DefinitionError(f"definition {raw.get('name')!r}: 'definition' must be a string or "
                                  f"an object with one expression per action")
        return cls(str(raw.get("name", "")).strip(), str(raw.get("description", "")).strip(), text)

    def listing(self) -> str:
        if isinstance(self.text, str):
            return self.text
        return "; ".join(f"{a}: {e}" for a, e in self.text.items())


# ---------------------------------------------------------------- syntax

@dataclass(frozen=True)
class Num:
    value: Union[int, bool]


@dataclass(frozen=True)
class Name:
    name: str            # a state variable or a constant


@dataclass(frozen=True)
class Label:
    name: str


@dataclass(frozen=True)
class Not:
    arg: "Node"


@dataclass(frozen=True)
class Neg:
    arg: "Node"


@dataclass(frozen=True)
class Op:
    op: str              # + - = != < <= > >= & | =>
    left: "Node"
    right: "Node"


@dataclass(frozen=True)
class Quant:
    kind: str            # exists | forall
    var: str
    body: "Node"


@dataclass(frozen=True)
class Old:
    var: str


@dataclass(frozen=True)
class Outcome:
    kind: str            # next | slip
    body: "Node"


@dataclass(frozen=True)
class Stays:
    vars: Tuple[str, ...]


@dataclass(frozen=True)
class Distance:
    kind: str            # closer_to | progress_to
    target: "Node"
    avoid: "Node"


Node = Union[Num, Name, Label, Not, Neg, Op, Quant, Old, Outcome, Stays, Distance]

_TOKEN = re.compile(r'\s*("[^"]*"|=>|<=|>=|!=|==|&&|\|\||[()!&|<>=+\-,:]|\d+|[A-Za-z_][A-Za-z_0-9]*)')
_SPELLINGS = {"==": "=", "&&": "&", "||": "|", "and": "&", "or": "|", "not": "!", "True": "true", "False": "false"}


def _tokens(text: str) -> List[str]:
    out, pos, text = [], 0, text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise DefinitionError(f"unexpected character {text[pos:].strip()[:1]!r} in {text!r}")
        tok = m.group(1)
        out.append(_SPELLINGS.get(tok, tok))
        pos = m.end()
    return out


class _Parser:
    """Precedence, low to high: =>, |, &, !, comparison, + -, unary -, atoms. A quantifier's body extends as
    far right as it can; parenthesize to stop it."""

    def __init__(self, text: str, general: bool):
        self.text, self.general = text, general
        self.toks, self.i = _tokens(text), 0

    def peek(self) -> Optional[str]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, expected: Optional[str] = None) -> str:
        tok = self.peek()
        if tok is None or (expected is not None and tok != expected):
            raise DefinitionError(f"expected {expected or 'more input'!r} in {self.text!r}")
        self.i += 1
        return tok

    def parse(self) -> Node:
        if not self.toks:
            raise DefinitionError("empty definition")
        node = self.implies()
        if self.peek() is not None:
            raise DefinitionError(f"unexpected {self.peek()!r} in {self.text!r}")
        return node

    def implies(self) -> Node:
        left = self.or_()
        if self.peek() == "=>":
            self.take()
            return Op("=>", left, self.implies())
        return left

    def or_(self) -> Node:
        node = self.and_()
        while self.peek() == "|":
            self.take()
            node = Op("|", node, self.and_())
        return node

    def and_(self) -> Node:
        node = self.not_()
        while self.peek() == "&":
            self.take()
            node = Op("&", node, self.not_())
        return node

    def not_(self) -> Node:
        if self.peek() == "!":
            self.take()
            return Not(self.not_())
        if self.peek() in ("exists", "forall"):
            kind = self.take()
            var = self.ident()
            self.take(":")
            return Quant(kind, var, self.implies())
        return self.comparison()

    def comparison(self) -> Node:
        left = self.additive()
        if self.peek() in ("=", "!=", "<", "<=", ">", ">="):
            op = self.take()
            return Op(op, left, self.additive())
        return left

    def additive(self) -> Node:
        node = self.unary()
        while self.peek() in ("+", "-"):
            op = self.take()
            node = Op(op, node, self.unary())
        return node

    def unary(self) -> Node:
        if self.peek() == "-":
            self.take()
            return Neg(self.unary())
        return self.atom()

    def ident(self) -> str:
        tok = self.take()
        if not re.fullmatch(r"[A-Za-z_]\w*", tok) or tok in KEYWORDS:
            raise DefinitionError(f"expected a name, got {tok!r} in {self.text!r}")
        return tok

    def atom(self) -> Node:
        tok = self.take()
        if tok == "(":
            node = self.implies()
            self.take(")")
            return node
        if tok in ("true", "false"):
            return Num(tok == "true")
        if tok.isdigit():
            if self.general and int(tok) > 1:
                raise DefinitionError(f"number {tok} is not allowed: definitions may only use the numbers 0 and 1, "
                                      f"so that they mean the same on every instance; use the named constants")
            return Num(int(tok))
        if tok.startswith('"'):
            return Label(tok.strip('"'))
        if tok in OUTCOMES:
            self.take("(")
            body = self.implies()
            self.take(")")
            return Outcome(tok, body)
        if tok == "stays":
            self.take("(")
            names = [self.ident()]
            while self.peek() == ",":
                self.take()
                names.append(self.ident())
            self.take(")")
            return Stays(tuple(names))
        if tok == "old":
            self.take("(")
            var = self.ident()
            self.take(")")
            return Old(var)
        if tok in DISTANCES:
            self.take("(")
            target, avoid = self.implies(), Num(False)
            if self.peek() == ",":
                self.take()
                if self.peek() == "avoid":
                    self.take()
                avoid = self.implies()
            self.take(")")
            return Distance(tok, target, avoid)
        if re.fullmatch(r"[A-Za-z_]\w*", tok) and tok not in KEYWORDS:
            return Name(tok)
        raise DefinitionError(f"unexpected {tok!r} in {self.text!r}")


def parse(text: str, general: bool = False) -> Node:
    """The syntax tree of one definition expression; `general` allows only the numbers 0 and 1."""
    return _Parser(text, general).parse()


def _walk(node: Node) -> Iterable[Node]:
    yield node
    for child in (getattr(node, f, None) for f in ("arg", "left", "right", "body", "target", "avoid")):
        if child is not None and hasattr(child, "__dataclass_fields__"):
            yield from _walk(child)


def level_of(node: Node) -> int:
    """The lowest power level that allows every construct in `node`."""
    return max([DISTANCES.get(n.kind, 0) for n in _walk(node) if isinstance(n, Distance)] or [0])


def is_action_node(node: Node) -> bool:
    return any(isinstance(n, (Outcome, Stays, Distance)) for n in _walk(node))


# ---------------------------------------------------------------- the model as definitions see it

_LABEL_RE = re.compile(r'^\s*label\s+"([^"]+)"\s*=\s*([^;]+);', re.M)
_CONST_RE = re.compile(r"^\s*const\s+(?:int\s+|bool\s+)?(\w+)\s*=\s*([^;]+);", re.M)


@dataclass
class _Scope:
    """Names a definition may use on one instance, and their types."""
    types: Dict[str, str]                 # state variable -> "int" | "bool"
    constants: Dict[str, int]             # the named constants rules see
    labels: Dict[str, Optional[Node]]     # label -> its expression over variables (None: cannot be read)
    actions: List[str]

    def check(self, node: Node, inside: Optional[str] = None) -> str:
        """Type of `node` ("int" | "bool"); `inside` is the enclosing outcome or distance, if any."""
        if isinstance(node, Num):
            return "bool" if isinstance(node.value, bool) else "int"
        if isinstance(node, Name):
            if node.name in self.types:
                return self.types[node.name]
            if node.name in self.constants:
                return "int"
            raise DefinitionError(f"unknown name {node.name!r}; variables: {', '.join(self.types)}; constants: "
                                  f"{', '.join(self.constants) or 'none'}")
        if isinstance(node, Label):
            if node.name not in self.labels:
                raise DefinitionError(f"unknown label \"{node.name}\"; labels: "
                                      + ", ".join(f'"{l}"' for l in self.labels))
            if self.labels[node.name] is None:
                raise DefinitionError(f"label \"{node.name}\" cannot be used: its definition in the model uses "
                                      f"something definitions cannot read")
            return "bool"
        if isinstance(node, Not):
            return self._want(node.arg, "bool", "!", inside)
        if isinstance(node, Neg):
            return self._want(node.arg, "int", "-", inside)
        if isinstance(node, Quant):
            if node.var not in self.types:
                raise DefinitionError(f"{node.kind} needs a state variable, not {node.var!r}")
            return self._want(node.body, "bool", node.kind, inside)
        if isinstance(node, Old):
            if inside not in OUTCOMES:
                raise DefinitionError("old(...) is only allowed inside next(...) or slip(...)")
            if node.var not in self.types:
                raise DefinitionError(f"old needs a state variable, not {node.var!r}")
            return self.types[node.var]
        if isinstance(node, Outcome):
            if inside is not None:
                raise DefinitionError(f"{node.kind}(...) cannot be nested inside {inside}(...)")
            return self._want(node.body, "bool", node.kind, node.kind)
        if isinstance(node, Stays):
            if inside is not None:
                raise DefinitionError(f"stays(...) cannot be nested inside {inside}(...)")
            for v in node.vars:
                if v not in self.types:
                    raise DefinitionError(f"stays needs state variables, not {v!r}")
            return "bool"
        if isinstance(node, Distance):
            if inside is not None:
                raise DefinitionError(f"{node.kind}(...) cannot be nested inside {inside}(...)")
            self._want(node.target, "bool", node.kind, node.kind)
            self._want(node.avoid, "bool", node.kind, node.kind)
            return "bool"
        lt, rt = self.check(node.left, inside), self.check(node.right, inside)
        if node.op in ("+", "-", "<", "<=", ">", ">="):
            if lt != "int" or rt != "int":
                raise DefinitionError(f"operator {node.op!r} needs integer operands")
            return "int" if node.op in ("+", "-") else "bool"
        if node.op in ("=", "!="):
            if lt != rt:
                raise DefinitionError(f"cannot compare {lt} with {rt}")
            return "bool"
        if lt != "bool" or rt != "bool":
            raise DefinitionError(f"operator {node.op!r} needs boolean operands")
        return "bool"

    def _want(self, node: Node, want: str, what: str, inside: Optional[str]) -> str:
        got = self.check(node, inside)
        if got != want:
            raise DefinitionError(f"{what} needs a {want} operand")
        return want


def model_labels(model: str, types: Dict[str, str]) -> Dict[str, Optional[Node]]:
    """The model's labels, each parsed into an expression over its state variables (model constants
    substituted). A label whose expression definitions cannot read (e.g. it uses a formula) maps to None."""
    consts: Dict[str, int] = {}
    for name, expr in _CONST_RE.findall(model):
        try:
            consts[name] = _plain_value(parse(expr), consts)
        except (DefinitionError, KeyError, TypeError):
            pass
    labels: Dict[str, Optional[Node]] = {}
    for name, expr in _LABEL_RE.findall(model):
        try:
            node = _substitute(parse(expr), consts)
            _Scope(types, {}, {}, []).check(node)
            labels[name] = node
        except DefinitionError:
            labels[name] = None
    return labels


def _plain_value(node: Node, consts: Dict[str, int]):
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Name):
        return consts[node.name]
    if isinstance(node, Neg):
        return -_plain_value(node.arg, consts)
    if isinstance(node, Op) and node.op in ("+", "-"):
        a, b = _plain_value(node.left, consts), _plain_value(node.right, consts)
        return a + b if node.op == "+" else a - b
    raise DefinitionError("not a constant expression")


def _substitute(node: Node, values: Dict[str, Any]) -> Node:
    """`node` with every Name in `values` replaced by its value."""
    if isinstance(node, Name):
        return Num(values[node.name]) if node.name in values else node
    if isinstance(node, (Not, Neg)):
        return type(node)(_substitute(node.arg, values))
    if isinstance(node, Op):
        return Op(node.op, _substitute(node.left, values), _substitute(node.right, values))
    if isinstance(node, Quant):
        return Quant(node.kind, node.var, _substitute(node.body, {k: v for k, v in values.items() if k != node.var}))
    if isinstance(node, Outcome):
        return Outcome(node.kind, _substitute(node.body, values))
    if isinstance(node, Distance):
        return Distance(node.kind, _substitute(node.target, values), _substitute(node.avoid, values))
    return node


# ---------------------------------------------------------------- evaluation

Valuation = Dict[str, Any]


class Evaluator:
    """Evaluates definitions on one instance's model and compiles them into rule-vocabulary features.

    `verifier` gives the model, the policy-visible variables, the actions and the constants rules see
    (its spec), and the bare MDP's states and transitions (its optimum run, cached).
    """

    def __init__(self, verifier, level: int = 1, general: bool = False, penalty: float = 1000.0):
        if level not in LEVELS:
            raise ValueError(f"definition level must be one of {LEVELS}, not {level!r}")
        self.verifier, self.level, self.general, self.penalty = verifier, level, general, penalty
        spec = verifier.spec
        self.actions = list(spec.actions)
        self.visible: List[Variable] = list(spec.variables)
        self._result: Optional[PrismResult] = None
        self._progress: Dict[Tuple[Node, Node], List[float]] = {}
        self._distances: Dict[Tuple[Node, Node], List[float]] = {}
        declared = parse_model_variables(verifier.model)
        self.types = {name: v.type for name, v in declared.items()}
        self.scope = _Scope(self.types, {c.name: c.value for c in spec.constants},
                            model_labels(verifier.model, self.types), self.actions)

    # ------------------------------------------------------------ checking

    def check(self, definition: Definition, taken: Iterable[str] = ()) -> Dict[Optional[str], Node]:
        """Parse and check `definition`; returns its expression per action (one, under None, for a single
        expression). Raises DefinitionError naming the definition."""
        name = definition.name
        try:
            if not re.fullmatch(r"[A-Za-z_]\w*", name) or name in KEYWORDS:
                raise DefinitionError("the name must be a word (letters, digits, _) and not a keyword")
            clashes = set(self.types) | set(self.scope.constants) | set(self.actions) | set(taken)
            if name in clashes:
                raise DefinitionError("the name is already used (a variable, constant, action or feature)")
            if isinstance(definition.text, dict):
                if set(definition.text) != set(self.actions):
                    raise DefinitionError(f"a per-action definition needs exactly the actions "
                                          f"{', '.join(self.actions)}")
                nodes = {a: parse(definition.text[a], self.general) for a in self.actions}
            else:
                nodes = {None: parse(definition.text, self.general)}
            for node in nodes.values():
                if self.scope.check(node) != "bool":
                    raise DefinitionError("a definition must be a condition (true or false)")
                if level_of(node) > self.level:
                    construct = next(n.kind for n in _walk(node) if isinstance(n, Distance)
                                     and DISTANCES[n.kind] > self.level)
                    raise DefinitionError(f"{construct}(...) needs level {DISTANCES[construct]}; this run allows "
                                          f"level {self.level}")
        except DefinitionError as e:
            raise DefinitionError(f"definition {name!r}: {e}") from None
        return nodes

    # ------------------------------------------------------------ the model's state space

    def prepare_steps(self) -> PrismSteps[None]:
        """The bare MDP's states and transitions (the verifier's cached optimum run)."""
        if self._result is None:
            _, _, self._result = yield from self.verifier.optimum_steps()
            r = self._result
            self.valuations: List[Valuation] = [dict(zip(r.variables, s)) for s in r.states]
            self.domains = {v: sorted({val[v] for val in self.valuations}, key=int) for v in r.variables}
            policy = set(self.actions)
            self._moves: List[Dict[str, Tuple[int, List[int]]]] = []
            for s, choices in enumerate(r.choices):
                moves = {}
                for c in choices:
                    if c.action not in policy:
                        continue
                    if c.action in moves:
                        raise DefinitionError(f"action {c.action} has several choices in one state; definitions "
                                              f"need one choice per action")
                    probs = sorted(c.successors, key=lambda t: -t[1])
                    if len(probs) > 1 and probs[0][1] == probs[1][1]:
                        raise DefinitionError(f"action {c.action} has no single most likely outcome")
                    moves[c.action] = (probs[0][0], [t for t, p in probs[1:] if p > 0])
                self._moves.append(moves)
            for moves in self._moves:
                for intended, _ in moves.values():
                    if r.choices[intended] and not self._moves[intended]:
                        raise DefinitionError("an action leads to a state where the policy does not choose; "
                                              "definitions do not support environment steps yet")

    def _progress_steps(self, nodes: Iterable[Node]) -> PrismSteps[None]:
        """Per-state minimum expected cost to each progress_to target (one PRISM run per target)."""
        for n in (n for node in nodes for n in _walk(node)):
            if isinstance(n, Distance) and n.kind == "progress_to" and (n.target, n.avoid) not in self._progress:
                target, avoid = self._to_prism(n.target), self._to_prism(n.avoid)
                model = (self.verifier.model + f'\nrewards "{COST_REWARD}"\n  true : 1;\n'
                         f"  {avoid} : {self.penalty};\nendrewards\n")
                prop = f'filter(printall, R{{"{COST_REWARD}"}}min=? [ F ({target}) ])'
                result = yield from self.verifier.runner.run_steps(model, [prop])
                by_state = dict(zip(result.states, result.state_values[0]))
                self._progress[(n.target, n.avoid)] = [by_state.get(s, float("inf")) for s in self._result.states]

    def features_steps(self, definitions: Sequence[Definition], taken: Iterable[str] = ()) -> PrismSteps[List[Feature]]:
        """Check each definition (its name must differ from `taken` and from the earlier ones), evaluate it on
        this instance and compile it into a Feature."""
        taken = list(taken)
        checked = [(d, self.check(d, taken + [x.name for x in definitions[:i]])) for i, d in enumerate(definitions)]
        yield from self.prepare_steps()
        yield from self._progress_steps(n for _, nodes in checked for n in nodes.values())
        return [self._compile(d, nodes) for d, nodes in checked]

    def features(self, definitions: Sequence[Definition], taken: Iterable[str] = ()) -> List[Feature]:
        return run_now(self.features_steps(definitions, taken))

    def table(self, definition: Definition) -> Dict[Optional[str], List[bool]]:
        """Whether `definition` holds in each reachable state of the bare MDP, per action (None: a state
        feature). For tests and feedback."""
        nodes = self.check(definition)
        run_now(self.prepare_steps())
        run_now(self._progress_steps(nodes.values()))
        return self._table(nodes)

    def _table(self, nodes: Dict[Optional[str], Node]) -> Dict[Optional[str], List[bool]]:
        single = nodes.get(None)
        if single is not None and not is_action_node(single):
            return {None: [bool(self._eval(single, s, None, self.valuations[s], None))
                           for s in range(len(self.valuations))]}
        out = {}
        for a in self.actions:
            node = single if single is not None else nodes[a]
            out[a] = [bool(self._eval(node, s, a, self.valuations[s], None)) for s in range(len(self.valuations))]
        return out

    def _compile(self, definition: Definition, nodes: Dict[Optional[str], Node]) -> Feature:
        table = self._table(nodes)
        if None in table:
            return Feature(definition.name, definition.description, expr=self._expression(table[None]))
        return Feature(definition.name, definition.description,
                       per_action={a: self._expression(holds) for a, holds in table.items()})

    # ------------------------------------------------------------ evaluating one state

    def _eval(self, node: Node, s: int, action: Optional[str], val: Valuation, old: Optional[Valuation]):
        """Value of `node` at reachable state `s` for `action`, reading variables from `val` (the state
        itself, or one of its outcomes inside next/slip; `old` is then the state before the move)."""
        if isinstance(node, Num):
            return node.value
        if isinstance(node, Name):
            return val[node.name] if node.name in val else self.scope.constants[node.name]
        if isinstance(node, Label):
            return self._eval(self.scope.labels[node.name], s, action, val, old)
        if isinstance(node, Not):
            return not self._eval(node.arg, s, action, val, old)
        if isinstance(node, Neg):
            return -self._eval(node.arg, s, action, val, old)
        if isinstance(node, Old):
            return old[node.var]
        if isinstance(node, Quant):
            pick = any if node.kind == "exists" else all
            return pick(self._eval(node.body, s, action, {**val, node.var: v}, old) for v in self.domains[node.var])
        if isinstance(node, Outcome):
            move = self._moves[s].get(action)
            if move is None:
                return False
            targets = [move[0]] if node.kind == "next" else move[1]
            return any(self._eval(node.body, t, action, self.valuations[t], val) for t in targets)
        if isinstance(node, Stays):
            move = self._moves[s].get(action)
            return move is not None and all(self.valuations[move[0]][v] == val[v] for v in node.vars)
        if isinstance(node, Distance):
            move = self._moves[s].get(action)
            if move is None:
                return False
            dist = (self._shortest(node.target, node.avoid) if node.kind == "closer_to"
                    else self._progress[(node.target, node.avoid)])
            return dist[s] < float("inf") and dist[move[0]] < dist[s]
        op = node.op
        if op == "&":
            return bool(self._eval(node.left, s, action, val, old)) and bool(self._eval(node.right, s, action, val, old))
        if op == "|":
            return bool(self._eval(node.left, s, action, val, old)) or bool(self._eval(node.right, s, action, val, old))
        if op == "=>":
            return (not self._eval(node.left, s, action, val, old)) or bool(self._eval(node.right, s, action, val, old))
        a, b = self._eval(node.left, s, action, val, old), self._eval(node.right, s, action, val, old)
        return {"+": lambda: a + b, "-": lambda: a - b, "=": lambda: a == b, "!=": lambda: a != b,
                "<": lambda: a < b, "<=": lambda: a <= b, ">": lambda: a > b, ">=": lambda: a >= b}[op]()

    def _shortest(self, target: Node, avoid: Node) -> List[float]:
        """Fewest most-likely moves from each reachable state to a state where `target` holds, never passing
        through a state where `avoid` holds (target states count as reached even there); inf if none."""
        key = (target, avoid)
        if key not in self._distances:
            n = len(self.valuations)
            is_target = [bool(self._eval(target, s, None, self.valuations[s], None)) for s in range(n)]
            blocked = [bool(self._eval(avoid, s, None, self.valuations[s], None)) for s in range(n)]
            preds: List[List[int]] = [[] for _ in range(n)]
            for s, moves in enumerate(self._moves):
                for intended, _ in moves.values():
                    preds[intended].append(s)
            dist = [float("inf")] * n
            queue = deque(s for s in range(n) if is_target[s])
            for s in queue:
                dist[s] = 0
            while queue:
                t = queue.popleft()
                for p in preds[t]:
                    if dist[p] == float("inf") and not blocked[p]:
                        dist[p] = dist[t] + 1
                        queue.append(p)
            self._distances[key] = dist
        return self._distances[key]

    def _to_prism(self, node: Node, env: Optional[Dict[str, Any]] = None) -> str:
        """A predicate as a PRISM expression over the state variables (labels and quantifiers expanded)."""
        env = env or {}
        if isinstance(node, Num):
            return str(node.value).lower()
        if isinstance(node, Name):
            if node.name in env:
                return str(env[node.name]).lower()
            return node.name if node.name in self.types else str(self.scope.constants[node.name])
        if isinstance(node, Label):
            return f"({self._to_prism(self.scope.labels[node.name], env)})"
        if isinstance(node, Not):
            return f"!({self._to_prism(node.arg, env)})"
        if isinstance(node, Neg):
            return f"-({self._to_prism(node.arg, env)})"
        if isinstance(node, Quant):
            joiner = " | " if node.kind == "exists" else " & "
            return "(" + joiner.join(f"({self._to_prism(node.body, {**env, node.var: v})})"
                                     for v in self.domains[node.var]) + ")"
        if isinstance(node, Op):
            return f"({self._to_prism(node.left, env)} {node.op} {self._to_prism(node.right, env)})"
        raise DefinitionError(f"{type(node).__name__} cannot appear in a progress_to target or avoid set")

    # ------------------------------------------------------------ compiling a table

    def _expression(self, holds: List[bool]) -> str:
        """A rule-language expression over the policy-visible variables that is true exactly in the reachable
        states where `holds` (a policy-visible valuation shared by several states: true only if all hold)."""
        names = [v.name for v in self.visible]
        table: Dict[Tuple, bool] = {}
        for val, h in zip(self.valuations, holds):
            key = tuple(val[n] for n in names)
            table[key] = table.get(key, True) and h
        return _factor(table, list(self.visible))


def _drop_irrelevant(table: Dict[Tuple, bool], variables: List[Variable]) -> Tuple[Dict[Tuple, bool], List[Variable]]:
    """Leave out variables the table does not depend on (over the valuations it has)."""
    i = 0
    while i < len(variables):
        reduced: Dict[Tuple, bool] = {}
        consistent = True
        for key, h in table.items():
            k = key[:i] + key[i + 1:]
            if reduced.setdefault(k, h) != h:
                consistent = False
                break
        if consistent and len(variables) > 0:
            table, variables = reduced, variables[:i] + variables[i + 1:]
        else:
            i += 1
    return table, variables


def _join(terms: List[str], op: str) -> str:
    """`terms` joined by `op`, as a balanced tree, so that long lists do not nest deeply."""
    if len(terms) == 1:
        return terms[0]
    if len(terms) <= 4:
        return f" {op} ".join(f"({t})" if " " in t else t for t in terms)
    mid = len(terms) // 2
    return f"({_join(terms[:mid], op)}) {op} ({_join(terms[mid:], op)})"


def _factor(table: Dict[Tuple, bool], variables: List[Variable]) -> str:
    table, variables = _drop_irrelevant(table, variables)
    values = set(table.values())
    if values <= {True}:
        return "true"
    if values == {False}:
        return "false"
    # Split on a boolean variable first, then on the integer variable with the fewest values
    i = min(range(len(variables)), key=lambda j: (variables[j].type != "bool", len({k[j] for k in table})))
    var, rest = variables[i], variables[:i] + variables[i + 1:]
    branches: List[Tuple[Any, str]] = []
    for value in sorted({k[i] for k in table}, key=int):
        sub = _factor({k[:i] + k[i + 1:]: h for k, h in table.items() if k[i] == value}, rest)
        branches.append((value, sub))
    terms = []
    if var.type == "bool":
        for value, sub in branches:
            if sub != "false":
                lit = var.name if value else f"!{var.name}"
                terms.append(lit if sub == "true" else f"{lit} & ({sub})")
    else:
        run: List[Tuple[Any, str]] = []
        for value, sub in branches + [(None, "false")]:
            if run and sub != run[0][1]:
                lo, hi, s = run[0][0], run[-1][0], run[0][1]
                if s != "false":
                    lit = f"{var.name} = {lo}" if lo == hi else f"{var.name} >= {lo} & {var.name} <= {hi}"
                    terms.append(lit if s == "true" else f"{lit} & ({s})")
                run = []
            run.append((value, sub))
    return _join(terms, "|") if terms else "false"


def load_definitions(raw: Iterable[Dict[str, Any]]) -> List[Definition]:
    return [Definition.from_dict(d) for d in raw]


def domain_definitions(domain, rules, visible: Iterable[str]) -> List[Definition]:
    """The definitions in `domains/<name>/definitions.yaml`, in file order, minus `rules.hidden_features` and
    those whose `needs` (variables that must be policy-visible) are not all in `visible`."""
    import yaml
    path = domain.root / "definitions.yaml"
    if not path.exists():
        raise DefinitionError(f"rules.vocabulary=definitions, but {path} does not exist")
    visible = set(visible)
    return [Definition.from_dict(d) for d in yaml.safe_load(path.read_text(encoding="utf-8")) or []
            if d["name"] not in rules.hidden_features and set(d.get("needs") or []) <= visible]


def model_label_texts(model: str) -> List[Tuple[str, str]]:
    """The model's labels with their PRISM expressions, as written, for the prompt."""
    return [(name, " ".join(expr.split())) for name, expr in _LABEL_RE.findall(model)]
