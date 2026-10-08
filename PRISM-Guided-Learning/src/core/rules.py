"""Symbolic policies: ordered rules `condition -> action` over the policy-visible state variables.

Semantics: the first rule whose condition holds in a state decides the action (decision list).
A state matched by no rule is *uncovered*; the verifier leaves every action enabled there, so
PRISM's max/min over the resulting MDP give the best/worst case over all completions.

Condition language (a small, PRISM-compatible expression subset):
    literals      3, true, false
    variables     any declared policy variable
    arithmetic    +  -  (integers)
    comparison    =  !=  <  <=  >  >=
    boolean       !  &  |  =>  and parentheses
Common LLM spellings are accepted and normalized: ==, &&, ||, and, or, not, True, False, and a
trailing "-> <action>" repeating the rule's own action.

Extended vocabulary (`Vocabulary`, off unless a run enables it):
    constants     named instance values (a goal's row, the horizon), replaced by their value
    features      conditions the domain defines (sensors); a state feature is one expression, an
                  action feature has one per action and means its value for the rule's action
    `any`         a rule action that allows every action whose condition holds, with action
                  features evaluated for each action; the rule decides a state if it allows at
                  least one action there, and every allowed action stays enabled (the verifier's
                  worst case covers all of them)
    general mode  only the numbers 0 and 1 in conditions, so rules name instance values and carry
                  over to other instances
"""
import itertools
import json
import operator
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

Value = Union[int, bool]


class RuleError(ValueError):
    """Raised for a rule that cannot be parsed or type-checked."""


@dataclass(frozen=True)
class Variable:
    name: str
    type: str                 # "int" or "bool"
    low: int = 0
    high: int = 1
    description: str = ""

    def domain(self) -> List[Value]:
        return [False, True] if self.type == "bool" else list(range(self.low, self.high + 1))

    def describe_type(self) -> str:
        return "boolean" if self.type == "bool" else f"integer {self.low}..{self.high}"


# ---------------------------------------------------------------- expression AST

@dataclass(frozen=True)
class Const:
    value: Value


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Unary:
    op: str      # "!" or "-"
    arg: "Expr"


@dataclass(frozen=True)
class Binary:
    op: str      # + - = != < <= > >= & | =>
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class ActionRef:
    name: str    # an action feature; bound to an action when its rule is compiled


@dataclass(frozen=True)
class FeatureCall:
    """A feature's value (for `action`, if it is an action feature). Compiles to a named PRISM formula
    and is evaluated once per state when a cache is given."""
    name: str
    action: Optional[str]
    expr: "Expr" = field(compare=False, repr=False)

    @property
    def formula(self) -> str:
        return f"feature_{self.name}" + (f"_{self.action}" if self.action else "")


Expr = Union[Const, Var, Unary, Binary, ActionRef, FeatureCall]

ANY = "any"    # rule action: allow every action whose condition holds


@dataclass(frozen=True)
class Constant:
    """A named instance value conditions may use, e.g. a goal's row."""
    name: str
    value: int
    description: str = ""


@dataclass(frozen=True)
class Feature:
    """A condition or value the domain computes for rules, like a sensor. A state feature has one
    expression (`expr`); an action feature has one per action (`per_action`). Expressions are in the
    rule language over the policy variables and constants."""
    name: str
    description: str = ""
    expr: Optional[str] = None
    per_action: Optional[Dict[str, str]] = None

    @property
    def is_action_feature(self) -> bool:
        return self.per_action is not None


@dataclass
class Vocabulary:
    """What conditions may refer to besides the policy variables. Empty by default: the base language."""
    constants: List[Constant] = field(default_factory=list)
    features: List[Feature] = field(default_factory=list)
    allow_any: bool = False     # rules may use the action `any`
    general: bool = False       # only the numbers 0 and 1 in conditions


@dataclass
class _Scope:
    """Name resolution for one parse: variables, constant values, compiled features, literal limit."""
    variables: Dict[str, Variable]
    constants: Dict[str, int] = field(default_factory=dict)
    state_features: Dict[str, "Expr"] = field(default_factory=dict)
    action_features: Dict[str, Dict[str, "Expr"]] = field(default_factory=dict)
    general: bool = False

    @classmethod
    def build(cls, variables: Sequence[Variable], actions: Sequence[str], vocab: Vocabulary) -> "_Scope":
        """Compile the vocabulary's feature expressions (over variables and constants) for these actions."""
        base = cls({v.name: v for v in variables}, {c.name: c.value for c in vocab.constants})
        scope = cls(base.variables, base.constants, general=vocab.general)
        for f in vocab.features:
            if f.is_action_feature:
                missing = [a for a in actions if a not in f.per_action]
                if missing:
                    raise RuleError(f"action feature {f.name!r} has no expression for {', '.join(missing)}")
                scope.action_features[f.name] = {a: _parse(f.per_action[a], base) for a in actions}
            else:
                scope.state_features[f.name] = _parse(f.expr, base)
        return scope

    def names(self) -> List[str]:
        return list(self.variables) + list(self.constants) + list(self.state_features) + list(self.action_features)


_TOKEN_RE = re.compile(r"\s*(=>|<=|>=|!=|==|&&|\|\||[()!&|<>=+\-]|\d+|[A-Za-z_][A-Za-z_0-9]*)")
_WORD_OPS = {"and": "&", "or": "|", "not": "!", "true": "true", "false": "false", "True": "true", "False": "false"}
_NORMALIZE = {"==": "=", "&&": "&", "||": "|"}


def _tokenize(text: str) -> List[str]:
    tokens, pos = [], 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            raise RuleError(f"unexpected character {text[pos:].strip()[:1]!r} in condition {text!r}")
        tok = m.group(1)
        tok = _NORMALIZE.get(tok, tok)
        tok = _WORD_OPS.get(tok, tok)
        tokens.append(tok)
        pos = m.end()
    return tokens


class _Parser:
    # precedence (low -> high): =>, |, &, !, comparison, + -, unary -, atoms
    def __init__(self, text: str, scope: Optional[_Scope] = None):
        self.text = text
        self.scope = scope
        self.tokens = _tokenize(text)
        self.i = 0

    def peek(self) -> Optional[str]:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self, expected: Optional[str] = None) -> str:
        tok = self.peek()
        if tok is None or (expected is not None and tok != expected):
            raise RuleError(f"expected {expected or 'more input'!r} in condition {self.text!r}")
        self.i += 1
        return tok

    def parse(self) -> Expr:
        if not self.tokens:
            raise RuleError("empty condition")
        expr = self.implies()
        if self.peek() is not None:
            raise RuleError(f"unexpected {self.peek()!r} in condition {self.text!r}")
        return expr

    def implies(self) -> Expr:
        left = self.or_()
        if self.peek() == "=>":
            self.take()
            return Binary("=>", left, self.implies())
        return left

    def or_(self) -> Expr:
        expr = self.and_()
        while self.peek() == "|":
            self.take()
            expr = Binary("|", expr, self.and_())
        return expr

    def and_(self) -> Expr:
        expr = self.not_()
        while self.peek() == "&":
            self.take()
            expr = Binary("&", expr, self.not_())
        return expr

    def not_(self) -> Expr:
        if self.peek() == "!":
            self.take()
            return Unary("!", self.not_())
        return self.comparison()

    def comparison(self) -> Expr:
        left = self.additive()
        if self.peek() in ("=", "!=", "<", "<=", ">", ">="):
            op = self.take()
            return Binary(op, left, self.additive())
        return left

    def additive(self) -> Expr:
        expr = self.unary_minus()
        while self.peek() in ("+", "-"):
            op = self.take()
            expr = Binary(op, expr, self.unary_minus())
        return expr

    def unary_minus(self) -> Expr:
        if self.peek() == "-":
            self.take()
            return Unary("-", self.unary_minus())
        return self.atom()

    def atom(self) -> Expr:
        tok = self.take()
        if tok == "(":
            expr = self.implies()
            self.take(")")
            return expr
        if tok == "true":
            return Const(True)
        if tok == "false":
            return Const(False)
        if tok.isdigit():
            if self.scope is not None and self.scope.general and int(tok) > 1:
                names = ", ".join(self.scope.constants) or "none"
                raise RuleError(f"number {tok} is not allowed: conditions may only use the numbers 0 and 1, so that "
                                f"the rules carry over to other instances; use the named constants ({names}) "
                                f"or the features instead")
            return Const(int(tok))
        if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", tok):
            return self._name(tok)
        raise RuleError(f"unexpected {tok!r} in condition {self.text!r}")

    def _name(self, tok: str) -> "Expr":
        scope = self.scope
        if scope is None or tok in scope.variables:
            return Var(tok)
        if tok in scope.constants:
            return Const(scope.constants[tok])
        if tok in scope.state_features:
            return FeatureCall(tok, None, scope.state_features[tok])
        if tok in scope.action_features:
            return ActionRef(tok)
        return Var(tok)   # unknown: the type check reports it with the allowed names


def _parse(text: str, scope: _Scope) -> Expr:
    expr = _Parser(text, scope).parse()
    _type_of(expr, scope, text)
    return expr


def parse_condition(text: str, variables: Dict[str, Variable], scope: Optional[_Scope] = None) -> Expr:
    """Parse and type-check a boolean condition over `variables` (and `scope`'s vocabulary, if given)."""
    scope = scope or _Scope(variables)
    expr = _Parser(text, scope).parse()
    if _type_of(expr, scope, text) != "bool":
        raise RuleError(f"condition {text!r} is not a boolean expression")
    return expr


def _type_of(expr: Expr, scope: _Scope, text: str) -> str:
    variables = scope.variables
    if isinstance(expr, Const):
        return "bool" if isinstance(expr.value, bool) else "int"
    if isinstance(expr, Var):
        if expr.name not in variables:
            names = scope.names()
            kind = "variables" if len(names) == len(variables) else "names"
            raise RuleError(f"unknown variable {expr.name!r} in condition {text!r}; "
                            f"allowed {kind}: {', '.join(names)}")
        return variables[expr.name].type
    if isinstance(expr, FeatureCall):
        return _type_of(expr.expr, scope, text)
    if isinstance(expr, ActionRef):
        types = {_type_of(e, scope, text) for e in scope.action_features[expr.name].values()}
        if len(types) != 1:
            raise RuleError(f"action feature {expr.name!r} has expressions of different types")
        return types.pop()
    if isinstance(expr, Unary):
        t = _type_of(expr.arg, scope, text)
        want = "bool" if expr.op == "!" else "int"
        if t != want:
            raise RuleError(f"operator {expr.op!r} needs a {want} operand in condition {text!r}")
        return want
    lt, rt = _type_of(expr.left, scope, text), _type_of(expr.right, scope, text)
    if expr.op in ("+", "-", "<", "<=", ">", ">="):
        if lt != "int" or rt != "int":
            raise RuleError(f"operator {expr.op!r} needs integer operands in condition {text!r}")
        return "int" if expr.op in ("+", "-") else "bool"
    if expr.op in ("=", "!="):
        if lt != rt:
            raise RuleError(f"cannot compare {lt} with {rt} in condition {text!r}")
        return "bool"
    if lt != "bool" or rt != "bool":
        raise RuleError(f"operator {expr.op!r} needs boolean operands in condition {text!r}")
    return "bool"


# The other binary operators (& | =>) short-circuit, so `evaluate` handles them itself
_BINARY_OPS = {"+": operator.add, "-": operator.sub, "=": operator.eq, "!=": operator.ne,
               "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}


def evaluate(expr: Expr, state: Dict[str, Value], cache: Optional[Dict] = None) -> Value:
    """Value of `expr` in `state`. With `cache` (one dict per state), each feature is computed once."""
    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, Var):
        return state[expr.name]
    if isinstance(expr, FeatureCall):
        if cache is None:
            return evaluate(expr.expr, state)
        key = (expr.name, expr.action)
        if key not in cache:
            cache[key] = evaluate(expr.expr, state)
        return cache[key]
    if isinstance(expr, Unary):
        v = evaluate(expr.arg, state, cache)
        return (not v) if expr.op == "!" else -v
    op = expr.op
    if op == "&":
        return bool(evaluate(expr.left, state, cache)) and bool(evaluate(expr.right, state, cache))
    if op == "|":
        return bool(evaluate(expr.left, state, cache)) or bool(evaluate(expr.right, state, cache))
    if op == "=>":
        return (not evaluate(expr.left, state, cache)) or bool(evaluate(expr.right, state, cache))
    return _BINARY_OPS[op](evaluate(expr.left, state, cache), evaluate(expr.right, state, cache))


def feature_calls(expr: Expr) -> List[FeatureCall]:
    """The features `expr` uses (bound ones; action features must be bound first)."""
    if isinstance(expr, FeatureCall):
        return [expr]
    if isinstance(expr, Unary):
        return feature_calls(expr.arg)
    if isinstance(expr, Binary):
        return feature_calls(expr.left) + feature_calls(expr.right)
    return []


def bind(expr: Expr, action: str, scope: _Scope) -> Expr:
    """`expr` with every action feature replaced by its expression for `action`."""
    if isinstance(expr, ActionRef):
        return FeatureCall(expr.name, action, scope.action_features[expr.name][action])
    if isinstance(expr, Unary):
        return Unary(expr.op, bind(expr.arg, action, scope))
    if isinstance(expr, Binary):
        return Binary(expr.op, bind(expr.left, action, scope), bind(expr.right, action, scope))
    return expr


def _any_of(exprs: List[Expr]) -> Expr:
    out = exprs[0]
    for e in exprs[1:]:
        out = Binary("|", out, e)
    return out


def to_prism(expr: Expr) -> str:
    if isinstance(expr, Const):
        return str(expr.value).lower()
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, FeatureCall):
        return expr.formula
    if isinstance(expr, Unary):
        return f"{expr.op}({to_prism(expr.arg)})"
    return f"({to_prism(expr.left)} {expr.op} {to_prism(expr.right)})"


# ---------------------------------------------------------------- policies

def _strip_action_suffix(condition: str, action: str, actions: Sequence[str]) -> str:
    m = re.fullmatch(r"(.*?)\s*->\s*([A-Za-z_]\w*)", condition, re.S)
    if not m or m.group(2) not in actions:
        return condition
    if m.group(2) != action:
        raise RuleError(f"condition ends with '-> {m.group(2)}' but the rule's action is {action!r}; "
                        f"put only the condition in 'condition'")
    return m.group(1)


@dataclass
class Rule:
    condition: str
    action: str                 # a policy action, or ANY
    expr: Expr = field(repr=False, compare=False, default=None)    # the rule decides a state where this holds
    allows: Dict[str, Expr] = field(repr=False, compare=False, default_factory=dict)  # action -> when it is allowed

    def text(self) -> str:
        return f"{self.condition} -> {self.action}"


@dataclass
class SymbolicPolicy:
    """An ordered list of rules over `variables` choosing among `actions`."""
    variables: List[Variable]
    actions: List[str]
    rules: List[Rule] = field(default_factory=list)
    vocabulary: Vocabulary = field(default_factory=Vocabulary)

    @classmethod
    def from_raw(cls, variables: Sequence[Variable], actions: Sequence[str],
                 raw_rules: Iterable[Tuple[str, str]], vocabulary: Optional[Vocabulary] = None) -> "SymbolicPolicy":
        """Parse `(condition, action)` pairs, with `vocabulary`'s names if given; raises RuleError listing
        every bad rule."""
        vocabulary = vocabulary or Vocabulary()
        policy = cls(list(variables), list(actions), vocabulary=vocabulary)
        scope = _Scope.build(variables, actions, vocabulary)
        allowed = list(actions) + ([ANY] if vocabulary.allow_any else [])
        errors = []
        for i, (condition, action) in enumerate(raw_rules, start=1):
            try:
                if action not in allowed:
                    raise RuleError(f"unknown action {action!r}; allowed actions: {', '.join(allowed)}")
                condition = _strip_action_suffix(condition.strip(), action, allowed)
                expr = parse_condition(condition, scope.variables, scope)
                allows = {a: bind(expr, a, scope) for a in (actions if action == ANY else [action])}
                policy.rules.append(Rule(condition, action, _any_of(list(allows.values())), allows))
            except RuleError as e:
                errors.append(f"rule {i} ({condition} -> {action}): {e}")
        if errors:
            raise RuleError("\n".join(errors))
        return policy

    def extended(self, other: "SymbolicPolicy") -> "SymbolicPolicy":
        """This policy followed by `other`'s rules (they only apply where no existing rule matches)."""
        return SymbolicPolicy(self.variables, self.actions, self.rules + other.rules, self.vocabulary)

    def allowed(self, state: Dict[str, Value]) -> Optional[List[str]]:
        """The actions the deciding rule allows in `state`, or None if no rule decides it."""
        i = self.first_match(state)
        if i is None:
            return None
        return [a for a, e in self.rules[i].allows.items() if evaluate(e, state)]

    def first_match(self, state: Dict[str, Value]) -> Optional[int]:
        """Index of the rule deciding `state` (policy variables only), or None if uncovered."""
        cache: Dict = {}
        for i, rule in enumerate(self.rules):
            if evaluate(rule.expr, state, cache):
                return i
        return None

    def to_dicts(self) -> List[Dict[str, str]]:
        return [{"condition": r.condition, "action": r.action} for r in self.rules]

    def listing(self) -> str:
        """One JSON object per line, numbered, in the same shape the LLM answers with."""
        return "\n".join(f'{i}. {{"condition": {json.dumps(r.condition)}, "action": {json.dumps(r.action)}}}'
                         for i, r in enumerate(self.rules, start=1)) or "(no rules)"

    # ------------------------------------------------------------ PRISM compilation

    def _space_size(self) -> int:
        size = 1
        for v in self.variables:
            size *= len(v.domain())
        return size

    def _overlaps(self, max_enumeration: int) -> List[List[int]]:
        """For each rule i, the earlier rules j < i whose conditions can hold together with rule i.

        Only these need `!c_j` in rule i's effective guard. Found by enumerating the policy-variable
        space when it is small enough; otherwise every earlier rule is assumed to overlap.
        """
        n = len(self.rules)
        if self._space_size() > max_enumeration:
            return [list(range(i)) for i in range(n)]
        overlaps = [set() for _ in range(n)]
        names = [v.name for v in self.variables]
        for values in itertools.product(*(v.domain() for v in self.variables)):
            state, cache = dict(zip(names, values)), {}
            matching = [i for i, r in enumerate(self.rules) if evaluate(r.expr, state, cache)]
            for a, i in enumerate(matching):
                overlaps[i].update(matching[:a])
        return [sorted(o) for o in overlaps]

    def to_prism_module(self, max_enumeration: int) -> str:
        """A variable-free PRISM module that synchronizes on every action label, preceded by a formula for
        each feature the rules use.

        Action `a` is enabled iff the first rule that decides the state allows `a`, or no rule decides it.
        """
        overlaps = self._overlaps(max_enumeration)
        earlier = [[f"!{to_prism(self.rules[j].expr)}" for j in overlaps[i]] for i in range(len(self.rules))]
        covered = " | ".join(to_prism(r.expr) for r in self.rules) or "false"

        used = {f.formula: f for r in self.rules for e in r.allows.values() for f in feature_calls(e)}
        lines = [f"formula {name} = {to_prism(f.expr)};" for name, f in sorted(used.items())] + ["module policy"]
        for action in self.actions:
            chosen = ["(" + " & ".join([to_prism(r.allows[action])] + earlier[i]) + ")"
                      for i, r in enumerate(self.rules) if action in r.allows]
            guard = " | ".join(chosen + [f"!({covered})"])
            lines.append(f"  [{action}] {guard} -> true;")
        lines.append("endmodule")
        return "\n".join(lines)


def atomic_rules(assignments: Iterable[Tuple[Dict[str, Value], str]]) -> List[Tuple[str, str]]:
    """Per-state assignments -> one exact-match rule each (how a legacy full policy is expressed)."""
    rules = []
    for state, action in assignments:
        cond = " & ".join(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in state.items())
        rules.append((cond or "true", action))
    return rules
