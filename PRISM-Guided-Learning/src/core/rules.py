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
"""
import itertools
import json
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


Expr = Union[Const, Var, Unary, Binary]

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
    def __init__(self, text: str):
        self.text = text
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
            return Const(int(tok))
        if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", tok):
            return Var(tok)
        raise RuleError(f"unexpected {tok!r} in condition {self.text!r}")


def parse_condition(text: str, variables: Dict[str, Variable]) -> Expr:
    expr = _Parser(text).parse()
    if _type_of(expr, variables, text) != "bool":
        raise RuleError(f"condition {text!r} is not a boolean expression")
    return expr


def _type_of(expr: Expr, variables: Dict[str, Variable], text: str) -> str:
    if isinstance(expr, Const):
        return "bool" if isinstance(expr.value, bool) else "int"
    if isinstance(expr, Var):
        if expr.name not in variables:
            raise RuleError(f"unknown variable {expr.name!r} in condition {text!r}; "
                            f"allowed variables: {', '.join(variables)}")
        return variables[expr.name].type
    if isinstance(expr, Unary):
        t = _type_of(expr.arg, variables, text)
        want = "bool" if expr.op == "!" else "int"
        if t != want:
            raise RuleError(f"operator {expr.op!r} needs a {want} operand in condition {text!r}")
        return want
    lt, rt = _type_of(expr.left, variables, text), _type_of(expr.right, variables, text)
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


def evaluate(expr: Expr, state: Dict[str, Value]) -> Value:
    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, Var):
        return state[expr.name]
    if isinstance(expr, Unary):
        v = evaluate(expr.arg, state)
        return (not v) if expr.op == "!" else -v
    op = expr.op
    if op == "&":
        return bool(evaluate(expr.left, state)) and bool(evaluate(expr.right, state))
    if op == "|":
        return bool(evaluate(expr.left, state)) or bool(evaluate(expr.right, state))
    if op == "=>":
        return (not evaluate(expr.left, state)) or bool(evaluate(expr.right, state))
    a, b = evaluate(expr.left, state), evaluate(expr.right, state)
    return {"+": lambda: a + b, "-": lambda: a - b, "=": lambda: a == b, "!=": lambda: a != b,
            "<": lambda: a < b, "<=": lambda: a <= b, ">": lambda: a > b, ">=": lambda: a >= b}[op]()


def to_prism(expr: Expr) -> str:
    if isinstance(expr, Const):
        return str(expr.value).lower()
    if isinstance(expr, Var):
        return expr.name
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
    action: str
    expr: Expr = field(repr=False, compare=False, default=None)

    def text(self) -> str:
        return f"{self.condition} -> {self.action}"


@dataclass
class SymbolicPolicy:
    """An ordered list of rules over `variables` choosing among `actions`."""
    variables: List[Variable]
    actions: List[str]
    rules: List[Rule] = field(default_factory=list)

    @classmethod
    def from_raw(cls, variables: Sequence[Variable], actions: Sequence[str],
                 raw_rules: Iterable[Tuple[str, str]]) -> "SymbolicPolicy":
        """Parse `(condition, action)` pairs; raises RuleError listing every bad rule."""
        policy = cls(list(variables), list(actions))
        var_map = {v.name: v for v in variables}
        errors = []
        for i, (condition, action) in enumerate(raw_rules, start=1):
            try:
                if action not in actions:
                    raise RuleError(f"unknown action {action!r}; allowed actions: {', '.join(actions)}")
                condition = _strip_action_suffix(condition.strip(), action, actions)
                policy.rules.append(Rule(condition, action, parse_condition(condition, var_map)))
            except RuleError as e:
                errors.append(f"rule {i} ({condition} -> {action}): {e}")
        if errors:
            raise RuleError("\n".join(errors))
        return policy

    def extended(self, other: "SymbolicPolicy") -> "SymbolicPolicy":
        """This policy followed by `other`'s rules (they only apply where no existing rule matches)."""
        return SymbolicPolicy(self.variables, self.actions, self.rules + other.rules)

    def first_match(self, state: Dict[str, Value]) -> Optional[int]:
        """Index of the rule deciding `state` (policy variables only), or None if uncovered."""
        for i, rule in enumerate(self.rules):
            if evaluate(rule.expr, state):
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
            state = dict(zip(names, values))
            matching = [i for i, r in enumerate(self.rules) if evaluate(r.expr, state)]
            for a, i in enumerate(matching):
                overlaps[i].update(matching[:a])
        return [sorted(o) for o in overlaps]

    def to_prism_module(self, max_enumeration: int = 200_000) -> str:
        """A variable-free PRISM module that synchronizes on every action label.

        Action `a` is enabled iff the first matching rule chooses `a`, or no rule matches.
        """
        overlaps = self._overlaps(max_enumeration)
        effective = []
        for i, rule in enumerate(self.rules):
            parts = [to_prism(rule.expr)] + [f"!{to_prism(self.rules[j].expr)}" for j in overlaps[i]]
            effective.append(" & ".join(parts))
        covered = " | ".join(to_prism(r.expr) for r in self.rules) or "false"

        lines = ["module policy"]
        for action in self.actions:
            chosen = [f"({effective[i]})" for i, r in enumerate(self.rules) if r.action == action]
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
