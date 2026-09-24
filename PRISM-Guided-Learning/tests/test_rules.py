import itertools

import pytest

from core.rules import RuleError, SymbolicPolicy, Variable, atomic_rules, evaluate, parse_condition, to_prism

VARS = [Variable("x", "int", 0, 3), Variable("y", "int", 0, 3), Variable("g1", "bool")]
VAR_MAP = {v.name: v for v in VARS}
ACTIONS = ["up", "down"]


def test_parse_normalizes_llm_spellings():
    a = parse_condition("x == 2 and not g1 || y >= 3", VAR_MAP)
    b = parse_condition("(x = 2 & !g1) | y >= 3", VAR_MAP)
    assert to_prism(a) == to_prism(b)


def test_precedence_and_arithmetic():
    expr = parse_condition("!g1 & x + y < 3 | y = 3", VAR_MAP)
    assert evaluate(expr, {"x": 1, "y": 1, "g1": False})
    assert not evaluate(expr, {"x": 1, "y": 1, "g1": True})
    assert evaluate(expr, {"x": 3, "y": 3, "g1": True})


@pytest.mark.parametrize("bad", ["z > 1", "x + g1 > 0", "x <", "g1 < 2", "x", ""])
def test_invalid_conditions(bad):
    with pytest.raises(RuleError):
        parse_condition(bad, VAR_MAP)


def test_trailing_action_is_stripped_or_rejected():
    policy = SymbolicPolicy.from_raw(VARS, ACTIONS, [("x = 1 -> up", "up")])
    assert policy.rules[0].condition == "x = 1"
    with pytest.raises(RuleError):
        SymbolicPolicy.from_raw(VARS, ACTIONS, [("x = 1 -> down", "up")])


def test_first_match_and_coverage():
    policy = SymbolicPolicy.from_raw(VARS, ACTIONS, [("x = 0", "up"), ("x < 2", "down")])
    assert policy.first_match({"x": 0, "y": 0, "g1": False}) == 0
    assert policy.first_match({"x": 1, "y": 0, "g1": False}) == 1
    assert policy.first_match({"x": 2, "y": 0, "g1": False}) is None


def test_prism_module_guards_match_first_match_semantics():
    """Each action's guard must hold exactly where first-match picks it (or nothing matches)."""
    policy = SymbolicPolicy.from_raw(VARS, ACTIONS, [("x = 0 & !g1", "up"), ("y > 1", "down"), ("x < 3", "up")])
    module = policy.to_prism_module()
    guards = {}
    for line in module.splitlines():
        if line.strip().startswith("["):
            action = line.split("]")[0].strip(" [")
            guards[action] = parse_condition(line.split("]", 1)[1].rsplit("->", 1)[0], VAR_MAP)
    for x, y, g1 in itertools.product(range(4), range(4), [False, True]):
        state = {"x": x, "y": y, "g1": g1}
        rule = policy.first_match(state)
        for action in ACTIONS:
            expected = rule is None or policy.rules[rule].action == action
            assert evaluate(guards[action], state) == expected, (state, action)


def test_atomic_rules_round_trip():
    rules = atomic_rules([({"x": 1, "y": 2, "g1": True}, "down")])
    policy = SymbolicPolicy.from_raw(VARS, ACTIONS, rules)
    assert policy.first_match({"x": 1, "y": 2, "g1": True}) == 0
    assert policy.first_match({"x": 1, "y": 2, "g1": False}) is None
