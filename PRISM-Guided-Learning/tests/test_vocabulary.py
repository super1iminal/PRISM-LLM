"""Extended rule vocabulary: instance constants, domain features, `any` rules and general mode."""
from types import SimpleNamespace

import pytest

from core.domain import Spec
from core.rules import ANY, Constant, Feature, RuleError, SymbolicPolicy, Variable, Vocabulary

VARS = [Variable("x", "int", 0, 3), Variable("y", "int", 0, 3), Variable("g1", "bool")]
ACTIONS = ["up", "down"]
VOCAB = Vocabulary(
    constants=[Constant("GR", 2, "goal row")],
    features=[Feature("at_goal_row", "agent on the goal row", expr="x = GR"),
              Feature("toward", "the move goes towards the goal row", per_action={"up": "GR < x", "down": "GR > x"}),
              Feature("wall", "the move leaves the grid", per_action={"up": "x = 0", "down": "x = 3"})],
    allow_any=True)


def policy(rules, vocab=VOCAB):
    return SymbolicPolicy.from_raw(VARS, ACTIONS, rules, vocab)


def test_constants_and_state_features_resolve():
    p = policy([("x = GR & y = 1", "up"), ("at_goal_row", "down")])
    assert p.first_match({"x": 2, "y": 1, "g1": False}) == 0
    assert p.first_match({"x": 2, "y": 3, "g1": False}) == 1
    assert p.first_match({"x": 1, "y": 3, "g1": False}) is None


def test_action_feature_means_the_rules_action():
    p = policy([("toward", "down")])
    assert p.allowed({"x": 0, "y": 0, "g1": False}) == ["down"]      # GR = 2 > 0: down goes towards it
    assert p.allowed({"x": 3, "y": 0, "g1": False}) is None          # down goes away: the rule does not decide


def test_any_allows_every_action_whose_condition_holds():
    p = policy([("!wall", ANY), ("true", "up")])
    assert p.allowed({"x": 1, "y": 0, "g1": False}) == ["up", "down"]
    assert p.allowed({"x": 0, "y": 0, "g1": False}) == ["down"]      # up would leave the grid
    p = policy([("toward & !wall", ANY), ("true", "up")])
    assert p.allowed({"x": 2, "y": 0, "g1": False}) == ["up"]        # on the goal row: rule 1 allows nothing, rule 2 decides


def test_prism_module_defines_each_used_feature_once():
    module = policy([("toward & !wall", ANY), ("at_goal_row", "up")]).to_prism_module(200_000)
    for name in ("feature_toward_up", "feature_toward_down", "feature_wall_up", "feature_wall_down", "feature_at_goal_row"):
        assert module.count(f"formula {name} =") == 1
    assert "formula feature_toward_up = (2 < x);" in module                # the constant is replaced by its value
    up_guard = next(line.strip() for line in module.splitlines() if line.strip().startswith("[up]"))
    assert up_guard.startswith("[up] ((feature_toward_up & !(feature_wall_up))) | (feature_at_goal_row) | !(")


def test_general_mode_rejects_instance_numbers():
    general = Vocabulary(VOCAB.constants, VOCAB.features, allow_any=True, general=True)
    policy([("x = GR & y = 1 & !g1", "up"), ("x = 0", "down")], general)   # 0 and 1 are fine
    with pytest.raises(RuleError, match="number 2 is not allowed.*GR"):
        policy([("x = 2", "up")], general)


def test_base_language_unchanged():
    with pytest.raises(RuleError, match="unknown action 'any'"):
        policy([("true", ANY)], Vocabulary())
    with pytest.raises(RuleError, match="allowed variables: x, y, g1"):
        policy([("GR = 2", "up")], Vocabulary())


def test_spec_vocabulary_follows_the_run_config():
    spec = Spec(VARS, {a: a for a in ACTIONS}, [], VOCAB.constants, VOCAB.features)
    off = SimpleNamespace(extended=False, general=False, hidden_features=[])
    on = SimpleNamespace(extended=True, general=True, hidden_features=["wall"])
    assert spec.vocabulary(off) == Vocabulary()
    vocab = spec.vocabulary(on)
    assert vocab.allow_any and vocab.general and [f.name for f in vocab.features] == ["at_goal_row", "toward"]
