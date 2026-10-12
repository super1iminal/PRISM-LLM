"""Feature definitions (core/definitions.py): the language, its power levels, evaluation on a model, and the
correctness check that gridworld's hand-written features, written as definitions, hold in exactly the same
states and actions on every grid of grid_20_large."""
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from config import load_config
from core.definitions import Definition, DefinitionError, Evaluator, _factor, level_of, load_definitions, parse
from core.domain import load_domain
from core.prism import Choice, PrismResult
from core.rules import Constant, Variable, Vocabulary, _Scope, evaluate, parse_condition
from core.verifier import PolicyVerifier
from settings import DOMAINS_PATH

needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")

# A corridor x = 0..3: "right" moves up one cell with probability 0.8 and stays otherwise, "left" moves down
# one cell with 0.8 and stays otherwise (both clamp at the ends). Cell 1 is a pit, cell 3 the goal.
CHAIN_MODEL = """mdp
const int N = 3;
formula far = x > 2;
module walker
  x : [0..N] init 0;
endmodule
label "goal" = x=N;
label "pit" = x=1;
label "far_away" = far;
"""


def _chain_result() -> PrismResult:
    states = [(x,) for x in range(4)]
    choices = []
    for x in range(4):
        right = [(x + 1, 0.8), (x, 0.2)] if x < 3 else [(x, 1.0)]
        left = [(x - 1, 0.8), (x, 0.2)] if x > 0 else [(x, 1.0)]
        choices.append([Choice("right", right), Choice("left", left)])
    return PrismResult(["x"], states, [], [], choices)


def chain_evaluator(level: int = 1, general: bool = False) -> Evaluator:
    result = _chain_result()

    def optimum_steps():
        return ({}, {}, result)
        yield   # a generator that runs no PRISM call

    spec = SimpleNamespace(actions={"right": "", "left": ""}, variables=[Variable("x", "int", 0, 3)],
                           constants=[Constant("G", 3, "goal cell")])
    verifier = SimpleNamespace(spec=spec, model=CHAIN_MODEL, runner=None, optimum_steps=optimum_steps)
    return Evaluator(verifier, level=level, general=general)


# ---------------------------------------------------------------- the language

def test_levels_count_the_strongest_construct():
    assert level_of(parse('next("goal") & stays(x)')) == 0
    assert level_of(parse('closer_to("goal", avoid "pit")')) == 1
    assert level_of(parse('closer_to("goal") | progress_to("goal", avoid "pit")')) == 2


@pytest.mark.parametrize("text,message", [
    ('closer_to("goal")', "needs level 1; this run allows level 0"),
])
def test_level_cap_rejects_stronger_constructs(text, message):
    with pytest.raises(DefinitionError, match=message):
        chain_evaluator(level=0).check(Definition("d", "", text))


@pytest.mark.parametrize("text,message", [
    ("x = 2", "number 2 is not allowed"),
    ("next(next(x = 0))", "cannot be nested"),
    ("old(x) = 0", "only allowed inside next"),
    ('"nope"', 'unknown label "nope"'),
    ('"far_away"', "cannot be used"),               # defined through a formula
    ("y = 0", "unknown name 'y'"),
    ("x + 1", "must be a condition"),
    ("exists G: x = G", "needs a state variable"),
    ('closer_to(x)', "closer_to needs a bool operand"),
])
def test_bad_definitions_are_rejected_with_the_reason(text, message):
    with pytest.raises(DefinitionError, match=message):
        chain_evaluator(general=True).check(Definition("d", "", text))


def test_names_and_per_action_definitions_are_checked():
    ev = chain_evaluator()
    for name in ("x", "G", "right", "next", "2x"):
        with pytest.raises(DefinitionError, match="name"):
            ev.check(Definition(name, "", "true"))
    with pytest.raises(DefinitionError, match="already used"):
        ev.check(Definition("safe", "", "true"), taken=["safe"])
    with pytest.raises(DefinitionError, match="exactly the actions"):
        ev.check(Definition("d", "", {"right": "true"}))


def test_definitions_round_trip_as_dicts():
    d = Definition("toward", "goes right", {"right": "x < G", "left": "false"})
    assert Definition.from_dict(d.to_dict()) == d
    with pytest.raises(DefinitionError):
        Definition.from_dict({"name": "d", "definition": 3})


# ---------------------------------------------------------------- evaluation on a model

def test_outcomes_read_the_most_likely_and_the_other_outcomes():
    ev = chain_evaluator()
    assert ev.table(Definition("d", "", 'next("goal")'))["right"] == [False, False, True, True]
    assert ev.table(Definition("d", "", "stays(x)"))["right"] == [False, False, False, True]
    assert ev.table(Definition("d", "", "stays(x)"))["left"] == [True, False, False, False]
    # the other outcome of a move is staying put
    assert ev.table(Definition("d", "", 'slip("pit")'))["left"] == [False, True, False, False]
    assert ev.table(Definition("d", "", "next(x > old(x))"))["right"] == [True, True, True, False]


def test_state_features_and_quantifiers():
    ev = chain_evaluator()
    assert ev.table(Definition("d", "", '"pit" | x = G'))[None] == [False, True, False, True]
    assert ev.table(Definition("d", "", 'exists x: "goal"'))[None] == [True] * 4
    assert ev.table(Definition("d", "", 'forall x: "goal"'))[None] == [False] * 4


def test_closer_to_counts_most_likely_moves_around_the_avoid_set():
    ev = chain_evaluator()
    # Without the pit every cell below the goal can approach it by moving right
    assert ev.table(Definition("d", "", 'closer_to("goal")'))["right"] == [True, True, True, False]
    # Cells 0 and 1 cannot reach the goal without passing the pit: no distance, so never closer
    avoiding = ev.table(Definition("d", "", 'closer_to("goal", avoid "pit")'))
    assert avoiding["right"] == [False, False, True, False]
    assert avoiding["left"] == [False] * 4


def test_checks_compare_a_definition_with_its_definers_expectations():
    from core.definitions import Check
    ev = chain_evaluator()
    approach = 'closer_to("goal", avoid "pit")'
    ok = Definition("d", "", approach, (Check("x = 2", True, "right"), Check("x = 0", False, "right")))
    ev.check_examples(ok)
    wrong = Definition("d", "", approach, (Check("x = 0 | x = 2", True, "right"), Check("x = 3", False, "right")))
    with pytest.raises(DefinitionError, match="check 1 expects it holds for right in x = 0 \\| x = 2, but it does "
                                              "not hold in: x=0$"):
        ev.check_examples(wrong)
    for checks, message in [
        ((Check("x = 2", True, "right"),), "give 2 to 4 checks"),
        ((Check("x = 2", True, "right"), Check("x = 3", True, "left")), "one where it does not"),
        ((Check("x = 2", True), Check("x = 0", False, "right")), "check 1 needs an action"),
        ((Check("x = 9", True, "right"), Check("x = 0", False, "right")), "no reachable state matches x = 9"),
        ((Check("next(x = 1)", True, "right"), Check("x = 0", False, "right")), "must be a condition on the state"),
    ]:
        with pytest.raises(DefinitionError, match=message):
            ev.check_examples(Definition("d", "", approach, checks))
    state_feature = Definition("d", "", '"pit"', (Check("x = 1", True), Check("x = 0", False, "left")))
    with pytest.raises(DefinitionError, match="does not depend on the action"):
        ev.check_examples(state_feature)
    ev.check_examples(Definition("d", "", '"pit"'), required=False)       # domain definitions need none
    assert Definition.from_dict(ok.to_dict()) == ok


def test_compiled_features_hold_where_the_table_does():
    ev = chain_evaluator()
    definitions = [Definition("approach", "", 'closer_to("goal", avoid "pit")'), Definition("pit_ahead", "", 'next("pit")'),
                   Definition("on_pit", "", '"pit"')]
    features = ev.features(definitions)
    assert [f.is_action_feature for f in features] == [True, True, False]
    scope = _Scope.build(ev.visible, ev.actions, Vocabulary(features=features))
    for d in definitions:
        table = ev.table(d)
        for s, val in enumerate(ev.valuations):
            for a, holds in table.items():
                expr = scope.state_features[d.name] if a is None else scope.action_features[d.name][a]
                assert bool(evaluate(expr, val)) == holds[s], (d.name, a, val)


def test_factor_reproduces_any_table_on_the_valuations_it_has():
    rng = random.Random(1)
    variables = [Variable("x", "int", 0, 6), Variable("g", "bool"), Variable("k", "int", 0, 2)]
    for _ in range(50):
        table = {(x, g, k): rng.random() < 0.4 for x in range(7) for g in (False, True) for k in range(3)
                 if rng.random() < 0.7}
        expr = parse_condition(_factor(table, variables), {v.name: v for v in variables})
        for (x, g, k), holds in table.items():
            assert bool(evaluate(expr, {"x": x, "g": g, "k": k})) == holds


# ---------------------------------------------------------------- reproducing gridworld's features

GRIDWORLD_DEFINITIONS = load_definitions(yaml.safe_load(
    (DOMAINS_PATH / "gridworld" / "definitions.yaml").read_text(encoding="utf-8")))


def test_gridworld_definitions_cover_every_hand_written_feature():
    assert [d.name for d in GRIDWORLD_DEFINITIONS] == ["wall", "hazard", "risky", "toward", "closer", "obstacle"]
    assert {d.name: level_of(parse(d.text)) for d in GRIDWORLD_DEFINITIONS if isinstance(d.text, str)}["closer"] == 1


@needs_prism
@pytest.mark.parametrize("grid", range(20))
def test_gridworld_definitions_reproduce_the_hand_written_features(grid):
    """Every definition holds in exactly the reachable states (and actions) where the hand-written feature
    does, both as evaluated and as compiled into a feature expression."""
    cfg = load_config(overrides=["rules.extended=true", "rules.general=true"])
    domain = load_domain("gridworld", cfg.domain.visible_extra)
    verifier = PolicyVerifier(domain, domain.load_instances("grid_20_large.csv")[grid], cfg)
    spec = verifier.spec
    evaluator = Evaluator(verifier, level=1, general=True)
    compiled = evaluator.features(GRIDWORLD_DEFINITIONS)
    hand = _Scope.build(spec.variables, list(spec.actions), Vocabulary(spec.constants, spec.features))
    ours = _Scope.build(spec.variables, list(spec.actions), Vocabulary(spec.constants, compiled))
    names = [v.name for v in spec.variables]
    for d in GRIDWORLD_DEFINITIONS:
        table = evaluator.table(d)
        for a in spec.actions:
            for s, val in enumerate(evaluator.valuations):
                visible = {n: val[n] for n in names}
                expected = bool(evaluate(hand.action_features[d.name][a], visible))
                assert table[a][s] == expected, (d.name, a, visible)
                assert bool(evaluate(ours.action_features[d.name][a], visible)) == expected, (d.name, a, visible)
