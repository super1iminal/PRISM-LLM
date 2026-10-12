"""LLM-written vocabulary (rules.vocabulary=llm): the vocabulary step before the rule loop, definitions added
during the loop (add-only, capped), definitions saved with the results and re-evaluated by transfer.py; and
rules.vocabulary=definitions, which must reproduce a saved run with gridworld's hand-written features."""
import dataclasses
import json
import logging
import shutil
from pathlib import Path

import pytest

import run_symbolic
from config import load_config, load_run_config
from core.domain import load_domain
from core.definitions import Definition
from core.planner import SymbolicPlanner
from fakes import TINY_GRID, ScriptedBackend, answer
from transfer import parse_targets, transfer

needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")

# Rows 1 and 2 are other 4x4 grids with two goals (see test_training.py), so definitions over the goal labels
# mean something on each.
ROWS = [TINY_GRID.splitlines()[1],
        '4,"{1: (0, 3), 2: (3, 3)}","[(0, 1)]","[(2, 1), (2, 2)]",6',
        '4,"{1: (3, 0), 2: (3, 3)}","[(1, 1)]","[(2, 2), (1, 2)]",6']

BLOCKED = {"name": "blocked", "description": "the move does not change the cell", "definition": "stays(x, y)"}
TOWARD = {"name": "toward_goal", "description": "the move gets nearer the next goal",
          "definition": '("in_seg1" & closer_to("at_goal1")) | ("in_seg2" & closer_to("at_goal2"))'}
UNSAFE = {"name": "unsafe", "description": "the move's likely outcome may meet the obstacle",
          "definition": 'next(exists obs_idx: "at_moving_obs")'}


def definitions(*defs) -> str:
    return json.dumps({"definitions": list(defs)})


def rules_with(rules, *defs) -> str:
    return json.dumps({"rules": [{"condition": c, "action": a} for c, a in rules], "definitions": list(defs)})


@pytest.fixture
def grids(tmp_path):
    path = tmp_path / "grids.csv"
    path.write_text(TINY_GRID.splitlines()[0] + "\n" + "\n".join(ROWS) + "\n", encoding="utf-8")
    return path


def llm_config(grids, *overrides):
    return load_config(overrides=[f"domain.dataset={grids.as_posix()}", "rules.extended=true", "rules.general=true",
                                  "rules.vocabulary=llm", "llm.seed=1", "planner.max_rounds=1", *overrides])


def solve(cfg, answers, instance=0):
    domain = load_domain("gridworld", cfg.domain.visible_extra)
    llm = ScriptedBackend(answers)
    result = SymbolicPlanner(domain, llm, cfg).solve(domain.load_instances(cfg.domain.dataset)[instance],
                                                     logging.getLogger("test_vocabulary_step"))
    return result, llm


def test_llm_vocabulary_needs_the_extended_language():
    with pytest.raises(ValueError, match="needs rules.extended"):
        load_config(overrides=["rules.vocabulary=llm"])


@needs_prism
def test_the_vocabulary_step_defines_features_the_rules_then_use(grids):
    result, llm = solve(llm_config(grids), [definitions(BLOCKED, TOWARD),
                                            answer(("toward_goal & !blocked", "any"), ("true", "right"))])
    vocabulary, first = llm.prompts[0], result["iterations"][0]
    assert "## Defining features" in vocabulary and '`"at_goal1"`' in vocabulary and "closer_to(T, avoid A)" in vocabulary
    assert "progress_to" not in vocabulary                         # level 1
    assert "Example rules for a different gridworld" not in vocabulary   # no hand-written feature names
    assert [d["name"] for d in result["definitions"]] == ["blocked", "toward_goal"]
    assert result["vocabulary_step"]["invalid_answers"] == [] and result["vocabulary_step"]["llm_calls"] == 1
    assert first["invalid_answers"] == [] and first["llm_calls"] == 1
    assert "`toward_goal` (per action): the move gets nearer the next goal (defined as" in first["prompt"]
    assert "You may add features with your answer" in first["prompt"]
    assert result["final_rules"][0] == {"condition": "toward_goal & !blocked", "action": "any"}


@needs_prism
def test_invalid_definitions_are_asked_again_with_the_reason(grids):
    bad = {**BLOCKED, "definition": "x = 3"}                       # a number other than 0 and 1
    result, llm = solve(llm_config(grids), [definitions(bad), definitions(BLOCKED), answer(("true", "right"))])
    assert len(result["vocabulary_step"]["invalid_answers"]) == 1
    assert "number 3 is not allowed" in llm.prompts[1] and "Your previous answer was invalid" in llm.prompts[1]
    assert [d["name"] for d in result["definitions"]] == ["blocked"]


@needs_prism
def test_rules_can_add_definitions_add_only_and_capped(grids):
    cfg = llm_config(grids, "planner.max_rounds=2", "rules.max_definitions=2", "rules.max_new_definitions=1")
    result, llm = solve(cfg, [
        definitions(BLOCKED),
        rules_with([("true", "up")], TOWARD, UNSAFE),              # two new definitions: more than one per answer
        rules_with([("toward_goal", "up")], {**TOWARD, "name": "blocked"}),   # redefines an existing feature
        rules_with([("toward_goal & blocked", "up"), ("true", "up")], TOWARD),   # accepted; "up" fails the goals
        rules_with([("true", "up")], UNSAFE),                       # over the total
        answer(("true", "up")),
    ])
    first, second = result["iterations"]
    assert len(first["invalid_answers"]) == 2 and "already used" in first["invalid_answers"][1]
    assert [d["name"] for d in first["new_definitions"]] == ["toward_goal"]
    assert [d["name"] for d in result["definitions"]] == ["blocked", "toward_goal"]
    assert "You have defined the most features allowed" in second["prompt"]
    assert "`toward_goal` (per action)" in second["prompt"]       # the feature stays in the vocabulary
    assert len(second["invalid_answers"]) == 1 and second["new_definitions"] == []


@needs_prism
def test_a_training_set_evaluates_each_definition_on_every_member(grids):
    from core.training import training_sets
    cfg = llm_config(grids)
    domain = load_domain("gridworld", cfg.domain.visible_extra)
    target, = training_sets(domain.load_instances(cfg.domain.dataset), [[0, 1]])
    llm = ScriptedBackend([definitions(BLOCKED, TOWARD), answer(("toward_goal & !blocked", "any"), ("true", "right"))])
    result = SymbolicPlanner(domain, llm, cfg).solve(target, logging.getLogger("test_vocabulary_step"))
    assert "(as on instance 1; the values differ on each instance)" in llm.prompts[0]
    assert [m["instance"] for m in result["members"]] == ["0", "1"]
    assert result["iterations"][0]["invalid_answers"] == []
    # the same text, compiled per member: grid 1's static obstacle sits elsewhere, so blocked differs
    planner = SymbolicPlanner(domain, None, cfg)
    ep = planner._episode(target, logging.getLogger("test_vocabulary_step").info)
    from core.prism import run_now
    for m in ep.members:
        run_now(m.verifier.optimum_steps())
    features = run_now(planner._evaluate(ep, [Definition.from_dict(BLOCKED)]))
    assert features[0][0].per_action != features[1][0].per_action


@needs_prism
def test_transfer_evaluates_each_rule_sets_definitions_on_every_target(tmp_path, grids, monkeypatch):
    monkeypatch.setattr(run_symbolic, "make_backend", lambda config: ScriptedBackend(
        [definitions(BLOCKED, TOWARD), answer(("toward_goal & !blocked", "any"), ("true", "right"))]))
    cfg = llm_config(grids, "run.workers=1", "run.limit=1")
    run_dir = tmp_path / "run"
    run_symbolic.run(cfg, str(run_dir))
    assert json.loads((run_dir / "outputs" / "sample_000.json").read_text())["definitions"][1]["name"] == "toward_goal"
    df, out = transfer(run_dir, targets=parse_targets("1-2"))
    assert out.name == "transfer_grids_targets1-2.csv" and sorted(df.target) == [1, 2]
    assert df.parsed.all() and set(df.columns) >= {"certified", "met"}


def test_targets_parse_ranges_and_lists():
    assert parse_targets("12-19") == list(range(12, 20))
    assert parse_targets("1, 3-4") == [1, 3, 4]
    assert parse_targets(None) is None


# ---------------------------------------------------------------- the domain's definitions, replayed

CLOSER_RUN = Path(__file__).resolve().parents[1] / "out/results/ablations/haiku_rl2_grid_large_closer/seed_none_2"


@needs_prism
@pytest.mark.parametrize("instance", ["7", "8"])   # a refine round with feedback; solved at once
def test_definitions_replay_a_saved_run_with_the_hand_written_features(instance):
    """With gridworld's definitions.yaml in place of the hand-written features, a saved Haiku v2 + closer run
    replays exactly: the same prompts, rules, PRISM values and invalid answers in every round."""
    from test_scheduler import assert_same_rounds, replay_backend
    if not (CLOSER_RUN / "config.json").exists():
        pytest.skip("saved run not available")
    cfg = load_run_config(CLOSER_RUN)
    cfg = dataclasses.replace(cfg, prism=dataclasses.replace(cfg.prism, exact_check=False),
                              rules=dataclasses.replace(cfg.rules, vocabulary="definitions"))
    sample = next(s for s in (json.loads(p.read_text()) for p in (CLOSER_RUN / "outputs").glob("sample_*.json"))
                  if str(s["instance"]) == instance)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    target = next(i for i in domain.load_instances(cfg.domain.dataset) if i.id == instance)
    result = SymbolicPlanner(domain, replay_backend({instance: sample}), cfg).solve(target, logging.getLogger("replay"))
    assert_same_rounds(result, sample)
    assert [d["name"] for d in result["definitions"]] == ["wall", "hazard", "risky", "toward", "closer", "obstacle"]
