"""SymbolicPlanner.solve on the tiny grid with scripted LLM answers: the loop's branches and its records."""
import logging
import shutil

import pytest

from config import load_config
from core.domain import load_domain
from core.planner import SymbolicPlanner
from core.prism import PrismError
from core.verifier import PolicyVerifier
from fakes import ScriptedLLM, answer

needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")
pytestmark = needs_prism

EMPTY = answer()
LEFT = answer(("true", "left"))                                     # never reaches goal 1: fails in every case
PRE_G1 = answer(("!g1 & x = 0", "right"), ("!g1", "up"))            # good rules up to goal 1
POST_G1 = answer(("g1 & !g2 & y < 3", "right"), ("g1 & !g2", "down"))   # with PRE_G1: worst case passes
UNKNOWN_VAR = answer(("z > 1", "up"))
REQUIREMENTS = ["goal1", "goal2", "seq_1_before_2", "complete_sequence", "avoid_moving_seg1", "avoid_moving_seg2"]


def solve(tiny_grid, answers, rounds, *overrides):
    """Run the planner; returns (result, the scripted LLM)."""
    domain = load_domain("gridworld", ["obs_idx"])
    cfg = load_config(overrides=[f"planner.max_rounds={rounds}", "llm.seed=1", *overrides])
    llm = ScriptedLLM(answers)
    result = SymbolicPlanner(domain, llm, cfg).solve(domain.load_instances(str(tiny_grid))[0],
                                                     logging.getLogger("test_planner"))
    return result, llm


def modes(result):
    return [it["mode"] for it in result["iterations"]]


def test_extend_appends_rules_and_invalid_answers_are_retried(tiny_grid):
    result, llm = solve(tiny_grid, [EMPTY, PRE_G1, UNKNOWN_VAR, POST_G1], 4)
    assert modes(result) == ["initial", "extend", "extend"]
    assert result["success"] and result["best_case_success"]
    first, second, third = result["iterations"]
    assert first["best_case_success"] and not first["worst_case_success"] and first["kept_joint_feasible"] is True
    assert "## Uncovered states where probability is lost" in second["prompt"]
    assert [r["condition"] for r in third["rules"]] == ["!g1 & x = 0", "!g1", "g1 & !g2 & y < 3", "g1 & !g2"]
    assert third["llm_calls"] == 2 and third["invalid_answers"] == [
        "rule 1 (z > 1 -> up): unknown variable 'z' in condition 'z > 1'; allowed variables: x, y, g1, g2, obs_idx"]
    assert llm.prompts[3].startswith(llm.prompts[2]) and "## Your previous answer was invalid" in llm.prompts[3]
    assert result["final_rules"] == third["rules"] and list(result["optimum"]) == REQUIREMENTS
    assert all(it["improved"] for it in result["iterations"])
    assert {"iteration_time", "llm_time", "prism_time", "joint_time"} <= set(third)


def test_refine_blames_the_rule_that_loses_probability(tiny_grid):
    result, _ = solve(tiny_grid, [LEFT], 2)
    assert modes(result) == ["initial", "refine"]
    first, second = result["iterations"]
    assert not first["best_case_success"] and first["kept_joint_feasible"] is None   # no joint query needed
    assert "100%: rule 1 `true (action left)`" in second["prompt"]
    assert not result["success"] and result["final_rules"] == [{"condition": "true", "action": "left"}]


def test_restart_and_unparseable_rounds(tiny_grid):
    result, llm = solve(tiny_grid, [UNKNOWN_VAR, UNKNOWN_VAR, LEFT], 2, "planner.retry=always", "planner.max_fixups=1")
    assert modes(result) == ["initial", "initial"]
    first, second = result["iterations"]
    assert first["llm_calls"] == 2 and len(first["invalid_answers"]) == 2 and first["rules"] == []
    assert second["prompt"] == first["prompt"] == llm.prompts[0]
    # LEFT fails 4 requirements in the worst case, nothing fails all 6, so LEFT is kept
    assert second["improved"] and result["final_rules"] == second["rules"] == [{"condition": "true", "action": "left"}]


def test_table_feedback(tiny_grid):
    result, _ = solve(tiny_grid, [LEFT], 2, "planner.feedback=table")
    assert modes(result) == ["initial", "table"]
    prompt = result["iterations"][1]["prompt"]
    assert "## Verification results" in prompt and "Where probability is lost" not in prompt


def test_without_blame_or_joint_query(tiny_grid, monkeypatch):
    def no_joint_query(*_):
        raise AssertionError("per_requirement branch must not ask the joint query")
    monkeypatch.setattr(PolicyVerifier, "jointly_feasible", no_joint_query)
    result, _ = solve(tiny_grid, [EMPTY], 2, "feedback.blame=none", "planner.branch=per_requirement")
    assert modes(result) == ["initial", "extend"]
    assert result["iterations"][0]["kept_joint_feasible"] is None
    assert "probability is lost" not in result["iterations"][1]["prompt"]


def test_joint_conflict_refines(tiny_grid, monkeypatch):
    monkeypatch.setattr(PolicyVerifier, "jointly_feasible", lambda self, policy=None: False)
    result, _ = solve(tiny_grid, [EMPTY], 2)
    assert modes(result) == ["initial", "refine"]
    first = result["iterations"][0]
    assert first["best_case_success"] and first["kept_joint_feasible"] is False and first["branch_disagreement"]
    assert "no single way of choosing meets all of them at once" in result["iterations"][1]["prompt"]


def test_joint_query_is_cached_for_the_kept_policy(tiny_grid, monkeypatch):
    calls = []
    original = PolicyVerifier.jointly_feasible
    monkeypatch.setattr(PolicyVerifier, "jointly_feasible",
                        lambda self, policy=None: calls.append(policy) or original(self, policy))
    result, _ = solve(tiny_grid, [EMPTY, EMPTY, EMPTY], 3, "planner.retry=never")
    assert modes(result) == ["initial", "extend", "extend"]
    assert len(calls) == 1   # rounds 2 and 3 add nothing, so the kept policy and its answer stay the same
    assert [it["kept_joint_feasible"] for it in result["iterations"]] == [True, True, None]   # no query after the last round
    assert [it["improved"] for it in result["iterations"]] == [True, False, False]


@pytest.mark.parametrize("answers,kept_rules", [([LEFT], 0), ([PRE_G1, PRE_G1], 2)])
def test_prism_failure_scores_the_round_without_losing_the_instance(tiny_grid, monkeypatch, answers, kept_rules):
    """The first candidate PRISM fails on counts as the empty policy, or as the kept one when extending."""
    original, failed = PolicyVerifier.verify, []

    def verify(self, policy, analysis=True):
        if len(policy.rules) > kept_rules and not failed:
            failed.append(policy)
            raise PrismError("injected failure\ndetails")
        return original(self, policy, analysis)
    monkeypatch.setattr(PolicyVerifier, "verify", verify)
    result, _ = solve(tiny_grid, answers, len(answers))
    last = result["iterations"][-1]
    assert failed and last["invalid_answers"] == ["verification failed: injected failure"]
    assert last["num_rules"] == kept_rules
