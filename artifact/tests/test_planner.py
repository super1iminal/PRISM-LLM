"""SymbolicPlanner.solve on the tiny grid with scripted LLM answers: the loop's branches and its records."""
import logging
import re
import shutil

import pytest

from config import load_config
from core.domain import load_domain
from core.episode import Check
from core.planner import SymbolicPlanner
from core.prism import PrismError, run_now
from core.rules import action_orders, fixed_order
from core.verifier import PolicyVerifier
from fakes import ScriptedBackend, answer

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
    llm = ScriptedBackend(answers)
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
        yield
    monkeypatch.setattr(PolicyVerifier, "jointly_feasible_steps", no_joint_query)
    result, _ = solve(tiny_grid, [EMPTY], 2, "feedback.blame=none", "planner.branch=per_requirement")
    assert modes(result) == ["initial", "extend"]
    assert result["iterations"][0]["kept_joint_feasible"] is None
    assert "probability is lost" not in result["iterations"][1]["prompt"]


def test_joint_conflict_refines(tiny_grid, monkeypatch):
    def infeasible(self, policy=None):
        return False
        yield
    monkeypatch.setattr(PolicyVerifier, "jointly_feasible_steps", infeasible)
    result, _ = solve(tiny_grid, [EMPTY], 2)
    assert modes(result) == ["initial", "refine"]
    first = result["iterations"][0]
    assert first["best_case_success"] and first["kept_joint_feasible"] is False and first["branch_disagreement"]
    assert "no single way of choosing meets all of them at once" in result["iterations"][1]["prompt"]


def test_joint_query_is_cached_for_the_kept_policy(tiny_grid, monkeypatch):
    calls, original = [], PolicyVerifier.jointly_feasible_steps

    def counted(self, policy=None):
        calls.append(policy)
        return (yield from original(self, policy))
    monkeypatch.setattr(PolicyVerifier, "jointly_feasible_steps", counted)
    result, _ = solve(tiny_grid, [EMPTY, EMPTY, EMPTY], 3, "planner.retry=never")
    assert modes(result) == ["initial", "extend", "extend"]
    assert len(calls) == 1   # rounds 2 and 3 add nothing, so the kept policy and its answer stay the same
    assert [it["kept_joint_feasible"] for it in result["iterations"]] == [True, True, None]   # no query after the last round
    assert [it["improved"] for it in result["iterations"]] == [True, False, False]


@pytest.mark.parametrize("answers,kept_rules", [([LEFT], 0), ([PRE_G1, PRE_G1], 2)])
def test_prism_failure_scores_the_round_without_losing_the_instance(tiny_grid, monkeypatch, answers, kept_rules):
    """The first candidate PRISM fails on counts as the empty policy, or as the kept one when extending."""
    original, failed = PolicyVerifier.verify_steps, []

    def verify(self, policy, analysis=True, runner=None):
        if len(policy.rules) > kept_rules and not failed:
            failed.append(policy)
            raise PrismError("injected failure\ndetails")
        return (yield from original(self, policy, analysis, runner))
    monkeypatch.setattr(PolicyVerifier, "verify_steps", verify)
    result, _ = solve(tiny_grid, answers, len(answers))
    last = result["iterations"][-1]
    assert failed and last["invalid_answers"] == ["verification failed: injected failure"]
    assert last["num_rules"] == kept_rules


def test_the_final_policy_is_checked_exactly(tiny_grid):
    result, _ = solve(tiny_grid, [PRE_G1, POST_G1], 2)
    assert result["success"] and result["final_check"] == "interval iteration"
    for case in ("best", "worst"):
        loop, final = result[f"loop_{case}"], result[f"final_{case}"]
        assert loop == result["iterations"][-1][case] and list(final) == REQUIREMENTS
        assert all(abs(final[r] - loop[r]) < 1e-6 for r in REQUIREMENTS)


def test_without_the_exact_check_the_run_reports_the_loops_values(tiny_grid):
    result, _ = solve(tiny_grid, [LEFT], 1, "prism.exact_check=false")
    assert result["final_check"] == "off"
    assert result["final_worst"] == result["loop_worst"] == result["iterations"][0]["worst"]


def test_the_exact_check_falls_back_to_lp_then_gauss_seidel_then_to_the_loops_values(tiny_grid):
    bad = "prism.exact_epsilon=bad"                     # PRISM rejects it: interval iteration fails
    result, _ = solve(tiny_grid, [LEFT], 1, bad)
    assert result["final_check"] == "linear programming"
    result, _ = solve(tiny_grid, [LEFT], 1, bad, "prism.exact_methods=[intervaliter]")
    assert result["final_check"] == "Gauss-Seidel 1e-12"
    result, _ = solve(tiny_grid, [LEFT], 1, bad, "prism.exact_methods=[intervaliter]", "prism.exact_fallback_epsilon=bad")
    assert result["final_check"] == "failed" and result["final_worst"] == result["loop_worst"]


def test_any_rules_are_resolved_into_the_best_order_on_the_members(tiny_grid):
    extended = ["rules.extended=true", "rules.general=true"]
    worst, _ = solve(tiny_grid, [answer(("true", "any"))], 1, *extended)
    assert not worst["success"] and "any_order" not in worst["iterations"][0]   # left is allowed: the worst case fails
    result, llm = solve(tiny_grid, [answer(("true", "any"))], 1, *extended, "rules.any_order=rotations")
    record = result["iterations"][0]
    assert result["success"] and record["any_order"][0] == "right"             # `true -> right` passes (fakes.py)
    assert [r["action"] for r in result["final_rules"]] == record["any_order"]
    assert "resolved into one fixed order of the actions" in llm.prompts[0]


def _orders_scored(tiny_grid, rules, *overrides):
    """(planner, episode, policies, raw rules, every order) for one candidate on the tiny grid."""
    domain = load_domain("gridworld", ["obs_idx"])
    cfg = load_config(overrides=["rules.extended=true", "rules.general=true", "rules.any_order=all", *overrides])
    planner = SymbolicPlanner(domain, None, cfg)
    ep = planner._episode(domain.load_instances(str(tiny_grid))[0], logging.getLogger("test_planner").info)
    raw = list(rules)
    return planner, ep, planner._parse(ep, raw), raw, action_orders(list(ep.primary.verifier.spec.actions), "all")


def _counting(monkeypatch):
    calls = []
    original = PolicyVerifier.verify_steps

    def verify_steps(self, policy, analysis=True, runner=None):
        calls.append(analysis)
        return (yield from original(self, policy, analysis, runner))
    monkeypatch.setattr(PolicyVerifier, "verify_steps", verify_steps)
    return calls


ANY_RULES = [("x = 0 & !wall", "any"), ("!g1 & toward", "any"), ("true", "down")]


def test_orders_that_resolve_into_the_same_model_are_verified_once(tiny_grid, monkeypatch):
    """Every order scores as if verified on its own; orders that choose alike wherever an already verified order
    reaches take its values, without another PRISM check."""
    planner, ep, policies, raw, orders = _orders_scored(tiny_grid, ANY_RULES)
    calls = _counting(monkeypatch)
    scored = run_now(planner._score_orders(ep, policies, raw, orders))
    checks = len(calls)
    assert sorted(scored) == list(range(24)) and 1 < checks < 24
    for k, order in enumerate(orders):
        alone = [run_now(m.verifier.verify_steps(p, analysis=False))
                 for m, p in zip(ep.members, planner._parse(ep, fixed_order(raw, order)))]
        assert scored[k][0] == Check(scored[k][1], alone).score(ep.members), order
        assert [p.to_dicts() for p in scored[k][1]] == [p.to_dicts() for p in planner._parse(ep, fixed_order(raw, order))]


def test_the_fast_model_prunes_orders_or_scores_them(tiny_grid, monkeypatch):
    """`rules.any_prune`: the same choice with fewer PRISM checks; `rules.any_scorer: domain`: no PRISM check, and
    the values PRISM finds."""
    planner, ep, policies, raw, orders = _orders_scored(tiny_grid, ANY_RULES)
    full = run_now(planner._score_orders(ep, policies, raw, orders))
    pick = min(full, key=lambda k: (full[k][0], k))
    pruner, ep, policies, raw, orders = _orders_scored(tiny_grid, ANY_RULES, "rules.any_prune=true")
    calls = _counting(monkeypatch)
    pruned = run_now(pruner._score_orders(ep, policies, raw, orders))
    assert min(pruned, key=lambda k: (pruned[k][0], k)) == pick and set(pruned) < set(full)
    assert all(pruned[k][0] == full[k][0] for k in pruned) and len(calls) < 24
    scorer, ep, policies, raw, orders = _orders_scored(tiny_grid, ANY_RULES, "rules.any_scorer=domain")
    calls.clear()
    fast = run_now(scorer._score_orders(ep, policies, raw, orders))
    assert not calls and sorted(fast) == list(range(24))
    for k in fast:   # PRISM's loop solver stops early: its values are within about 1e-5 of the exact ones
        for a, b in zip(fast[k][2].vs, full[k][2].vs):
            assert a.worst == pytest.approx(b.worst, abs=1e-4) and a.best == pytest.approx(b.best, abs=1e-4)


def test_any_rules_in_the_declared_order_or_with_the_best_order_reported(tiny_grid):
    extended = ["rules.extended=true", "rules.general=true"]
    declared, llm = solve(tiny_grid, [answer(("true", "any"))], 1, *extended, "rules.any_order=declared")
    assert declared["iterations"][0]["any_order"] == ["up", "right", "down", "left"]
    assert "in the fixed order up, right, down, left" in llm.prompts[0]
    reported, llm = solve(tiny_grid, [answer(("true", "any")), answer(("true", "right"))], 2, *extended,
                          "rules.any_order=report")
    first, second = reported["iterations"]
    assert "any_order" not in first and [r["action"] for r in first["rules"]] == ["any"]   # the rules stay as written
    assert "took the first action it allows in the order right" in second["prompt"]
    assert "1 of 1 instance would meet every requirement" in second["prompt"]
    assert reported["success"]


def test_a_likely_failing_trace_is_shown_when_asked(tiny_grid):
    result, _ = solve(tiny_grid, [LEFT, LEFT], 2, "feedback.trajectory=true")
    prompt = result["iterations"][1]["prompt"]
    assert "## A likely way to fail `goal1`" in prompt and "1. In x=0 & y=0" in prompt and "rule 1 chose left" in prompt
    plain, _ = solve(tiny_grid, [LEFT, LEFT], 2)
    assert "A likely way to fail" not in plain["iterations"][1]["prompt"]


def test_feature_space_feedback_names_no_position(tiny_grid):
    """With `feedback.space: features`, blame, uncovered hotspots and the trace name states by the goal flags and
    the features that hold per action, never by the agent's cell or the obstacle's phase."""
    features = ["feedback.space=features", "feedback.trajectory=true", "rules.extended=true", "rules.general=true"]
    refined, _ = solve(tiny_grid, [LEFT, LEFT], 2, *features)
    extended, _ = solve(tiny_grid, [EMPTY, EMPTY], 2, *features)
    assert modes(refined) == ["initial", "refine"] and modes(extended) == ["initial", "extend"]
    for result, section in ((refined, "## Where probability is lost"),
                            (extended, "## Uncovered states where probability is lost")):
        prompt = result["iterations"][1]["prompt"]
        feedback = prompt[prompt.index(section):prompt.index("## Task")]
        assert "## A likely way to fail" in feedback and "g1=false & g2=false | up: wall" in feedback
        assert not re.search(r"\b(x|y|obs_idx)=", feedback)


def test_optimal_feedback_names_what_the_optimal_strategy_does(tiny_grid):
    """With `feedback.optimal`, the states blame, uncovered hotspots and the trace name say what the optimal strategy
    (the gridworld trees' combined objective) does there, against the rules' action, with the features that hold for
    each of the actions named (extended vocabulary)."""
    refined, _ = solve(tiny_grid, [LEFT, LEFT], 2, "feedback.optimal=true", "feedback.trajectory=true")
    prompt = refined["iterations"][1]["prompt"]
    assert "Most in:\n  - x=0 & y=0 & g1=false & g2=false & obs_idx=0: optimal: right, not left\n" in prompt
    assert "rule 1 chose left; optimal: right, not left; this outcome" in prompt
    assert "`optimal` names what the best strategy" in prompt and "brackets" not in prompt
    extended = ["rules.extended=true", "rules.general=true", "feedback.optimal=true", "feedback.trajectory=true"]
    covered, _ = solve(tiny_grid, [EMPTY, EMPTY], 2, *extended)
    prompt = covered["iterations"][1]["prompt"]
    assert "%: x=0 & y=0 & g1=false & g2=false & obs_idx=0: optimal: right [right: toward, closer]\n" in prompt
    assert "up (no rule decides here); optimal: right, not up [right: toward, closer; up: wall]" in prompt
    assert "brackets list the features" in prompt
    plain, _ = solve(tiny_grid, [LEFT, LEFT], 2, "feedback.trajectory=true")
    assert "optimal" not in plain["iterations"][1]["prompt"]


def test_demonstration_shows_one_optimal_run(tiny_grid):
    """With `prompt.demonstration`, the initial prompt shows the optimal strategy's run along the outcomes most likely
    to succeed, until the last goal is reached: per decision the state (with the extended vocabulary, the features
    per action) and the optimal action."""
    result, llm = solve(tiny_grid, [LEFT], 1, "prompt.demonstration=true")
    prompt = llm.prompts[0]
    run = prompt[prompt.index("## An optimal run on this instance"):prompt.index("## Output")]
    assert "1. x=0 & y=0 & g1=false & g2=false & obs_idx=0 -> right\n" in run
    assert "6. x=2 & y=3 & g1=true & g2=false & obs_idx=1 -> down\n\n" in run and "7." not in run
    _, llm = solve(tiny_grid, [LEFT], 1, "prompt.demonstration=true", "rules.extended=true", "rules.general=true")
    assert "1. x=0 & y=0 & g1=false & g2=false & obs_idx=0 | wall: up, left; toward: right; closer: right -> right\n" \
        in llm.prompts[0]
    _, llm = solve(tiny_grid, [LEFT], 1)
    assert "## An optimal run" not in llm.prompts[0]


def test_deviations_list_where_the_rules_leave_the_optimal_strategy(tiny_grid):
    """With `feedback.deviations`, the prompt lists the states the rules decide where their action loses most against
    the optimal strategy, weighted by how often the rules reach them; `feedback.stuck` says which actions change
    nothing (none here: the obstacle moves every step)."""
    result, _ = solve(tiny_grid, [LEFT, LEFT], 2, "feedback.optimal=true", "feedback.deviations=true",
                      "feedback.stuck=true")
    prompt = result["iterations"][1]["prompt"]
    section = prompt[prompt.index("## Where your rules leave the optimal strategy"):prompt.index("## Task")]
    assert "%: x=1 & y=0 & g1=false & g2=false & obs_idx=1: optimal: up, not left\n" in section   # stuck at the wall
    assert "changes nothing" not in prompt
    shares = [int(m) for m in re.findall(r"^- (\d+)%", section, re.M)]
    assert shares == sorted(shares, reverse=True) and 95 <= sum(shares) <= 105
