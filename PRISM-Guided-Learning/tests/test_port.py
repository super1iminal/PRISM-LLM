"""The fresh-transfer method's port into the loop: strict vocabulary, the worst case spelled out, feedback on every
member with failure locations and the attempt history, and the final pick among every round's lists with repair."""
import json
import logging
import shutil

import pytest

import run_symbolic
from config import load_config
from core.domain import Requirement, load_domain
from core.pick import choose, repair
from core.planner import SymbolicPlanner
from core.training import training_sets
from fakes import TINY_GRID, FakeBackend, ScriptedBackend, answer

needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")

# Row 0 is the tiny grid; row 1 puts a static obstacle right of the start, where "true -> right" stalls.
ROWS = [TINY_GRID.splitlines()[1], '4,"{1: (0, 3), 2: (3, 3)}","[(0, 1)]","[(2, 1), (2, 2)]",6']
RIGHT = answer(("true", "right"))
CLOSER = answer(("closer & !early & !obstacle", "any"), ("true", "right"))    # passes both rows
PORT = ["rules.extended=true", "rules.general=true", "rules.strict=true", "rules.any_order=none",
        "prompt.worst_case=true", "planner.feedback=failures", "planner.retry=never", "llm.seed=1"]


@pytest.fixture
def grids(tmp_path):
    path = tmp_path / "grids.csv"
    path.write_text(TINY_GRID.splitlines()[0] + "\n" + "\n".join(ROWS) + "\n", encoding="utf-8")
    return path


def solve(grids, answers, rounds, *overrides):
    domain = load_domain("gridworld", ["obs_idx"])
    cfg = load_config(overrides=[*PORT, f"planner.max_rounds={rounds}", *overrides])
    llm = ScriptedBackend(answers)
    target, = training_sets(domain.load_instances(str(grids)), [[0, 1]])
    return SymbolicPlanner(domain, llm, cfg).solve(target, logging.getLogger("test_port")), llm


@needs_prism
def test_strict_rejects_coordinates_and_the_prompt_leaves_them_out(grids):
    result, llm = solve(grids, [answer(("x = 0 & y < N", "right")), RIGHT], 1)
    first = llm.prompts[0]
    assert "- `x` (" not in first and "- `N`" not in first and "- `g1` (boolean)" in first
    assert "The agent's coordinates (`x`, `y`)" in first and "**The worst case decides.**" in first
    assert "the obstacle's phase changes every step" in first
    errors = result["iterations"][0]["invalid_answers"]
    assert errors == ["rule 1 (x = 0 & y < N -> right): uses N, x, y, which conditions may not name"]
    plain, _ = solve(grids, [RIGHT], 1, "rules.strict=false", "prompt.worst_case=false")
    assert "- `x` (" in plain["iterations"][0]["prompt"] and "worst case decides" not in plain["iterations"][0]["prompt"]


@needs_prism
def test_failures_feedback_reports_every_member_with_locations_and_history(grids):
    """`true -> right` passes the tiny grid and enters goal 2 early on row 1: the revise prompt reports both
    members, where the worst-case run of each failing requirement is lost with the rule and what each action risks
    there, and from the third round on the earlier attempts."""
    result, llm = solve(grids, [RIGHT, RIGHT, RIGHT], 3)
    assert [it["mode"] for it in result["iterations"]] == ["initial", "revise", "revise"]
    second, third = llm.prompts[1], llm.prompts[2]
    assert "## Its exact worst-case results on each instance" in second
    assert "### Instance 1: every requirement holds (smallest margin +0.080)." in second
    assert "### Instance 2: FAILS seq_1_before_2, complete_sequence\n- `seq_1_before_2` = 0.559 (needs >= 0.8)." in second
    assert ("of runs fail from here: x=2 & y=3 & g1=false & g2=false & obs_idx=1; decided by rule 1 `true -> right`; "
            "allowed right; the worst case takes right.") in second
    assert "down: [hazard, risky, early] risk: loses seq_1_before_2, complete_sequence 0.70" in second
    assert "* the same state as above: x=2 & y=3" in second and "## Your attempts so far" not in second
    assert ("## Your attempts so far\n\n- attempt 1: every requirement holds on 1 of 2 instances, total shortfall "
            "0.483\n- attempt 2: every requirement holds on 1 of 2 instances, total shortfall 0.483") in third


@needs_prism
def test_pick_candidates_keeps_the_list_certified_on_most_members(tmp_path, grids, monkeypatch):
    """Two chains, every round of both competes: the list that passes both members wins, and the result records the
    candidates and the exact values of the chosen list."""
    backend = FakeBackend(lambda task: CLOSER if task.meta["instance"].endswith("chain1") else RIGHT)
    monkeypatch.setattr(run_symbolic, "make_backend", lambda config: backend)
    cfg = load_config(overrides=[*PORT, f"domain.dataset={grids.as_posix()}", "run.workers=2", "planner.max_rounds=1",
                                 "run.train_sets=[[0, 1]]", "planner.chains=2", "planner.pick=candidates",
                                 "planner.repair=0"])
    run_dir = tmp_path / "run"
    run_symbolic.run(cfg, str(run_dir))
    record = json.loads((run_dir / "outputs" / "sample_000.json").read_text(encoding="utf-8"))
    pick = record["pick"]
    assert len(pick["candidates"]) == 2 and pick["candidates"][pick["chosen"]]["from"] == [1, 1]
    assert record["success"] and pick["candidates"][pick["chosen"]]["final_score"][0] == 2
    assert record["final_rules"] == [{"condition": "closer & !early & !obstacle", "action": "any"},
                                     {"condition": "true", "action": "right"}]
    assert all(m["success"] for m in record["members"])
    shutil.rmtree(run_dir)


def test_repair_climbs_to_a_passing_list_and_choose_prefers_more_members():
    """Repair keeps the first improving neighbour: a list whose value depends on its action reaches the action that
    passes, within its steps."""
    req = Requirement("ok", 'F "goal"', 0.5)
    good = {"up": 1.0, "right": 0.6, "down": 0.2, "left": 0.0}

    def evaluate(rules):
        return [{"ok": good.get(rules[-1][1], 0.0)}]
    fixed = repair([("true", "left")], evaluate, [[req]], ["wall", "!wall"], ["up", "right", "down", "left"], steps=20,
                   tries=60, seed=0)
    assert evaluate(fixed)[0]["ok"] >= 0.55
    a = {"rules": [("a", "up")], "final_score": (1, 0.0, 0.1), "resolved_score": (1, 0.0, 0.1)}
    b = {"rules": [("b", "up")], "final_score": (2, 0.0, 0.01), "resolved_score": (2, 0.0, 0.01)}
    assert choose([a, b]) is b
