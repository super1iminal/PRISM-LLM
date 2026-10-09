"""Training sets (core/training.py): one rule list written for, verified on and solved over several instances."""
import json
import logging
import shutil

import pandas as pd
import pytest

import run_symbolic
from config import load_config
from core.domain import Instance, load_domain
from core.planner import SymbolicPlanner
from core.training import training_sets
from fakes import TINY_GRID, ScriptedBackend, answer
from results_io import SYMBOLIC_RESULTS

needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")

# Row 0 is the tiny grid; row 1 puts a static obstacle right of the start, where "true -> right" fails;
# row 2 has a third goal, so it has another goal flag.
ROWS = [TINY_GRID.splitlines()[1],
        '4,"{1: (0, 3), 2: (3, 3)}","[(0, 1)]","[(2, 1), (2, 2)]",6',
        '4,"{1: (0, 3), 2: (3, 3), 3: (3, 0)}","[(1, 1)]","[(2, 1), (2, 2)]",9']
RIGHT = answer(("true", "right"))                                    # passes row 0 only
BOTH = answer(("!g1 & x = 0", "right"), ("!g1", "up"), ("g1 & !g2 & y < 3", "right"), ("g1 & !g2", "down"))


@pytest.fixture
def grids(tmp_path):
    path = tmp_path / "grids.csv"
    path.write_text(TINY_GRID.splitlines()[0] + "\n" + "\n".join(ROWS) + "\n", encoding="utf-8")
    return path


def solve(grids, ids, answers, rounds):
    domain = load_domain("gridworld", ["obs_idx"])
    cfg = load_config(overrides=[f"planner.max_rounds={rounds}", "llm.seed=1"])
    llm = ScriptedBackend(answers)
    target, = training_sets(domain.load_instances(str(grids)), [ids])
    return SymbolicPlanner(domain, llm, cfg).solve(target, logging.getLogger("test_training")), llm


def test_training_sets_group_instances_by_id():
    instances = [Instance(str(i)) for i in range(4)]
    sets = training_sets(instances, [[0, 2], [3]])
    assert [s.id for s in sets] == ["set0", "set1"]
    assert [[m.id for m in s.members] for s in sets] == [["0", "2"], ["3"]]
    with pytest.raises(ValueError, match="no instances with ids \\[7\\]"):
        training_sets(instances, [[0, 7]])


@needs_prism
def test_feedback_is_about_the_member_that_fails_worst(grids):
    result, llm = solve(grids, [0, 1], [RIGHT, BOTH], 2)
    first, second = result["iterations"]
    assert "### Instance 1" in llm.prompts[0] and "### Instance 2" in llm.prompts[0]
    assert "Write **one** rule list that works, unchanged, on each of the 2 instances" in llm.prompts[0]
    assert [m["instance"] for m in first["members"]] == ["0", "1"]
    assert not first["worst_case_success"] and second["mode"] == "refine"
    assert "| Instance 1 | 6 of 6 | - |" in second["prompt"] and "| Instance 2 | 4 of 6 |" in second["prompt"]
    assert "The rest of this prompt is about instance 2" in second["prompt"]
    assert result["success"] and [m["success"] for m in result["members"]] == [True, True]


@needs_prism
def test_a_set_is_solved_only_when_every_member_passes(grids):
    result, _ = solve(grids, [0, 1], [RIGHT], 1)
    assert not result["success"] and [m["success"] for m in result["members"]] == [True, False]
    worst = {m["instance"]: m["final_worst"]["goal2"] for m in result["members"]}
    assert result["final_worst"]["goal2"] == min(worst.values())       # each requirement's worst member


@needs_prism
def test_members_must_share_variables(grids):
    with pytest.raises(ValueError, match="instance 2 has other variables"):
        solve(grids, [0, 2], [RIGHT], 1)


@needs_prism
def test_run_with_train_sets_writes_one_sample_per_set(tmp_path, grids, monkeypatch):
    monkeypatch.setattr(run_symbolic, "make_backend", lambda config: ScriptedBackend([BOTH]))
    cfg = load_config(overrides=[f"domain.dataset={grids.as_posix()}", "run.workers=1", "planner.max_rounds=1",
                                 "run.train_sets=[[0, 1]]"])
    run_dir = tmp_path / "run"
    run_symbolic.run(cfg, str(run_dir))
    df = pd.read_parquet(run_dir / SYMBOLIC_RESULTS).reset_index()
    assert len(df) == 1 and df.instance.tolist() == ["set0"] and df.success.tolist() == [True]
    record = json.loads((run_dir / "outputs" / "sample_000.json").read_text(encoding="utf-8"))
    assert [m["instance"] for m in record["members"]] == ["0", "1"]
    assert "on instance 2" in (run_dir / "worker_set0.log").read_text(encoding="utf-8")
    shutil.rmtree(run_dir)
