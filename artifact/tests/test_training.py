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
from fakes import TINY_GRID, FakeBackend, ScriptedBackend, answer
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


@needs_prism
def test_every_scheduler_gives_the_same_training_results(tmp_path, grids, monkeypatch):
    """Two training sets in flight at once: lockstep and the event scheduler (PRISM as background processes,
    answers arriving out of order) write the same rounds as worker threads, and log every task."""
    by_round = {1: RIGHT, 2: BOTH}   # round 1 fails on row 1, round 2 solves both rows
    monkeypatch.setattr(run_symbolic, "make_backend", lambda config: FakeBackend(
        lambda task: by_round.get(task.meta.get("round"), answer()), delay=0.05))
    frames, samples = {}, {}
    for scheduler in ("threads", "lockstep", "event"):
        cfg = load_config(overrides=[f"domain.dataset={grids.as_posix()}", "run.workers=2", "planner.max_rounds=2",
                                     "llm.seed=1", "run.train_sets=[[0, 1], [1, 0]]", f"run.scheduler={scheduler}"])
        run_dir = tmp_path / scheduler
        run_symbolic.run(cfg, str(run_dir))
        frames[scheduler] = pd.read_parquet(run_dir / SYMBOLIC_RESULTS)
        samples[scheduler] = [json.loads((run_dir / "outputs" / f"sample_00{k}.json").read_text(encoding="utf-8"))
                              for k in range(2)]
        if scheduler != "threads":
            tasks = (run_dir / "llm_tasks.jsonl").read_text(encoding="utf-8").splitlines()
            assert len(tasks) == frames[scheduler]["iter_llm_calls"].sum()
    assert frames["threads"].reset_index().instance.unique().tolist() == ["set0", "set1"]
    assert frames["threads"].reset_index().success.unique().tolist() == [True]
    assert [len(sample["iterations"]) for sample in samples["threads"]] == [2, 2]   # a refine round on each set
    timing = [c for c in frames["threads"].columns if c.endswith("_time")]
    for scheduler in ("lockstep", "event"):
        pd.testing.assert_frame_equal(frames["threads"].drop(columns=timing), frames[scheduler].drop(columns=timing))
        for got, want in zip(samples[scheduler], samples["threads"]):
            assert [it["prompt"] for it in got["iterations"]] == [it["prompt"] for it in want["iterations"]]
            assert [it["members"] for it in got["iterations"]] == [it["members"] for it in want["iterations"]]
            assert got["members"] == want["members"]


def test_hidden_members_must_be_given_per_training_set():
    instances = [Instance(str(i)) for i in range(3)]
    sets = training_sets(instances, [[0]], [[1, 2]])
    assert [m.id for m in sets[0].hidden] == ["1", "2"]
    with pytest.raises(ValueError, match="hidden_sets has 2 groups"):
        training_sets(instances, [[0]], [[1], [2]])


@needs_prism
def test_hidden_members_are_scored_but_never_shown(grids):
    domain = load_domain("gridworld", ["obs_idx"])
    cfg = load_config(overrides=["planner.max_rounds=2", "llm.seed=1"])
    llm = ScriptedBackend([RIGHT, BOTH])
    target, = training_sets(domain.load_instances(str(grids)), [[0]], [[1]])
    result = SymbolicPlanner(domain, llm, cfg).solve(target, logging.getLogger("test_training"))
    first, second = result["iterations"]
    assert not first["worst_case_success"]                  # RIGHT passes the shown grid, not the hidden one
    assert "hidden" not in llm.prompts[0].lower() and "Instance 2" not in llm.prompts[0]
    assert "On 1 further instances of this problem that you are not shown" in second["prompt"]
    assert "the rules meet every requirement in the worst case on 0." in second["prompt"]
    assert result["success"] and result["members"][1]["hidden"] and "hidden" not in result["members"][0]


@needs_prism
def test_a_failing_pool_instance_joins_the_training_set(grids):
    domain = load_domain("gridworld", ["obs_idx"])
    cfg = load_config(overrides=["planner.max_rounds=2", "llm.seed=1"])
    llm = ScriptedBackend([RIGHT, BOTH])
    target, = training_sets(domain.load_instances(str(grids)), [[0]], pool=[[1]])
    result = SymbolicPlanner(domain, llm, cfg).solve(target, logging.getLogger("test_training"))
    first, second = result["iterations"]
    assert first["worst_case_success"]                       # RIGHT solves grid 0 ...
    assert result["counterexamples"] == ["1"]                # ... fails on grid 1, which joins the set
    assert "### Instance 2" in second["prompt"] and [m["instance"] for m in result["members"]] == ["0", "1"]
    assert result["success"]                                 # BOTH passes both


def test_chains_need_training_sets_without_a_pool():
    load_config(overrides=["planner.chains=2", "run.train_sets=[[0, 1]]"])
    for extra in ([], ["run.train_sets=[[0]]", "run.pool_sets=[[1]]"]):
        with pytest.raises(ValueError, match="planner.chains > 1"):
            load_config(overrides=["planner.chains=2"] + extra)


@needs_prism
def test_chains_keep_the_chain_with_the_largest_margin(tmp_path, grids, monkeypatch):
    """Two independent loops on one set: the one whose rules pass both members wins over the one that fails a
    member, wherever it comes in the chain order; every chain's rounds are kept, and the chains' seeds differ."""
    backend = FakeBackend(lambda task: BOTH if task.meta["instance"].endswith("chain1") else RIGHT)
    monkeypatch.setattr(run_symbolic, "make_backend", lambda config: backend)
    cfg = load_config(overrides=[f"domain.dataset={grids.as_posix()}", "run.workers=2", "planner.max_rounds=1",
                                 "llm.seed=1", "run.train_sets=[[0, 1]]", "planner.chains=2"])
    run_dir = tmp_path / "run"
    run_symbolic.run(cfg, str(run_dir))
    record = json.loads((run_dir / "outputs" / "sample_000.json").read_text(encoding="utf-8"))
    assert record["instance"] == "set0" and record["success"] and record["chain"] == 1
    margins = [c["margin"] for c in record["chains"]]
    assert margins[0] < 0 <= margins[1] and [c["success"] for c in record["chains"]] == [False, True]
    assert [(it["iteration"], it["chain"], it["chain_iteration"]) for it in record["iterations"]] == [(1, 0, 1), (2, 1, 1)]
    assert record["final_rules"] == record["iterations"][1]["rules"]
    df = pd.read_parquet(run_dir / SYMBOLIC_RESULTS).reset_index()
    assert df.instance.tolist() == ["set0", "set0"] and df.success.tolist() == [True, True]
    seeds = {task.meta["instance"]: task.params["seed"] for batch in backend.batches for task in batch}
    assert seeds["set0.chain1"] - seeds["set0.chain0"] == 100_000
    shutil.rmtree(run_dir)


@needs_prism
def test_keep_ties_newer_keeps_the_newer_of_two_equal_lists(grids):
    """Two lists with the same values on the members (neither passes row 1): keep-best keeps the first by default,
    the second with planner.keep_ties=newer, so the next feedback is about the newer list."""
    domain = load_domain("gridworld", ["obs_idx"])
    same = answer(("x >= 0", "right"))                                   # the same policy as RIGHT
    for ties, want in (("older", "true"), ("newer", "x >= 0")):
        cfg = load_config(overrides=["planner.max_rounds=2", "llm.seed=1", f"planner.keep_ties={ties}"])
        target, = training_sets(domain.load_instances(str(grids)), [[0, 1]])
        result = SymbolicPlanner(domain, ScriptedBackend([RIGHT, same]), cfg).solve(
            target, logging.getLogger("test_training"))
        assert [r["condition"] for r in result["final_rules"]] == [want]
        assert result["iterations"][1]["improved"] == (ties == "newer")
