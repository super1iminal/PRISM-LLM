"""LLM tasks, backends and schedulers: the planner as a generator, driven one task at a time or in lockstep batches."""
import io
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path

import pytest

from config import load_config
from core.domain import load_domain
from core.planner import SymbolicPlanner
from core.scheduler import LockstepScheduler, drive, failed_result
from core.tasks import LLMError, LLMResult, TaskFactory
from fakes import FakeBackend

needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")
ABLATIONS = Path(__file__).resolve().parent.parent / "out" / "results" / "ablations"
LOG = logging.getLogger("test_scheduler")


# ---------------------------------------------------------------- helpers

def toy_steps(instance, logger):
    """A stand-in planner: asks `instance.rounds` times, fails if an answer is "boom", returns the answers."""
    answers = []
    for r in range(1, instance.rounds + 1):
        result = yield TaskFactory(load_config().llm, instance.id).make(f"{instance.id} round {r}", None, round=r)
        if result.error is not None:
            raise LLMError(result.error)
        if result.text == "boom":
            raise ValueError("planner crashed")
        answers.append(result.text)
    return {"success": True, "answers": answers, "iterations": []}


class Toy:
    def __init__(self, id, rounds):
        self.id, self.rounds = id, rounds


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def saved_run(name):
    """A saved `<condition>/seed_<n>` run: its condition's config with its seed, and its samples.

    The run's config.json may predate today's schema, so the config comes from the condition file; every
    setting the run recorded that still exists must match. The exact check is off: the runs reported the
    loop's values.
    """
    run = ABLATIONS / name
    if not (run / "config.json").exists():
        pytest.skip(f"saved run {name} not available")
    condition, seed = name.split("/seed_")
    cfg = load_config(condition, [f"llm.seed={seed}", "prism.exact_check=false"])
    saved, now = _flat(json.loads((run / "config.json").read_text())), _flat(cfg.to_dict())
    assert {k: v for k, v in saved.items() if k in now} == {k: now[k] for k in saved if k in now}
    samples = {}
    for path in sorted((run / "outputs").glob("sample_*.json")):
        sample = json.loads(path.read_text())
        samples[str(sample["instance"])] = sample
    return cfg, samples


def replay_backend(samples):
    """Answers each instance's tasks with that instance's saved raw outputs, in order."""
    queues = {inst: iter([o for it in s["iterations"] for o in it["raw_outputs"]]) for inst, s in samples.items()}
    return FakeBackend(lambda task: next(queues[str(task.meta["instance"])]))


def assert_same_rounds(replayed, saved):
    assert len(replayed["iterations"]) == len(saved["iterations"])
    for a, b in zip(replayed["iterations"], saved["iterations"]):
        for key in ("mode", "prompt", "rules", "num_rules", "invalid_answers", "improved", "llm_calls",
                    "raw_outputs", "reachable_situations", "uncovered_situations"):
            assert a[key] == b[key], (a["iteration"], key)
        for key in ("best", "worst"):
            assert a[key] == pytest.approx(b[key], abs=1e-9), (a["iteration"], key)
    assert replayed["success"] == saved["success"]
    assert replayed["final_rules"] == saved["final_rules"]


# ---------------------------------------------------------------- 1. equivalence with saved runs

# Saved gridworld runs covering every loop path: initial / refine / extend / table
# rounds, retries, and invalid answers with re-asks (fixups).
REPLAY_CASES = [
    ("D7/seed_1", ["1", "14", "15"]),     # 7 rounds; extend; 2 and 6 invalid answers
    ("B2/seed_2", ["3", "15"]),           # extend then refine; extends with 3 invalid answers
    ("R1/seed_2", ["6"]),                 # four extends in a row
    ("S1/seed_1", ["0"]),                 # table feedback (ablation S1)
]


@needs_prism
@pytest.mark.parametrize("run,instances", REPLAY_CASES)
def test_replay_one_at_a_time(run, instances):
    """planner.solve (the threads path) reproduces the saved rounds exactly from the saved LLM outputs."""
    cfg, samples = saved_run(run)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    by_id = {str(i.id): i for i in domain.load_instances(cfg.domain.dataset)}
    planner = SymbolicPlanner(domain, replay_backend({k: samples[k] for k in instances}), cfg)
    for inst in instances:
        assert_same_rounds(planner.solve(by_id[inst], LOG), samples[inst])


@needs_prism
@pytest.mark.parametrize("run,instances", REPLAY_CASES[:2])
def test_replay_lockstep(run, instances):
    """The lockstep scheduler gives the same rounds as the saved run, with batches of at most `slots`."""
    cfg, samples = saved_run(run)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    by_id = {str(i.id): i for i in domain.load_instances(cfg.domain.dataset)}
    backend = replay_backend({k: samples[k] for k in instances})
    planner = SymbolicPlanner(domain, None, cfg)
    log = io.StringIO()
    results = LockstepScheduler(planner.solve_steps, backend, 2, lambda i: LOG, task_log=log).run(
        [by_id[k] for k in instances])
    for inst in instances:
        assert_same_rounds(results[by_id[inst].id], samples[inst])
    assert max(len(b) for b in backend.batches) <= 2
    assert sum(len(b) for b in backend.batches) == sum(
        it["llm_calls"] for k in instances for it in samples[k]["iterations"])
    lines = [json.loads(line) for line in log.getvalue().splitlines()]
    assert len(lines) == sum(len(b) for b in backend.batches)
    assert all(line["result"]["text"] is not None for line in lines)


# ---------------------------------------------------------------- 2. scheduler behaviour

def test_lockstep_batches_and_refill():
    """5 instances, 3 slots: no batch exceeds 3, finished instances free their slot, all finish."""
    instances = [Toy(f"g{i}", rounds) for i, rounds in enumerate([1, 3, 2, 2, 1])]
    backend = FakeBackend(lambda task: f"answer to {task.prompt}")
    finished = []
    results = LockstepScheduler(toy_steps, backend, 3, lambda i: LOG,
                                on_finish=lambda inst, r: finished.append(inst.id)).run(instances)
    batches = [[t.meta["instance"] for t in b] for b in backend.batches]
    assert batches == [["g0", "g1", "g2"], ["g1", "g2", "g3"], ["g1", "g3", "g4"]]
    assert sorted(finished) == sorted(results) == [f"g{i}" for i in range(5)]
    assert results["g1"]["answers"] == [f"answer to g1 round {r}" for r in (1, 2, 3)]
    assert all("total_time" in r and r["instance"] == k for k, r in results.items())


def test_lockstep_failures_stay_local():
    """A crashing instance and a failed backend call end only their own instance, with `<Type>: <message>` errors."""
    def answer(task):
        if task.meta["instance"] == "bad_llm":
            raise ConnectionError("server gone")
        return "boom" if task.meta["instance"] == "crash" else "ok"

    instances = [Toy("crash", 2), Toy("bad_llm", 2), Toy("fine", 3)]
    results = LockstepScheduler(toy_steps, FakeBackend(answer), 3, lambda i: LOG).run(instances)
    assert results["crash"]["success"] is False and results["crash"]["error"] == "ValueError: planner crashed"
    assert results["bad_llm"]["success"] is False and results["bad_llm"]["error"] == "ConnectionError: server gone"
    assert results["fine"]["answers"] == ["ok"] * 3


def test_lockstep_respects_backend_max_batch():
    instances = [Toy(f"g{i}", 2) for i in range(5)]
    backend = FakeBackend(lambda task: "ok", max_batch=2)    # asserts on oversize batches
    results = LockstepScheduler(toy_steps, backend, 5, lambda i: LOG).run(instances)
    assert len(results) == 5 and [len(b) for b in backend.batches] == [2, 2, 1, 2, 2, 1]


def test_drive_and_error_format():
    assert drive(toy_steps(Toy("g", 2), LOG), lambda t: LLMResult(t.id, "x"))["answers"] == ["x", "x"]
    with pytest.raises(LLMError):
        drive(toy_steps(Toy("g", 2), LOG), lambda t: LLMResult(t.id, None, error="ResponseError: oom"))
    # backend errors keep the backend's own message; other exceptions are prefixed with their type
    assert failed_result(LLMError("ResponseError: oom"))["error"] == "ResponseError: oom"
    assert failed_result(KeyError("x"))["error"] == "KeyError: 'x'"


@needs_prism
def test_lockstep_matches_one_at_a_time():
    """Same instances, same scripted answers: lockstep and one-at-a-time give identical results."""
    cfg = load_config(overrides=["planner.max_rounds=2", "llm.seed=3"])
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    instances = domain.load_instances(cfg.domain.dataset)[:4]
    rules = {"rules": [{"condition": "!g1", "action": "down"}, {"condition": "true", "action": "right"}]}

    def answer(task):
        if task.meta["instance"] == instances[1].id and task.meta["fixup"] == 0:
            return '{"rules": [{"condition": "nonsense_var > 1", "action": "up"}]}'   # forces a re-ask
        return json.dumps(rules)

    serial = SymbolicPlanner(domain, FakeBackend(answer), cfg)
    expected = {i.id: serial.solve(i, LOG) for i in instances}
    backend = FakeBackend(answer)
    got = LockstepScheduler(SymbolicPlanner(domain, None, cfg).solve_steps, backend, 3, lambda i: LOG).run(instances)
    for inst in instances:
        a, b = got[inst.id], expected[inst.id]
        assert [(it["mode"], it["prompt"], it["rules"], it["best"], it["worst"], it["invalid_answers"])
                for it in a["iterations"]] == \
               [(it["mode"], it["prompt"], it["rules"], it["best"], it["worst"], it["invalid_answers"])
                for it in b["iterations"]]
    assert expected[instances[1].id]["iterations"][0]["invalid_answers"]       # the re-ask happened
    assert max(len(b) for b in backend.batches) <= 3


# ---------------------------------------------------------------- 3. seeds

def test_task_seeds_are_numbered_per_call():
    cfg = load_config(overrides=["llm.seed=7", "llm.temperature=0.2"]).llm
    tasks = TaskFactory(cfg, "g0")
    made = [tasks.make(f"p{n}", {"type": "object"}, round=1, fixup=n) for n in range(4)]
    assert [t.params["seed"] for t in made] == [7 * 1_000_003 + n for n in range(4)]
    assert made[0].params == {"think": False, "num_ctx": 16384, "num_predict": 8192, "temperature": 0.2,
                              "seed": 7 * 1_000_003}
    assert made[0].messages == [{"role": "user", "content": "p0"}] and made[0].model == cfg.model
    assert "seed" not in TaskFactory(load_config().llm, "g0").make("p", None).params   # seed: null


@needs_prism
def test_planner_seeds_count_calls_per_instance():
    """Each instance numbers its calls from 0, across rounds and re-asks."""
    cfg = load_config(overrides=["planner.max_rounds=2", "llm.seed=5"])
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    instances = domain.load_instances(cfg.domain.dataset)[:2]
    calls = defaultdict(int)

    def answer(task):
        calls[task.meta["instance"]] += 1
        return '{"rules": [{"condition": "nope", "action": "up"}]}' if task.meta["fixup"] < 2 else '{"rules": []}'

    backend = FakeBackend(answer)
    LockstepScheduler(SymbolicPlanner(domain, None, cfg).solve_steps, backend, 2, lambda i: LOG).run(instances)
    for inst in instances:
        seeds = [t.params["seed"] for b in backend.batches for t in b if t.meta["instance"] == inst.id]
        assert seeds == [5 * 1_000_003 + n for n in range(calls[inst.id])] and len(seeds) == 6


# ---------------------------------------------------------------- run_symbolic wiring

@needs_prism
def test_run_symbolic_both_schedulers(tmp_path, monkeypatch):
    """run() with threads and with lockstep writes the same per-round results; lockstep also logs its tasks."""
    import pandas as pd
    import run_symbolic

    rules = json.dumps({"rules": [{"condition": "!g1", "action": "down"}, {"condition": "true", "action": "right"}]})
    monkeypatch.setattr(run_symbolic, "make_backend", lambda cfg: FakeBackend(lambda task: rules))
    frames = {}
    for scheduler in ("threads", "lockstep"):
        cfg = load_config(overrides=["run.limit=3", "planner.max_rounds=2", f"run.scheduler={scheduler}"])
        out = tmp_path / scheduler
        run_symbolic.run(cfg, str(out))
        frames[scheduler] = pd.read_parquet(out / "SYMBOLIC_results.parquet")
        assert len(list((out / "outputs").glob("sample_*.json"))) == 3
        assert (out / "llm_tasks.jsonl").exists() == (scheduler == "lockstep")
    timing = [c for c in frames["threads"].columns if c.endswith("_time")]
    pd.testing.assert_frame_equal(frames["threads"].drop(columns=timing), frames["lockstep"].drop(columns=timing))
    tasks = [json.loads(line) for line in (tmp_path / "lockstep" / "llm_tasks.jsonl").read_text().splitlines()]
    assert len(tasks) == frames["lockstep"]["iter_llm_calls"].sum()
    assert {t["task"]["meta"]["instance"] for t in tasks} == set(frames["lockstep"]["instance"])
