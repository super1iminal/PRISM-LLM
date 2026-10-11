"""The planner runs exactly the PRISM commands it ran before PRISM calls became yielded tasks
(tests/data/prism_calls.json, recorded on the earlier code; see tests/prism_calls.py): in order, one
instance at a time, and as the same commands when the schedulers run several instances at once."""
import json
import logging
import shutil

import pytest

from core.domain import load_domain
from core.planner import SymbolicPlanner
from core.scheduler import EventScheduler, LockstepScheduler
from prism_calls import BASELINE, collect, recording
from test_scheduler import REPLAY_CASES, replay_backend, saved_run

LOG = logging.getLogger("test_prism_calls")

needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")


@needs_prism
def test_each_instance_runs_the_recorded_prism_commands_in_order():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    now = collect()
    assert list(now) == list(baseline)
    for case in baseline:
        assert now[case] == baseline[case], case


@needs_prism
@pytest.mark.parametrize("kind", [LockstepScheduler, EventScheduler])
def test_the_schedulers_run_the_recorded_prism_commands(kind):
    """With every instance of a replay case in flight at once, the commands interleave, so compare them as a
    multiset: the same commands, each as often as before."""
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    for run, instances in REPLAY_CASES:
        cfg, samples = saved_run(run)
        domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
        by_id = {str(i.id): i for i in domain.load_instances(cfg.domain.dataset)}
        backend = replay_backend({k: samples[k] for k in instances}, delay=0.05)
        with recording() as calls:
            kind(SymbolicPlanner(domain, None, cfg).solve_steps, backend, len(instances), lambda i: LOG).run(
                [by_id[k] for k in instances])
        assert sorted(calls) == sorted(c for k in instances for c in baseline[f"{run}/{k}"]), run
