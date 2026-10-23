"""UUV's failure feedback in the domain's words (`feedback.words`; UUV.state_words, UUV.loss_words; Asher's UUV plan,
step 5): with words on, a training member's failure report names states and risks in words, and no failure state
falls back to `format_state`; with words off, port_uuv's prompts are exactly what they were without the hooks."""
import logging
import shutil

import pytest

from config import load_config
from core.domain import load_domain
from core.planner import SymbolicPlanner
from core.training import training_sets
from fakes import ScriptedBackend, answer

needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")
HIGH = answer(("true", "high"))       # fails on member 0 (w7_i6): done_in_time and energy_budget


def prompts(*overrides, domain=None):
    domain = domain or load_domain("uuv")
    cfg = load_config("port_uuv", ["planner.chains=1", "planner.max_rounds=2", "planner.repair=0", *overrides])
    llm = ScriptedBackend([HIGH, HIGH])
    target, = training_sets(domain.load_instances("uuv_family_v0.csv"), [[0]])
    SymbolicPlanner(domain, llm, cfg).solve(target, logging.getLogger("test_uuv_words"))
    return llm.prompts


@needs_prism
def test_failure_feedback_in_the_domains_words():
    first, second = prompts("feedback.words=true")
    failures = [l for l in second.splitlines() if " of runs " in l]
    assert failures, "the revise prompt reports failure states"
    for line in failures:
        assert ", active altitude " in line and " visibility " in line and "allowed)" in line
        assert "s=" not in line and "water_visib=" not in line            # never format_state's form
    assert "missing the deadline" in second or "a thruster failure" in second
    assert "loses done_in_time" not in second and "loses no_thruster_failure" not in second


@needs_prism
def test_prompts_unchanged_with_words_off(monkeypatch):
    on = load_domain("uuv")
    off = load_domain("uuv")
    monkeypatch.setattr(type(off), "state_words", lambda self, instance, valuation: None)
    monkeypatch.setattr(type(off), "loss_words", lambda self, requirement: None)
    plain = prompts(domain=off)                       # words off and no hooks: the prompts before step 5
    monkeypatch.undo()
    assert prompts(domain=on) == plain                # words off, hooks present: identical
    assert prompts("feedback.words=true", domain=on)[0] == plain[0]   # the first prompt has no feedback either way
