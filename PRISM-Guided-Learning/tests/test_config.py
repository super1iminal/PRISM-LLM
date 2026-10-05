"""Run config, retry policies, obstacle visibility and the joint best-case branch."""
import json
import shutil
from dataclasses import MISSING, fields, is_dataclass

import pytest
import yaml

import config
from config import Config, conditions, load_config
from core.domain import load_domain
from core.retry import RetryPolicy
from core.rules import SymbolicPolicy
from core.verifier import PolicyVerifier

needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")


def test_every_condition_loads_and_round_trips():
    names = conditions()   # one file per condition in configs/run/conditions/
    assert {"B1", "B2", "R1", "R2", "R3", "R4", "R5", "S1", "S4", "S5", "V1", "V2", "L1", "L2", "D7", "U1",
            "pre_phase_a"} <= set(names)
    for name in names:
        cfg = load_config(name, ["llm.seed=3"])
        assert cfg.condition == name and cfg.llm.seed == 3
        json.dumps(cfg.to_dict())   # what run dirs record
    with pytest.raises(ValueError, match="unknown condition"):
        load_config("no_such_condition")


@pytest.mark.parametrize("override", ["planner.retyr=never", "nosuch.key=1", "planner.retry=bogus",
                                      "planner.branch=sideways", "approach=rl", "feedback.horizon=0",
                                      "feedback.horizon=deadline", "feedback.horizon=true",
                                      "feedback.horizon_by_domain={uuv: domain}"])
def test_bad_config_is_rejected(override):
    with pytest.raises(ValueError):
        load_config(overrides=[override])


def test_defaults_live_only_in_default_yaml():
    """No setting has a second default in Python (only `condition`, which is a record, not a setting)."""
    def defaults(cls, path=""):
        for f in fields(cls):
            if is_dataclass(f.type):
                yield from defaults(f.type, f"{path}{f.name}.")
            elif f.default is not MISSING or f.default_factory is not MISSING:
                yield path + f.name
    assert list(defaults(Config)) == ["condition"]


def test_key_missing_from_default_yaml_is_an_error(tmp_path, monkeypatch):
    data = yaml.safe_load((config.RUN_CONFIG_DIR / "default.yaml").read_text(encoding="utf-8"))
    del data["planner"]["max_fixups"]
    (tmp_path / "default.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr(config, "RUN_CONFIG_DIR", tmp_path)
    with pytest.raises(ValueError, match=r"missing config key\(s\) \['planner.max_fixups'\]"):
        load_config()


@pytest.mark.parametrize("spec,next_round,stall,gain,expected", [
    ("stall:2", 3, 1, 0.0, False), ("stall:2", 3, 2, 0.0, True),
    ("never", 5, 9, 0.0, False), ("always", 2, 0, 1.0, True),
    ("every:3", 3, 0, 0.0, False), ("every:3", 4, 0, 0.0, True), ("every:3", 7, 0, 0.0, True),
    ("gain:0.05", 3, 0, 0.04, True), ("gain:0.05", 3, 0, 0.2, False), ("gain:0.05", 2, 0, float("inf"), False),
])
def test_retry_policy(spec, next_round, stall, gain, expected):
    assert RetryPolicy.parse(spec).restart(next_round, stall, gain) is expected


def test_obstacle_phase_visibility():
    hidden, visible = load_domain("gridworld"), load_domain("gridworld", ["obs_idx"])
    instance = visible.load_instances("grid_20_balanced.csv")[0]
    assert "obs_idx" not in [v.name for v in hidden.spec(instance).variables]
    obs = {v.name: v for v in visible.spec(instance).variables}["obs_idx"]
    assert (obs.low, obs.high) == (0, 5) and "(0,2)" in obs.description
    assert "can observe" in visible.description(instance) and "cannot observe" in hidden.description(instance)
    uuv, uuv_extra = load_domain("uuv"), load_domain("uuv", ["obs_idx"])   # not in the model: ignored
    scenario = uuv.load_instances("uuv_paper.csv")[0]
    assert [v.name for v in uuv_extra.spec(scenario).variables] == [v.name for v in uuv.spec(scenario).variables]


@needs_prism
def test_joint_best_case_query():
    domain = load_domain("gridworld", ["obs_idx"])
    verifier = PolicyVerifier(domain, domain.load_instances("grid_20_balanced.csv")[0], load_config())
    spec = verifier.spec
    assert verifier.jointly_feasible() is True                      # bare MDP: solvable
    complete = SymbolicPolicy.from_raw(spec.variables, list(spec.actions),
                                       [("!g1 & obs_idx = 0", "right"), ("!g1", "down"), ("true", "up")])
    v = verifier.verify(complete, analysis=False)                    # periodic chain: needs Gauss-Seidel
    assert v.uncovered_situations == 0
    passes = all(r.satisfied(v.worst[r.name]) for r in spec.requirements)
    assert verifier.jointly_feasible(complete) is passes             # complete policy: joint == pass
