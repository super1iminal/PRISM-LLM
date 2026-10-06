"""Reading finished runs: run facts from config.json (or the pre-config record) and legacy keep-best."""
import json

import pytest

from config import load_config
from core.domain import Requirement, load_domain
from results_io import (LEGACY_RESULTS, PRE_CONFIG_RUNS, legacy_kept, met_and_shortfall, requirements_by_sample,
                        run_facts)


def test_run_facts_come_from_the_recorded_config(tmp_path):
    cfg = load_config("U1", ["planner.max_rounds=3"])
    (tmp_path / "config.json").write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    facts = run_facts(tmp_path)
    assert (facts.approach, facts.domain, facts.dataset, facts.max_rounds, facts.recorded) == \
        ("symbolic", "uuv", "uuv_paper.csv", 3, True)
    assert facts.model == cfg.llm.model


def test_legacy_run_facts_use_the_legacy_round_budget(tmp_path):
    cfg = load_config("B1", ["legacy.max_rounds=4"])
    (tmp_path / "config.json").write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    (tmp_path / LEGACY_RESULTS).write_bytes(b"")
    facts = run_facts(tmp_path)
    assert facts.approach == "legacy" and facts.max_rounds == 4


def test_runs_without_config_json_read_as_the_pre_config_record(tmp_path):
    facts = run_facts(tmp_path)
    assert not facts.recorded and facts.domain == PRE_CONFIG_RUNS["domain"]
    assert (facts.dataset, facts.max_rounds) == (PRE_CONFIG_RUNS["dataset"], PRE_CONFIG_RUNS["max_rounds"])


def test_requirements_by_sample_follow_the_runs_domain_and_dataset(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(load_config("U1").to_dict()), encoding="utf-8")
    north_sea, caribbean = requirements_by_sample(tmp_path)
    energy = {r.name: r for r in north_sea}["energy_budget"]
    assert (energy.bound, energy.threshold) == ("<=", 26.5)
    assert {r.name: r for r in caribbean}["energy_budget"].threshold == 62.5   # per-instance thresholds
    assert [r.name for r in requirements_by_sample(tmp_path / "missing")[0]][:2] == ["goal1", "goal2"]


def test_met_and_shortfall_use_each_requirements_bound():
    reach = Requirement("reach", 'F "goal"', 0.8)
    energy = Requirement("energy", 'F "done"', 50.0, bound="<=", reward="energy")
    assert met_and_shortfall([reach, energy], {"reach": 0.9, "energy": 40.0}) == (2, 0.0)
    met, shortfall = met_and_shortfall([reach, energy], {"reach": 0.7, "energy": 60.0})
    assert met == 0 and shortfall == pytest.approx(0.1 + 10 / 50)   # reward shortfall is relative
    assert met_and_shortfall([reach, energy], {"energy": 60.0}) == (0, pytest.approx(0.2))   # a subset is fine
    with pytest.raises(ValueError, match="not requirements of this instance"):
        met_and_shortfall([reach], {"goal1": 0.9})


def test_legacy_kept_replays_keep_best():
    instance = load_domain("gridworld").load_instances("grid_20_balanced.csv")[0]
    names = ["goal1", "goal2", "goal3", "seq_1_before_2", "seq_2_before_3", "complete_sequence",
             "avoid_moving_seg1", "avoid_moving_seg2", "avoid_moving_seg3"]
    worse = dict.fromkeys(names, 0.9) | {"goal2": 0.1, "goal3": 0.1}        # 2 requirements missed
    tie_low = dict.fromkeys(names, 0.81) | {"goal3": 0.5}                   # 1 missed, lower score
    better = dict.fromkeys(names, 0.85) | {"goal3": 0.5}                    # 1 missed, higher score: kept
    record = {"iteration_prism_probs": [worse, tie_low, better]}
    assert legacy_kept(record, instance) == better
    assert legacy_kept({"final_prism_probs": worse, **record}, instance) == worse   # newer runs store it
    assert legacy_kept({}, instance) == {}
