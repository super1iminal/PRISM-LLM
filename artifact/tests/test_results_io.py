"""Reading finished runs: run facts from config.json."""
import json

import pytest

from config import load_config
from core.domain import Requirement, load_domain
from results_io import met_and_shortfall, requirements_by_sample, run_facts, sample_records


def test_run_facts_come_from_the_recorded_config(tmp_path):
    cfg = load_config("U1", ["planner.max_rounds=3"])
    (tmp_path / "config.json").write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    facts = run_facts(tmp_path)
    assert (facts.domain, facts.dataset, facts.max_rounds) == ("uuv", "uuv_paper.csv", 3)
    assert facts.model == cfg.llm.model


def test_requirements_by_sample_follow_the_runs_domain_and_dataset(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(load_config("U1").to_dict()), encoding="utf-8")
    north_sea, caribbean = requirements_by_sample(tmp_path)
    energy = {r.name: r for r in north_sea}["energy_budget"]
    assert (energy.bound, energy.threshold) == ("<=", 26.5)
    assert {r.name: r for r in caribbean}["energy_budget"].threshold == 62.5   # per-instance thresholds


def test_met_and_shortfall_use_each_requirements_bound():
    reach = Requirement("reach", 'F "goal"', 0.8)
    energy = Requirement("energy", 'F "done"', 50.0, bound="<=", reward="energy")
    assert met_and_shortfall([reach, energy], {"reach": 0.9, "energy": 40.0}) == (2, 0.0)
    met, shortfall = met_and_shortfall([reach, energy], {"reach": 0.7, "energy": 60.0})
    assert met == 0 and shortfall == pytest.approx(0.1 + 10 / 50)   # reward shortfall is relative
    assert met_and_shortfall([reach, energy], {"energy": 60.0}) == (0, pytest.approx(0.2))   # a subset is fine
    with pytest.raises(ValueError, match="not requirements of this instance"):
        met_and_shortfall([reach], {"goal1": 0.9})


def test_sample_records_are_keyed_by_sample_id_in_order(tmp_path):
    (tmp_path / "outputs").mkdir()
    for sid in (10, 2, 0):
        (tmp_path / "outputs" / f"sample_{sid:03d}.json").write_text(json.dumps({"id": sid}), encoding="utf-8")
    records = sample_records(tmp_path)
    assert list(records) == [0, 2, 10] and records[10] == {"id": 10}
    assert sample_records(tmp_path / "missing") == {}
