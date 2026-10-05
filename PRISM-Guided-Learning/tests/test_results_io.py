"""Reading finished runs: run facts from config.json (or the pre-config record) and legacy keep-best."""
import json

from config import load_config
from core.domain import load_domain
from results_io import LEGACY_RESULTS, PRE_CONFIG_RUNS, legacy_kept, run_facts


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
