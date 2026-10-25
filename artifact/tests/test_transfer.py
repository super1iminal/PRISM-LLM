"""src/transfer.py: frozen final rules from a run, certified on every instance of a dataset."""
import json
import shutil

import pandas as pd
import pytest

import transfer
from config import load_config, load_run_config
from core.domain import load_domain
from core.prism import PrismRunner

needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")


def fake_run(tmp_path, tiny_grid, records):
    """A finished run on the tiny grid with the given per-sample records."""
    run = tmp_path / "run"
    (run / "outputs").mkdir(parents=True)
    cfg = load_config(overrides=[f"domain.dataset={tiny_grid.as_posix()}"])
    (run / "config.json").write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    for sid, record in enumerate(records):
        (run / "outputs" / f"sample_{sid:03d}.json").write_text(json.dumps(record), encoding="utf-8")
    return run


def test_final_rules_skip_errors_and_training_sets_list_their_members(tmp_path, tiny_grid):
    run = fake_run(tmp_path, tiny_grid, [
        {"instance": "0", "final_rules": [{"condition": "true", "action": "right"}]},
        {"instance": "1", "error": "LLMError: server gone", "iterations": []},
        {"instance": "set2", "final_rules": [], "members": [{"instance": "4"}, {"instance": "5"}]},
    ])
    assert transfer.final_rules(run) == {0: [("true", "right")], 2: []}
    assert transfer.training_instances(run) == {0: [0], 1: [1], 2: [4, 5]}


@needs_prism
def test_certify_checks_every_source_on_a_target(tiny_grid):
    cfg = load_config()
    domain = load_domain("gridworld", cfg.domain.visible_extra)
    target = domain.load_instances(str(tiny_grid))[0]
    rows = transfer.certify(domain, cfg, PrismRunner(cfg.prism), target,
                            {0: [("true", "right")], 1: [("true", "left")], 2: [("z > 1", "up")]})
    by_source = {r["source"]: r for r in rows}
    assert by_source[0]["certified"] and by_source[0]["met"] == by_source[0]["requirements"]
    assert by_source[1]["parsed"] and not by_source[1]["certified"]
    assert not by_source[2]["parsed"] and "unknown variable 'z'" in by_source[2]["error"]


@needs_prism
def test_main_writes_one_row_per_source_and_target(tmp_path, tiny_grid, capsys):
    run = fake_run(tmp_path, tiny_grid, [{"instance": "0", "final_rules": [{"condition": "true", "action": "right"}]}])
    transfer.main([str(run), "--set", "workers=1"])
    df = pd.read_csv(run / f"transfer_{tiny_grid.stem}.csv")
    assert df[["source", "target", "certified", "trained"]].values.tolist() == [[0, 0, True, True]]
    assert "certifies 1/1" in capsys.readouterr().out


def test_workers_come_from_the_transfer_config():
    from config import load_file
    assert load_file(transfer.DEFAULT_CONFIG, transfer.TransferConfig).workers >= 1
    assert load_file(transfer.DEFAULT_CONFIG, transfer.TransferConfig, ["workers=3"]).workers == 3


def test_a_run_config_keeps_its_recorded_settings_and_fills_newer_keys_with_defaults(tmp_path):
    recorded = load_config("U1", ["llm.seed=7"]).to_dict()
    del recorded["rules"]["hidden_features"]          # a key the run predates
    (tmp_path / "config.json").write_text(json.dumps(recorded), encoding="utf-8")
    cfg = load_run_config(tmp_path)
    assert (cfg.domain.name, cfg.llm.seed, cfg.rules.hidden_features) == ("uuv", 7, [])


def test_target_specs_are_ids_and_inclusive_ranges():
    assert transfer.parse_ids("12-14") == [12, 13, 14]
    assert transfer.parse_ids("0,4, 8") == [0, 4, 8]
    assert transfer.parse_ids("0-1,5") == [0, 1, 5]


def test_a_partial_check_keeps_the_other_targets_of_an_earlier_csv(tmp_path):
    path = tmp_path / "transfer.csv"
    pd.DataFrame({"source": [0, 0, 0], "target": [0, 1, 2], "certified": [True, False, False]}).to_csv(path, index=False)
    new = pd.DataFrame({"source": [0], "target": [1], "certified": [True]})
    merged = transfer.merge(new, path).sort_values("target")
    assert merged[["target", "certified"]].values.tolist() == [[0, True], [1, True], [2, False]]
    assert transfer.merge(new, tmp_path / "missing.csv").equals(new)
