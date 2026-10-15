"""src/experiments/: the run store (identity, corrections), experiment files, metrics, and open ties."""
import json

import pandas as pd
import pytest
import yaml

from config import load_config
from core.rules import ANY, fixed_order, open_ties
from experiments import base
from experiments.ablation import bootstrap_diff
from experiments.metrics import transfer_numbers
from experiments.store import RunStore, fingerprint, identity
from results_io import RESULT_FILES

ACTIONS = ["up", "right", "down", "left"]
GRID_ONLY_FEATURES = {"hazard", "risky", "toward", "closer", "obstacle", "patrol", "early"}


def test_open_ties_inverts_fixed_order():
    rules = [("closer & !wall", ANY), ("!wall", "up"), ("!wall", "down"), ("true", ANY)]
    fixed = fixed_order(rules, ["down", "left", "up", "right"])
    assert open_ties(fixed, ACTIONS) == [("closer & !wall", ANY), ("!wall", "up"), ("!wall", "down"), ("true", ANY)]


def test_open_ties_keeps_partial_and_separate_runs():
    rules = [("a", "up"), ("a", "down"), ("b", "up"), ("a", "right"), ("a", "left")]
    assert open_ties(rules, ACTIONS) == rules


def test_identity_ignores_names_parallelism_and_seed():
    cfg = load_config("haiku_ts_order")
    other = load_config("haiku_ts_order", ["run.workers=1", "llm.seed=7", "run.scheduler=threads"])
    assert fingerprint(cfg) == fingerprint(other)
    assert "condition" not in identity(cfg)
    assert fingerprint(cfg) != fingerprint(load_config("haiku_ts_order", ["rules.any_order=none"]))
    # a condition that differs from the core only in these settings is the same configuration
    assert fingerprint(load_config("haiku_ts_norestart")) == fingerprint(
        load_config("haiku_ts_order", ["rules.any_order=none"]))


def _fake_run(root, name, cfg, done=True, fix=None):
    run = root / name / "seed_none_1"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    if done:
        (run / RESULT_FILES[cfg.approach]).write_text("", encoding="utf-8")
    if fix:
        (run / "identity.json").write_text(json.dumps(fix), encoding="utf-8")
    return run


def test_store_matches_across_names_and_applies_corrections(tmp_path):
    cfg = load_config("haiku_ts_order")
    a = _fake_run(tmp_path, "renamed", cfg)
    old = load_config("haiku_ts_order", ["rules.hidden_features=[]"])     # recorded before patrol/early existed
    b = _fake_run(tmp_path, "older", old, fix={"note": "x", "rules": {"hidden_features": ["patrol", "early"]}})
    c = _fake_run(tmp_path, "unfinished", cfg, done=False)
    store = RunStore([tmp_path])
    assert store.runs(cfg) == [b, a] or sorted(store.runs(cfg)) == sorted([a, b])
    assert store.pending(cfg) == [c]
    assert store.runs(old) == []


def _experiment(tmp_path, cells, kind="ablation"):
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump({
        "kind": kind, "title": "t", "repeats": 2,
        "benchmarks": {"grids": {"base": "haiku_ts_order", "targets": "12-19"},
                       "families": {"base": "haiku_fam_order", "targets": "16-35"}},
        "cells": cells}), encoding="utf-8")
    return path


def test_cells_share_runs_and_name_new_ones(tmp_path):
    root = tmp_path / "results"
    _fake_run(root, "haiku_ts_norestart", load_config("haiku_ts_norestart"))
    path = _experiment(tmp_path, [
        {"name": "Reference"},
        {"name": "Worst case", "set": ["rules.any_order=none"]},
        {"name": "No closer", "only": ["families"],
         "set": {"families": ["rules.hidden_features=[closer, hazard]"]}}])
    exp = base.experiment(path, RunStore([root]))
    units = {(u.cell.name, u.bench.name): u for u in exp.units}
    assert len(units) == 5
    shared = units[("Worst case", "grids")]
    assert [r.parent.name for r in shared.runs] == ["haiku_ts_norestart"] and shared.missing == 1
    assert exp._new_dirs(shared) == [root / "haiku_ts_norestart" / "seed_none_2"]
    fresh = units[("Worst case", "families")]
    assert fresh.run_name == "haiku_fam_order+worst-case" and fresh.missing == 2
    assert fresh.cfg.rules.any_order == "none"
    assert units[("No closer", "families")].cfg.rules.hidden_features == ["closer", "hazard"]


def test_cell_given_per_benchmark_skips_the_others(tmp_path):
    """A cell whose `set` (or `condition`) names some benchmarks only is skipped on the others, never run there as
    the base configuration; `only` cannot name a benchmark the mapping leaves out."""
    path = _experiment(tmp_path, [
        {"name": "Reference"},
        {"name": "Families only", "set": {"families": ["rules.hidden_features=[closer, hazard]"]}},
        {"name": "Other model", "condition": {"grids": "haiku_ts_norestart"}}])
    exp = base.experiment(path, RunStore([tmp_path / "results"]))
    assert sorted((u.cell.name, u.bench.name) for u in exp.units) == [
        ("Families only", "families"), ("Other model", "grids"), ("Reference", "families"), ("Reference", "grids")]
    with pytest.raises(ValueError, match="leaves out"):
        base.experiment(_experiment(tmp_path, [{"name": "Bad", "only": ["grids", "families"],
                                                "set": {"families": ["rules.any_order=none"]}}]), RunStore([]))
    with pytest.raises(ValueError, match="unknown benchmarks"):
        base.experiment(_experiment(tmp_path, [{"name": "Typo", "set": {"pacman": ["rules.any_order=all"]}}]),
                        RunStore([]))


def test_shared_configuration_launches_once(tmp_path):
    root = tmp_path / "results"
    a = base.experiment(_experiment(tmp_path, [{"name": "Worst case", "set": ["rules.any_order=none"],
                                                "only": ["grids"]}]), RunStore([root]))
    path = tmp_path / "other.yaml"
    path.write_text((tmp_path / "exp.yaml").read_text(encoding="utf-8").replace("Worst case", "No choice"),
                    encoding="utf-8")
    b = base.experiment(path, RunStore([root]))
    assert len(base.jobs([a, b], run=True, check=False, baselines=False)) == 2      # 2 repeats, once


def test_core_names_are_the_comparison_cells():
    names = base.core_names()
    assert {"haiku_ts_order", "haiku_fam_order", "sonnet5_5_ts_order"} <= set(names)
    assert not any("+" in n for n in names)


def test_transfer_numbers_groups_library_and_everywhere(tmp_path):
    rows = [(0, t, t in (1, 2)) for t in (1, 2, 3, 4)] + [(1, t, t in (3,)) for t in (1, 2, 3, 4)] + \
           [("tmpl", t, t == 4) for t in (1, 2, 3, 4)]
    path = tmp_path / "t.csv"
    pd.DataFrame([{"source": s, "target": t, "certified": c, "trained": False, "met": 1, "requirements": 1}
                  for s, t, c in rows]).to_csv(path, index=False)
    groups, source_groups = {"a": "1-2", "b": "3-4"}, {"0": "a", "1": "b", "tmpl": "*"}
    n = transfer_numbers(path, [1, 2, 3, 4], groups, source_groups, sources=["0", "1"])
    assert n["checked"] and n["certified"] == 1.5 and n["by_group"] == {"a": 2, "b": 1}
    assert n["certified_all"] == 1.5 and n["library"] == 3
    t = transfer_numbers(path, [1, 2, 3, 4], groups, source_groups, sources=["tmpl"])
    assert t["by_group"] == {"a": 0, "b": 1} and t["certified_all"] == 1


def test_bootstrap_diff():
    d, lo, hi = bootstrap_diff([6, 7, 8], [5, 5, 5])
    assert d == pytest.approx(2) and lo <= d <= hi and lo > 0


def test_experiment_files_load():
    for path in sorted(base.EXPERIMENTS_DIR.glob("rq*.yaml")):
        exp = base.experiment(path, RunStore([]))
        assert exp.units, path
        for u in exp.units:      # no gridworld override reaches another domain
            rows = len(pd.read_csv(base.PROJECT_ROOT / "domains" / u.cfg.domain.name / "data" / u.cfg.domain.dataset))
            ids = [i for key in ("train_sets", "hidden_sets", "pool_sets") for s in getattr(u.cfg.run, key) for i in s]
            assert all(i < rows for i in ids), (path, u.cell.name, u.bench.name)
            if u.cfg.domain.name != "gridworld":
                assert not set(u.cfg.rules.hidden_features) & GRID_ONLY_FEATURES, (path, u.cell.name)


def test_best_configuration_per_method(tmp_path):
    """Per benchmark and method: the highest mean certified, then the larger library, then the lower cost; a
    configuration with an unchecked run does not compete."""
    exp = base.experiment(_experiment(tmp_path, [{"name": "A", "method": "Ours"}, {"name": "B", "method": "Ours"},
                                                 {"name": "C", "method": "Ours"}, {"name": "D"}],
                                      kind="comparison"), RunStore([]))
    rows = [("grids", "A", True, 6, 8, 0.01), ("grids", "A", True, 7, 8, 0.01), ("grids", "B", True, 6.5, 7, 0.5),
            ("grids", "C", False, 9, 8, 0.0), ("families", "A", True, 5, 30, 0.02), ("families", "B", True, 5, 30, 0.01),
            ("grids", "D", True, 1, 1, 0.0)]
    runs = pd.DataFrame(rows, columns=["bench", "cell", "checked", "certified", "library", "dollars"])
    runs = runs.assign(method=runs.cell.map(exp.methods()))
    assert exp.chosen(runs) == {("grids", "Ours"): "A", ("families", "Ours"): "B", ("grids", "D"): "D"}


def test_index_lists_every_cell_and_baseline():
    from experiments.index import experiment_files, index_page
    exps = [base.experiment(p, RunStore([])) for p in experiment_files()]
    page = index_page(exps, None)
    assert all(c.name in page for e in exps for c in e.spec["cells"])
    assert "(baseline)" in page and "**Missing:**" in page


def test_held_out_leaves_out_training_targets():
    shared = base.benchmark_specs({"benchmarks": "rq2"})
    bench = {name: base.Benchmark(name=name, **spec) for name, spec in shared.items()}
    assert len(base.held_out(bench["pacman"])) == 5 and len(base.held_out(bench["grids"])) == 8
