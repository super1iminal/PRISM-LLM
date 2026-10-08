"""Tool config folders (configs/regression/, configs/ablation/): they load, and they are what the tools use."""
import sys

import pytest

import run_ablation
from config import load_file
from regression import DEFAULT_CONFIG as REGRESSION_CONFIG, RegressionConfig


def test_regression_config_loads_and_overrides():
    reg = load_file(REGRESSION_CONFIG, RegressionConfig)
    assert reg.epsilon == "1e-9" and reg.fallback_epsilon == "1e-12"   # passed to PRISM as written
    assert isinstance(reg.tol_stored, float) and isinstance(reg.tol_exact, float) and reg.workers >= 1
    assert load_file(REGRESSION_CONFIG, RegressionConfig, ["workers=1"]).workers == 1
    with pytest.raises(ValueError, match="unknown config key"):
        load_file(REGRESSION_CONFIG, RegressionConfig, ["tolerance=1"])


def test_ablation_seeds_come_from_its_config(tmp_path, monkeypatch, capsys):
    seeds = load_file(run_ablation.DEFAULT_CONFIG, run_ablation.AblationConfig).seeds
    monkeypatch.setattr(sys, "argv", ["run_ablation.py", "B2", "U1", "--dry-run", "--out-root", str(tmp_path)])
    run_ablation.main()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2 * len(seeds) and all("would run" in line for line in lines)
    assert [f"seed {s}]" in line for line, s in zip(lines, seeds + seeds)] == [True] * len(lines)

    monkeypatch.setattr(sys, "argv", ["run_ablation.py", "B2", "--seeds", "7", "--dry-run", "--out-root", str(tmp_path)])
    run_ablation.main()
    assert capsys.readouterr().out.count("seed 7]") == 1   # --seeds overrides the config

    monkeypatch.setattr(sys, "argv", ["run_ablation.py", "B2", "--unseeded", "2", "--dry-run", "--out-root", str(tmp_path)])
    run_ablation.main()
    lines = capsys.readouterr().out.splitlines()
    assert [line.endswith(str(tmp_path / "B2" / f"seed_none_{k}")) for line, k in zip(lines, (1, 2))] == [True, True]
