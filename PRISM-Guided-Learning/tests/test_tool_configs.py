"""Tool config folders (configs/ablation/): they load, and they are what the tools use."""
import sys

import run_ablation
from config import load_file


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
