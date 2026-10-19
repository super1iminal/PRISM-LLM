"""run_symbolic.run end to end on the tiny grid, with a scripted LLM instead of Ollama."""
import json
import shutil

import pandas as pd
import pytest

import run_symbolic
from config import load_config
from fakes import ScriptedBackend, answer
from results_io import SYMBOLIC_RESULTS

needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")


@needs_prism
def test_run_writes_results_and_releases_its_logs(tmp_path, tiny_grid, monkeypatch):
    monkeypatch.setattr(run_symbolic, "make_backend", lambda config: ScriptedBackend([answer(("true", "right"))]))
    cfg = load_config(overrides=[f"domain.dataset={tiny_grid.as_posix()}", "run.workers=1", "planner.max_rounds=2"])
    run_dir = tmp_path / "run"
    run_symbolic.run(cfg, str(run_dir))

    df = pd.read_parquet(run_dir / SYMBOLIC_RESULTS).reset_index()
    assert len(df) == 1 and df.success.tolist() == [True] and df.llm_calls.tolist() == [1]
    assert df.final_check.tolist() == ["interval iteration"]
    record = json.loads((run_dir / "outputs" / "sample_000.json").read_text(encoding="utf-8"))
    assert record["final_rules"] == [{"condition": "true", "action": "right"}]
    assert json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["planner"]["max_rounds"] == 2
    assert "Success after 1 attempts" in (run_dir / "worker_0.log").read_text(encoding="utf-8")
    assert "Results saved to" in (run_dir / "main.log").read_text(encoding="utf-8")
    shutil.rmtree(run_dir)   # every log file is closed (Windows refuses to delete open files)
