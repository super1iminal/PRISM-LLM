"""configs/plot/: every figure config loads with its own script's schema, and plot_config guards the file."""
import importlib
import sys
from pathlib import Path

import matplotlib
import pytest
import yaml

from config import CONFIG_DIR, plot_config

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viz"))
from loaders import short_model  # noqa: E402

PLOT_CONFIGS = sorted((CONFIG_DIR / "plot").glob("*.yaml"))


def test_there_is_a_config_per_committed_figure():
    names = {p.stem for p in PLOT_CONFIGS}
    assert {"grid20", "catchall_ablation", "catchall_ablation_full", "budget_grid20", "uuv", "uuv_summary",
            "ablation_summary", "ablation_grid", "compare_grid20"} <= names


@pytest.mark.parametrize("path", PLOT_CONFIGS, ids=lambda p: p.stem)
def test_every_plot_config_loads_with_its_scripts_schema(path):
    script = yaml.safe_load(path.read_text(encoding="utf-8"))["script"]
    module = importlib.import_module(script)
    cfg = plot_config(module.PlotConfig, script, [str(path)])
    assert cfg.script == script


def test_plot_config_rejects_the_wrong_script_and_unknown_keys():
    import plot_runs
    with pytest.raises(SystemExit, match="is a config for plot_comparison.py, not plot_runs.py"):
        plot_config(plot_runs.PlotConfig, "plot_runs", [str(CONFIG_DIR / "plot" / "grid20.yaml")])
    with pytest.raises(ValueError, match="unknown config key"):
        plot_config(plot_runs.PlotConfig, "plot_runs", [str(CONFIG_DIR / "plot" / "catchall_ablation.yaml"),
                                                        "--set", "colour=red"])


def test_short_model_drops_the_quantization_tag():
    assert short_model("qwen3:14b-q4_K_M") == "qwen3:14b"
    assert short_model("llama3.1:70b") == "llama3.1:70b"
