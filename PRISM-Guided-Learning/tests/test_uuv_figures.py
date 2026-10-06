"""The UUV figure scripts on the committed UUV run, which predates the energy requirement."""
import shutil
import sys

import matplotlib
import pytest

from conftest import ROOT

matplotlib.use("Agg")
sys.path.insert(0, str(ROOT / "viz"))
import plot_domain  # noqa: E402
import plot_uuv_summary  # noqa: E402

pytestmark = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")


def run(module, config, out, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", [module.__name__, f"configs/plot/{config}.yaml", "--set", f"out={out.as_posix()}"])
    module.main()
    return out.with_suffix(".md").read_text(encoding="utf-8").splitlines()


def test_plot_domain_shows_every_requirement_and_what_a_run_lacks(tmp_path, monkeypatch):
    lines = run(plot_domain, "uuv", tmp_path / "uuv.png", monkeypatch)
    assert lines[0].endswith("| no_thruster_failure | done_in_time | energy_budget |")
    run_rows = [line for line in lines if "qwen" in line]
    assert len(run_rows) == 2 and all("(no energy_budget)" in row and row.endswith("| — |") for row in run_rows)
    stay = [line for line in lines if "| stay |" in line]
    assert len(stay) == 2 and all(row.split("|")[-2].strip() != "—" for row in stay)   # references get energy
    assert (tmp_path / "uuv.png").exists()


def test_uuv_summary_checks_the_energy_budget(tmp_path, monkeypatch):
    lines = run(plot_uuv_summary, "uuv_summary", tmp_path / "uuv_summary.png", monkeypatch)
    assert "| meets every requirement |" in lines[0]
    ours = [line for line in lines if "ours (" in line]
    assert [row.split("|")[-3].strip() for row in ours] == ["yes", "no"]   # Caribbean exceeds its budget
    assert "59.08..4723.29" in "\n".join(lines)   # the paper's Table 2, with the paper's solver
    assert "energy_budget <= 26.5" in lines[-1]
