"""src/calibrate.py: each requirement's range over all controllers, and the domain's reference policies."""
import shutil

import pytest

from calibrate import calibrate

needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")


@needs_prism
def test_calibration_prints_the_range_and_every_reference_policy(tmp_path, capsys):
    data = tmp_path / "pacman_short.csv"
    data.write_text("name,max_steps,crash_threshold\nsteps_05,5,0.56\n", encoding="utf-8")
    calibrate("pacman", str(data))
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("steps_05: crash: best 0.5511")       # the benchmark's published minimum
    assert len(out) == 1 + 6                                       # one line per reference policy
    assert any(line.split()[0] == "keep_heading" for line in out[1:])
