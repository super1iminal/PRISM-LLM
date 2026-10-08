"""PrismRunner without PRISM: input files, the command lines of check and run, the solver fallback and timeouts."""
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from config import load_config
from core.prism import PrismError, PrismRunner

PRISM = load_config().prism


def runner_capturing(monkeypatch, outputs, **kwargs):
    """A runner whose PRISM calls return `outputs` in turn; the commands land in `runner.calls`."""
    runner = PrismRunner(PRISM, prism_path="prism-exe", **kwargs)
    runner.calls = []
    replies = iter(outputs)

    def call(cmd):
        runner.calls.append(cmd)
        return next(replies)
    monkeypatch.setattr(runner, "_call", call)
    return runner


def test_write_inputs_ends_every_property_with_one_semicolon(tmp_path):
    model_path, props_path = PrismRunner._write_inputs(str(tmp_path), "mdp\n", ['P=? [ F "a" ];', 'Pmax=? [ G "b" ]'])
    assert Path(model_path).read_text(encoding="utf-8") == "mdp\n"
    assert Path(props_path).read_text(encoding="utf-8") == 'P=? [ F "a" ];\nPmax=? [ G "b" ];\n'


@pytest.mark.parametrize("stdout,expected", [("Result: true\n", True), ("Result: false\n", False),
                                             ("... not currently supported with linear programming ...", None)])
def test_check_reads_the_answer(monkeypatch, stdout, expected):
    runner = runner_capturing(monkeypatch, [stdout], extra_args=["-epsilon", "1e-10"])
    assert runner.check("mdp", "multi(...)") is expected
    cmd = runner.calls[0]
    assert cmd[0] == "prism-exe" and cmd[3:] == [f"-{PRISM.multi_engine}", f"-{PRISM.multi_method}",
                                                 "-javamaxmem", PRISM.java_max_mem, "-maxiters", str(PRISM.max_iters)]


def test_check_raises_on_errors(monkeypatch):
    with pytest.raises(PrismError):
        runner_capturing(monkeypatch, ["Error: syntax error"]).check("mdp", "multi(...)")


def test_run_falls_back_while_prism_does_not_converge(monkeypatch):
    attempts = len(PRISM.fallback_methods) + 1
    runner = runner_capturing(monkeypatch, ["Iterative method did not converge"] * attempts, extra_args=["-x"])
    with pytest.raises(PrismError):   # no states file was exported
        runner.run("mdp", ['P=? [ F "a" ]'], export_transitions=True)
    assert [cmd[-2] for cmd in runner.calls] == [f"-{m}" for m in [PRISM.method, *PRISM.fallback_methods]]
    first = runner.calls[0]
    assert first[3:8] == ["-explicit", "-javamaxmem", PRISM.java_max_mem, "-maxiters", str(PRISM.max_iters)]
    assert "-exporttrans" in first and first[-1] == "-x"   # extra args go last, on every attempt


def test_timeout_kills_the_whole_process_tree(tmp_path):
    """Like prism.bat starting the JVM, the script starts a grandchild that holds the output pipe open."""
    sleeper = f'"{sys.executable}" -c "import time; time.sleep(60)"'
    if os.name == "nt":
        script = tmp_path / "prism.bat"
        script.write_text(f"@{sleeper}\n", encoding="utf-8")
    else:
        script = tmp_path / "prism.sh"
        script.write_text(f"#!/bin/sh\n{sleeper}\n", encoding="utf-8")
        script.chmod(0o755)
    runner = PrismRunner(replace(PRISM, timeout_s=1), prism_path=str(script))
    start = time.time()
    with pytest.raises(subprocess.TimeoutExpired):
        runner._call([str(script)])
    assert time.time() - start < 30
