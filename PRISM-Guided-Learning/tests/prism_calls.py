"""Record every PRISM command the planner runs, to check that a change to how PRISM is driven runs the
same commands on the same inputs as before.

Commands are captured where `core/prism.py` starts PRISM (`subprocess.run` / `subprocess.Popen`), and
normalized so they compare across runs: the PRISM path becomes "prism", input files (model and
properties) become their name plus a hash of their content, and other temporary paths their name.

`collect()` runs the saved-run replays and a tiny-grid run one instance at a time and returns
{case: [command, ...]}. `tests/data/prism_calls.json` holds its output from before PRISM calls became
yielded tasks (regenerate with `python tests/prism_calls.py` only after a deliberate change).
"""
import hashlib
import json
import logging
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "data" / "prism_calls.json"
INPUTS = ("model.prism", "props.props")
LOG = logging.getLogger("prism_calls")


def normalize(cmd: List[str]) -> List[str]:
    out = ["prism"]
    for arg in cmd[1:]:
        name = os.path.basename(arg)
        if name in INPUTS and os.path.isfile(arg):
            out.append(f"{name}#{hashlib.sha256(Path(arg).read_bytes()).hexdigest()[:16]}")
        elif os.path.isabs(arg) and "prism_" in arg:
            out.append(name)
        else:
            out.append(arg)
    return out


@contextmanager
def recording():
    """Within the block, every PRISM command started by core.prism is appended to the yielded list."""
    import core.prism as prism_module
    calls: List[List[str]] = []

    def run(cmd, *args, **kwargs):
        calls.append(normalize(cmd))
        return subprocess.run(cmd, *args, **kwargs)

    def popen(cmd, *args, **kwargs):
        calls.append(normalize(cmd))
        return subprocess.Popen(cmd, *args, **kwargs)

    shim = SimpleNamespace(**{k: getattr(subprocess, k) for k in dir(subprocess) if not k.startswith("_")})
    shim.run, shim.Popen = run, popen
    original = prism_module.subprocess
    prism_module.subprocess = shim
    try:
        yield calls
    finally:
        prism_module.subprocess = original


def collect() -> Dict[str, List[List[str]]]:
    """Every case's PRISM commands, solving one instance at a time."""
    from config import load_config
    from core.domain import load_domain
    from core.planner import SymbolicPlanner
    from fakes import ScriptedBackend, answer, write_tiny_grid
    from test_scheduler import REPLAY_CASES, replay_backend, saved_run

    out: Dict[str, List[List[str]]] = {}
    for run, instances in REPLAY_CASES:
        cfg, samples = saved_run(run)
        domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
        by_id = {str(i.id): i for i in domain.load_instances(cfg.domain.dataset)}
        planner = SymbolicPlanner(domain, replay_backend({k: samples[k] for k in instances}), cfg)
        for inst in instances:
            with recording() as calls:
                planner.solve(by_id[inst], LOG)
            out[f"{run}/{inst}"] = calls

    # The tiny grid with the default config: joint query, blame, retries and the exact final check.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        grid = write_tiny_grid(Path(tmp))
        cfg = load_config(overrides=[f"domain.dataset={grid.as_posix()}", "llm.seed=1", "planner.max_rounds=3"])
        domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
        backend = ScriptedBackend([answer(("x = 0", "up")), answer(("true", "left")), answer(("true", "right"))])
        with recording() as calls:
            SymbolicPlanner(domain, backend, cfg).solve(domain.load_instances(str(grid))[0], LOG)
        out["tiny_grid"] = calls
    return out


if __name__ == "__main__":
    root = HERE.parent
    sys.path[:0] = [str(root / "src"), str(root), str(HERE)]
    BASELINE.parent.mkdir(exist_ok=True)
    data = collect()
    BASELINE.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print({case: len(calls) for case, calls in data.items()})
