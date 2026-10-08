"""Run PRISM on an MDP and parse per-state results, reachable states and transitions."""
import os
import re
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Optional, Sequence, Tuple

from config import PrismConfig
from settings import get_prism_path

StateKey = Tuple  # valuation of all model variables, in PRISM's variable order


class PrismError(RuntimeError):
    pass


@dataclass
class Choice:
    action: Optional[str]
    successors: List[Tuple[int, float]]    # (state index, probability)


@dataclass
class PrismResult:
    variables: List[str]                          # model variable names (PRISM order)
    states: List[StateKey]                        # reachable states, indexed as PRISM does
    initial_values: List[float]                   # one per property: value in the initial state
    state_values: List[Optional[List[float]]]     # one per property: per-state values (printall filters)
    choices: List[List[Choice]] = field(default_factory=list)   # per state (only if transitions exported)
    initial_state: int = 0
    stdout: str = ""

    def index_of(self) -> Dict[StateKey, int]:
        return {s: i for i, s in enumerate(self.states)}

    @cached_property
    def positions(self) -> Dict[str, int]:
        """Variable name -> its position in every state tuple."""
        return {name: i for i, name in enumerate(self.variables)}


def _parse_value(text: str):
    if text == "true":
        return True
    if text == "false":
        return False
    return int(text)


def _parse_states(path: str) -> Tuple[List[str], List[StateKey]]:
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip()
        variables = header.strip("()").split(",") if header != "()" else []
        states = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, vals = line.split(":", 1)
            vals = vals.strip("()")
            states.append(tuple(_parse_value(v) for v in vals.split(",")) if vals else ())
    return variables, states


def _parse_transitions(path: str, num_states: int) -> List[List[Choice]]:
    choices: List[List[Choice]] = [[] for _ in range(num_states)]
    with open(path, encoding="utf-8") as f:
        f.readline()  # header: states choices transitions
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            s, c, t, p = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
            action = parts[4] if len(parts) > 4 else None
            while len(choices[s]) <= c:
                choices[s].append(Choice(action, []))
            choices[s][c].successors.append((t, p))
    return choices


def _parse_initial_states(path: str) -> List[int]:
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        init_label = next(int(k) for k, v in re.findall(r'(\d+)="([^"]*)"', header) if v == "init")
        return [int(line.split(":")[0]) for line in f
                if ":" in line and str(init_label) in line.split(":")[1].split()]


_SECTION_SPLIT = re.compile(r"^-{20,}\s*$", re.M)
_STATE_VALUE = re.compile(r"^(\d+):\((.*)\)=(\S+)\s*$")


def _parse_sections(stdout: str, num_properties: int, num_states: int):
    """Split stdout into one section per 'Model checking:' block and extract values."""
    sections = [s for s in _SECTION_SPLIT.split(stdout) if "Model checking:" in s]
    if len(sections) != num_properties:
        raise PrismError(f"expected {num_properties} results, PRISM reported {len(sections)}.\n"
                         + _error_excerpt(stdout))
    initial_values, state_values = [], []
    for section in sections:
        m = re.search(r"Value in the initial state: (\S+)", section) or re.search(r"^Result: (\S+)", section, re.M)
        if not m:
            raise PrismError("could not find a result in PRISM output:\n" + section[-2000:])
        initial_values.append(float(m.group(1)))
        if "Results (including zeros)" in section:
            values = [0.0] * num_states
            for line in section.splitlines():
                sm = _STATE_VALUE.match(line.strip())
                if sm:
                    values[int(sm.group(1))] = float(sm.group(3))
            state_values.append(values)
        else:
            state_values.append(None)
    return initial_values, state_values


def _error_excerpt(stdout: str) -> str:
    lines = [l for l in stdout.splitlines() if "Error" in l or "error" in l]
    return "\n".join(lines[:10]) or stdout[-2000:]


class PrismRunner:
    """Thin wrapper around the PRISM command line (explicit engine)."""

    def __init__(self, config: PrismConfig, extra_args: Sequence[str] = (), prism_path: Optional[str] = None):
        """`extra_args` are appended to every model-checking call (not to `check`)."""
        self.prism_path = prism_path or get_prism_path()
        self.extra_args = list(extra_args)
        self.java_max_mem = config.java_max_mem
        self.max_iters = config.max_iters
        self.timeout = config.timeout_s
        self.multi_args = [f"-{config.multi_engine}"] + ([f"-{config.multi_method}"] if config.multi_method else [])
        self.method_args = [f"-{config.method}"] if config.method else []
        self.fallback_args = [[f"-{m}"] for m in config.fallback_methods] if config.method else []

    def check(self, model: str, prop: str) -> Optional[bool]:
        """Decide one boolean property, e.g. a multi-objective achievability query `multi(...)`.

        Uses `prism.multi_engine` (sparse: the explicit engine has no multi-objective support) and
        `prism.multi_method` (lp: exact, where value iteration can fail on periodic chains). Returns None
        (undecided) when LP cannot handle the query.
        """
        with tempfile.TemporaryDirectory(prefix="prism_") as tmp:
            stdout = self._call(self._command(*self._write_inputs(tmp, model, [prop]), self.multi_args))
        if "not currently supported with linear programming" in stdout:
            # e.g. step-bounded objectives (UUV's deadline). PRISM's value-iteration alternative is
            # approximate and wrongly answers "no" at tight thresholds, so report "undecided".
            return None
        m = re.search(r"^Result: (true|false)", stdout, re.M)
        if "Error:" in stdout or not m:
            raise PrismError(_error_excerpt(stdout))
        return m.group(1) == "true"

    def run(self, model: str, properties: Sequence[str], export_transitions: bool = False) -> PrismResult:
        """Check `properties` (one per entry) on `model`.

        Properties wrapped in `filter(printall, ...)` also yield per-state values. The reachable
        states are always exported; transitions (with action labels) only if requested.
        """
        with tempfile.TemporaryDirectory(prefix="prism_") as tmp:
            states_path, trans_path, labels_path = (os.path.join(tmp, name)
                                                    for name in ("states.sta", "trans.tra", "labels.lab"))
            cmd = self._command(*self._write_inputs(tmp, model, properties), ["-explicit"])
            cmd += ["-exportstates", states_path, "-exportlabels", labels_path]
            if export_transitions:
                cmd += ["-exporttrans", trans_path]
            # Try the configured method, then the fallbacks, while PRISM reports non-convergence
            for method_args in [self.method_args] + self.fallback_args:
                stdout = self._call(cmd + method_args + self.extra_args)
                if "did not converge" not in stdout:
                    break
            if "Error:" in stdout or not os.path.exists(states_path):
                raise PrismError(_error_excerpt(stdout))

            variables, states = _parse_states(states_path)
            initial_values, state_values = _parse_sections(stdout, len(properties), len(states))
            choices = _parse_transitions(trans_path, len(states)) if export_transitions else []
            initial = _parse_initial_states(labels_path)

        if len(initial) != 1:
            raise PrismError(f"models must have exactly one initial state (found {len(initial)})")
        return PrismResult(variables, states, initial_values, state_values, choices,
                           initial_state=initial[0], stdout=stdout)

    @staticmethod
    def _write_inputs(tmp: str, model: str, properties: Sequence[str]) -> Tuple[str, str]:
        """Write the model and the properties file into `tmp`; returns their paths."""
        model_path, props_path = os.path.join(tmp, "model.prism"), os.path.join(tmp, "props.props")
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(model)
        with open(props_path, "w", encoding="utf-8") as f:
            f.write("\n".join(p.rstrip(";") + ";" for p in properties) + "\n")
        return model_path, props_path

    def _command(self, model_path: str, props_path: str, engine_args: Sequence[str]) -> List[str]:
        return [self.prism_path, model_path, props_path, *engine_args, "-javamaxmem", self.java_max_mem,
                "-maxiters", str(self.max_iters)]

    def _call(self, cmd: List[str]) -> str:
        """PRISM's output. On timeout, kills the whole process tree and raises `subprocess.TimeoutExpired`."""
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              start_new_session=os.name != "nt") as proc:
            try:
                return proc.communicate(timeout=self.timeout)[0]
            except subprocess.TimeoutExpired:
                _kill_tree(proc)
                proc.communicate()
                raise


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill `proc` and its children. Killing only the child leaves PRISM's JVM running (prism.bat on Windows
    starts it as a grandchild), and it keeps the output pipe open, so reading the output would block until it ends."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
