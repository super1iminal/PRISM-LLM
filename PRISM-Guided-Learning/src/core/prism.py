"""Run PRISM on an MDP and parse per-state results, reachable states and transitions."""
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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

    def __init__(self, prism_path: Optional[str] = None, extra_args: Sequence[str] = (),
                 java_max_mem: str = "4g", timeout: float = 900):
        self.prism_path = prism_path or get_prism_path()
        self.extra_args = list(extra_args)
        self.java_max_mem = java_max_mem
        self.timeout = timeout

    def run(self, model: str, properties: Sequence[str], export_transitions: bool = False) -> PrismResult:
        """Check `properties` (one per entry) on `model`.

        Properties wrapped in `filter(printall, ...)` also yield per-state values. The reachable
        states are always exported; transitions (with action labels) only if requested.
        """
        with tempfile.TemporaryDirectory(prefix="prism_") as tmp:
            model_path = os.path.join(tmp, "model.prism")
            props_path = os.path.join(tmp, "props.props")
            states_path = os.path.join(tmp, "states.sta")
            trans_path = os.path.join(tmp, "trans.tra")
            labels_path = os.path.join(tmp, "labels.lab")
            with open(model_path, "w", encoding="utf-8") as f:
                f.write(model)
            with open(props_path, "w", encoding="utf-8") as f:
                f.write("\n".join(p.rstrip(";") + ";" for p in properties) + "\n")

            cmd = [self.prism_path, model_path, props_path, "-explicit", "-javamaxmem", self.java_max_mem,
                   "-maxiters", "1000000", "-exportstates", states_path, "-exportlabels", labels_path]
            if export_transitions:
                cmd += ["-exporttrans", trans_path]
            cmd += self.extra_args
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  timeout=self.timeout)
            stdout = proc.stdout
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
