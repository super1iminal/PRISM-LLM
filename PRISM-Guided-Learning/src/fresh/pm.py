"""Pac-Man instances as explicit MDPs (exported once by PRISM) and exact worst-case evaluation of rule lists.

The policy only acts at Pac-Man's crossings (choices labelled right/up/left/down); a rule list restricts those
choices (every direction where no rule decides), every other choice stays open, and the worst case is Pmax of
reaching "Crash" over all remaining choices, computed by value iteration (the step counter makes the model
acyclic apart from self-loops at the end).
"""
import importlib
import os
import subprocess
import sys
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import jinja2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from core.rules import Constant, Feature, FeatureCall, SymbolicPolicy, Variable, Vocabulary  # noqa: E402
from fresh.vec import VecEval  # noqa: E402

_src = (ROOT / "domains" / "pacman" / "domain.py").read_text(encoding="utf-8")
_src = _src.replace("from core.domain import Domain, Instance", "Domain = object\nInstance = object")
_helpers: Dict = {}
exec(compile(_src, str(ROOT / "domains" / "pacman" / "domain.py"), "exec"), _helpers)
rule_vocabulary = _helpers["rule_vocabulary"]
CROSSINGS, START, GHOST_START, DIRECTIONS = (_helpers["CROSSINGS"], _helpers["START"], _helpers["GHOST_START"],
                                             _helpers["DIRECTIONS"])
ROWS = _helpers["ROWS"]

ACTIONS = ["right", "up", "left", "down"]
DATA = ROOT / "domains" / "pacman" / "data" / "pacman_transfer.csv"
CACHE = ROOT / "out" / "fresh" / "pm_cache"
VISIBLE = ["xP", "yP", "dP", "xG0", "yG0", "dG0", "xG1", "yG1", "dG1", "steps"]
_ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(str(ROOT / "domains" / "pacman")),
                          trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)


def export(max_steps: int) -> Path:
    d = CACHE / f"H{max_steps}"
    if (d / "t.tra").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    (d / "m.prism").write_text(_ENV.get_template("model.prism.j2").render(max_steps=max_steps), encoding="utf-8")
    (d / "p.props").write_text('Pmax=? [ F "Crash" ];\n', encoding="utf-8")
    cmd = ["prism.bat" if os.name == "nt" else "prism", str(d / "m.prism"), str(d / "p.props"), "-explicit",
           "-javamaxmem", "4g", "-exportstates", str(d / "s.sta"), "-exporttrans", str(d / "t.tra"),
           "-exportlabels", str(d / "l.lab")]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    if not (d / "t.tra").exists():
        raise RuntimeError(out[-2000:])
    return d


class PacInstance(VecEval):
    def __init__(self, idx: int, max_steps: int, threshold: float):
        self.id, self.H, self.threshold = idx, max_steps, threshold
        d = export(max_steps)
        with open(d / "s.sta") as f:
            names = f.readline().strip().strip("()").split(",")
            rows = [line.split(":", 1)[1].strip().strip("()").split(",") for line in f if line.strip()]
        vals = np.array(rows, dtype=np.int64)
        self.S = len(vals)
        self.state_vars = {n: vals[:, i] for i, n in enumerate(names)}
        tr = pd.read_csv(d / "t.tra", sep=" ", skiprows=1, header=None, names=["s", "c", "t", "p", "a"],
                         dtype={"s": np.int64, "c": np.int64, "t": np.int64, "p": float, "a": str})
        tr["a"] = tr["a"].fillna("")
        # one row per choice
        ch = tr.drop_duplicates(["s", "c"])[["s", "c", "a"]].reset_index(drop=True)
        key = tr["s"].to_numpy() * 64 + tr["c"].to_numpy()
        ckey = ch["s"].to_numpy() * 64 + ch["c"].to_numpy()
        order = np.argsort(ckey)
        self.tr_choice = order[np.searchsorted(ckey[order], key)]
        self.tr_target = tr["t"].to_numpy()
        self.tr_prob = tr["p"].to_numpy()
        self.ch_state = ch["s"].to_numpy()
        self.ch_action = np.array([ACTIONS.index(a) if a in ACTIONS else -1 for a in ch["a"]])
        with open(d / "l.lab") as f:
            header = f.readline()
            ids = {v: int(k) for k, v in (kv.split("=") for kv in header.split())}
            crash_id, init_id = ids['"Crash"'], ids['"init"']
            self.crash = np.zeros(self.S, bool)
            for line in f:
                s, labs = line.split(":")
                labs = [int(x) for x in labs.split()]
                if crash_id in labs:
                    self.crash[int(s)] = True
                if init_id in labs:
                    self.init = int(s)
        voc = rule_vocabulary(max_steps)
        consts = [Constant(c["name"], c["value"], c["description"]) for c in voc["rule_constants"]]
        feats = [Feature(f["name"], f["description"], expr=f.get("expr"), per_action=f.get("per_action"))
                 for f in voc["rule_features"]]
        self.vocabulary = Vocabulary(consts, feats, allow_any=True, general=True)
        ranges = {"xP": (1, 11), "yP": (1, 7), "dP": (0, 3), "xG0": (0, 11), "yG0": (0, 7), "dG0": (0, 3),
                  "xG1": (0, 11), "yG1": (0, 7), "dG1": (0, 3), "steps": (0, max_steps)}
        self.variables = [Variable(n, "int", *ranges[n]) for n in VISIBLE]
        self.decision = np.zeros(self.S, bool)
        self.decision[self.ch_state[self.ch_action >= 0]] = True

    @cached_property
    def scope(self):
        from core.rules import _Scope
        return _Scope.build(self.variables, ACTIONS, self.vocabulary)

    def policy(self, rules) -> SymbolicPolicy:
        return SymbolicPolicy.from_raw(self.variables, ACTIONS, rules, self.vocabulary)

    def allowed(self, rules) -> Tuple[np.ndarray, np.ndarray]:
        """(S x 4 allowed directions, deciding rule or -1), evaluated on every state."""
        pol = self.policy(rules)
        cache: Dict = {}
        allowed = np.zeros((self.S, 4), bool)
        decider = np.full(self.S, -1)
        undecided = np.ones(self.S, bool)
        for i, rule in enumerate(pol.rules):
            per = np.zeros((self.S, 4), bool)
            for a, e in rule.allows.items():
                per[:, ACTIONS.index(a)] = np.broadcast_to(self.veval(e, cache), (self.S,))
            fires = undecided & per.any(axis=1)
            allowed[fires] = per[fires]
            decider[fires] = i
            undecided &= ~fires
        allowed[undecided] = True
        return allowed, decider

    def choice_mask(self, allowed: np.ndarray) -> np.ndarray:
        m = np.ones(len(self.ch_state), bool)
        pol = self.ch_action >= 0
        m[pol] = allowed[self.ch_state[pol], self.ch_action[pol]]
        return m

    def values(self, mask: np.ndarray, maximize: bool = True) -> np.ndarray:
        """P{max,min} of eventually crashing, over the choices in `mask`."""
        v = self.crash.astype(float)
        live = ~self.crash
        neg = -1.0 if maximize else 2.0
        for _ in range(10 * self.H + 100):
            q = np.zeros(len(self.ch_state))
            np.add.at(q, self.tr_choice, self.tr_prob * v[self.tr_target])
            q[~mask] = neg
            nv = np.full(self.S, neg)
            (np.maximum if maximize else np.minimum).at(nv, self.ch_state, q)
            nv = np.where(live & (nv != neg), nv, v)
            if np.abs(nv - v).max() < 1e-13:
                return nv
            v = nv
        return v

    def worst(self, rules) -> float:
        allowed, _ = self.allowed(rules)
        return float(self.values(self.choice_mask(allowed))[self.init])

    def optimum(self) -> float:
        return float(self.values(np.ones(len(self.ch_state), bool), maximize=False)[self.init])


def load(ids: Sequence[int]) -> List[PacInstance]:
    df = pd.read_csv(DATA)
    return [PacInstance(i, int(df.loc[i, "max_steps"]), float(df.loc[i, "crash_threshold"])) for i in ids]


TRAIN = [3, 4, 5]
