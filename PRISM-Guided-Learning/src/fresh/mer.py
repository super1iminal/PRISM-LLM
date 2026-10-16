"""Mars Exploration Rovers instances as explicit MDPs (exported once by PRISM) and exact worst-case evaluation of
rule lists. Every step is a policy action (29 of them; one that is not possible leaves the state unchanged); a rule
list restricts the actions per state (every action where no rule decides), and the worst case of the requirement
P>=threshold [ F "err" ] is Pmin over the remaining choices."""
import os
import subprocess
import sys
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import jinja2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from core.rules import Constant, Feature, SymbolicPolicy, Variable, Vocabulary  # noqa: E402
from fresh.vec import VecEval  # noqa: E402

_src = (ROOT / "domains" / "mer" / "domain.py").read_text(encoding="utf-8")
_src = _src.replace("from core.domain import Domain, Instance", "Domain = object\nInstance = object")
_helpers: Dict = {}
exec(compile(_src, str(ROOT / "domains" / "mer" / "domain.py"), "exec"), _helpers)
rule_vocabulary, CODES = _helpers["rule_vocabulary"], _helpers["CODES"]

ACTIONS = ["arb_reclaim"] + [f"u{i}_{a}" for i in (1, 2) for a in (
    "want_comm", "want_drive", "ready", "request_comm", "request_drive", "grant_comm", "grant_drive", "deny_comm",
    "deny_drive", "use", "cancel_comm", "cancel_drive", "rescind_drive")]
DATA = ROOT / "domains" / "mer" / "data"
CACHE = ROOT / "out" / "fresh" / "mer_cache"
_ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(str(ROOT / "domains" / "mer")),
                          trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
BOOLS = {"gc", "dc", "gd", "dd", "rescind1", "rescind2"}


def ranges(n: int) -> Dict[str, Tuple[int, int]]:
    return {"s": (-1, 2), "r": (0, 2), "k": (0, n), "commUser": (0, 2), "driveUser": (0, 2), "s1": (0, 4),
            "r1": (0, 2), "s2": (0, 4), "r2": (0, 2), "e": (0, 3), "u": (0, 2)}


def export(n: int, x: float) -> Path:
    d = CACHE / f"n{n}_x{x}"
    if (d / "t.tra").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    (d / "m.prism").write_text(_ENV.get_template("model.prism.j2").render(n=n, x=x), encoding="utf-8")
    (d / "p.props").write_text('Pmax=? [ F "err" ];\n', encoding="utf-8")
    cmd = ["prism.bat" if os.name == "nt" else "prism", str(d / "m.prism"), str(d / "p.props"), "-explicit",
           "-javamaxmem", "4g", "-exportstates", str(d / "s.sta"), "-exporttrans", str(d / "t.tra"),
           "-exportlabels", str(d / "l.lab")]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    if not (d / "t.tra").exists():
        raise RuntimeError(out[-2000:])
    return d


class MerInstance(VecEval):
    def __init__(self, idx: int, name: str, n: int, x: float, threshold: float, optimum: float):
        self.id, self.name, self.n, self.x, self.threshold, self.optimum_value = idx, name, n, x, threshold, optimum
        d = export(n, x)
        with open(d / "s.sta") as f:
            names = f.readline().strip().strip("()").split(",")
            rows = [line.split(":", 1)[1].strip().strip("()").split(",") for line in f if line.strip()]
        vals = np.array([[1 if v == "true" else 0 if v == "false" else int(v) for v in r] for r in rows],
                        dtype=np.int64)
        self.S = len(vals)
        self.state_vars = {nm: (vals[:, i].astype(bool) if nm in BOOLS else vals[:, i]) for i, nm in enumerate(names)}
        tr = pd.read_csv(d / "t.tra", sep=" ", skiprows=1, header=None, names=["s", "c", "t", "p", "a"],
                         dtype={"s": np.int64, "c": np.int64, "t": np.int64, "p": float, "a": str})
        tr["a"] = tr["a"].fillna("")
        ch = tr.drop_duplicates(["s", "c"])[["s", "c", "a"]].reset_index(drop=True)
        key = tr["s"].to_numpy() * 256 + tr["c"].to_numpy()
        ckey = ch["s"].to_numpy() * 256 + ch["c"].to_numpy()
        order = np.argsort(ckey)
        self.tr_choice = order[np.searchsorted(ckey[order], key)]
        self.tr_target, self.tr_prob = tr["t"].to_numpy(), tr["p"].to_numpy()
        self.ch_state = ch["s"].to_numpy()
        self.ch_action = np.array([ACTIONS.index(a) if a in ACTIONS else -1 for a in ch["a"]])
        with open(d / "l.lab") as f:
            ids = {v: int(k) for k, v in (kv.split("=") for kv in f.readline().split())}
            err_id, init_id = ids['"err"'], ids['"init"']
            self.err = np.zeros(self.S, bool)
            for line in f:
                s, labs = line.split(":")
                labs = [int(v) for v in labs.split()]
                if err_id in labs:
                    self.err[int(s)] = True
                if init_id in labs:
                    self.init = int(s)
        voc = rule_vocabulary(n, x, ROOT / "domains" / "mer")
        self.vocabulary = Vocabulary(
            [Constant(c["name"], c["value"], c["description"]) for c in voc["rule_constants"]],
            [Feature(f["name"], f["description"], expr=f.get("expr"), per_action=f.get("per_action"))
             for f in voc["rule_features"]], allow_any=True, general=True)
        rg = ranges(n)
        self.variables = [Variable(nm, "bool") if nm in BOOLS else Variable(nm, "int", *rg[nm]) for nm in names]

    @cached_property
    def scope(self):
        from core.rules import _Scope
        return _Scope.build(self.variables, ACTIONS, self.vocabulary)

    def policy(self, rules) -> SymbolicPolicy:
        return SymbolicPolicy.from_raw(self.variables, ACTIONS, rules, self.vocabulary)

    def allowed(self, rules) -> Tuple[np.ndarray, np.ndarray]:
        pol = self.policy(rules)
        cache: Dict = {}
        A = len(ACTIONS)
        allowed = np.zeros((self.S, A), bool)
        decider = np.full(self.S, -1)
        undecided = np.ones(self.S, bool)
        for i, rule in enumerate(pol.rules):
            per = np.zeros((self.S, A), bool)
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

    def values(self, mask: np.ndarray, maximize: bool = False, tol: float = 1e-12) -> np.ndarray:
        """P{min,max} of reaching "err" over the choices in `mask` (value iteration from below; for min, the
        states where the adversary can avoid "err" forever are set to 0 first)."""
        S = self.S
        live = ~self.err
        if not maximize:
            # Prob0E: greatest set Z of non-err states with some allowed choice whose successors all stay in Z
            Z = live.copy()
            succ_in = np.zeros(len(self.tr_choice), bool)
            while True:
                succ_in = Z[self.tr_target]
                bad = np.zeros(len(self.ch_state), bool)
                np.logical_or.at(bad, self.tr_choice, ~succ_in)
                ok = mask & ~bad
                has = np.zeros(S, bool)
                np.logical_or.at(has, self.ch_state, ok)
                nz = Z & has
                if (nz == Z).all():
                    break
                Z = nz
            live = live & ~Z
        v = self.err.astype(float)
        neg = -1.0 if maximize else 2.0
        for _ in range(1000000):
            q = np.zeros(len(self.ch_state))
            np.add.at(q, self.tr_choice, self.tr_prob * v[self.tr_target])
            q[~mask] = neg
            nv = np.full(S, neg)
            (np.maximum if maximize else np.minimum).at(nv, self.ch_state, q)
            nv = np.where(live & (nv != neg), nv, v)
            if np.abs(nv - v).max() < tol:
                return nv
            v = nv
        return v

    def worst(self, rules) -> float:
        allowed, _ = self.allowed(rules)
        return float(self.values(self.choice_mask(allowed))[self.init])

    def optimum(self) -> float:
        return float(self.values(np.ones(len(self.ch_state), bool), maximize=True)[self.init])


def load(ids: Sequence[int], dataset: str = "mer_transfer.csv") -> List[MerInstance]:
    df = pd.read_csv(DATA / dataset)
    return [MerInstance(i, str(df.loc[i, "name"]), int(df.loc[i, "n"]), float(df.loc[i, "x"]),
                        float(df.loc[i, "err_threshold"]), float(df.loc[i, "optimum"])) for i in ids]
