"""Exact, vectorized gridworld evaluation of rule lists (the same MDP, features and requirements as PRISM's).

A member's state is (x, y, seg, obs_idx), where seg = number of goals reached in order (0..3) encodes the flags
g1, g2, g3. A rule list gives each state the set of actions its deciding rule allows (every action if no rule
decides it); `Member.check` returns each requirement's worst-case (adversarial over those sets) and best-case
probability, computed by value iteration on the reach / avoid form of each requirement:

    goalk                F at_goalk                               reach {at goal k}
    seq_1_before_2       !at_goal2 U at_goal1                     reach {g1} avoiding {at goal 2 & !g1}
    seq_2_before_3       !at_goal3 U (at_goal1 & (!at_goal3 U at_goal2))   reach {g2} avoiding {at goal 3 & !g2}
    complete_sequence    both of the above                        reach {g2} avoiding both sets
    avoid_moving_segk    G (in_segk => !at_moving_obs)            1 - reach {in_segk & at_moving_obs}
"""
import ast
import warnings
import sys
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # PRISM-Guided-Learning/
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "domains" / "gridworld"))
sys.setrecursionlimit(20000)

from fresh.vec import VecEval  # noqa: E402
from core.rules import (ANY, ActionRef, Binary, Const, Constant, Feature, FeatureCall, RuleError,  # noqa: E402
                        SymbolicPolicy, Unary, Var, Variable, Vocabulary)

# domain.py imports core.domain (rule-language's); only its pure helpers are needed here
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("gw_domain_helpers", ROOT / "domains" / "gridworld" / "domain.py")
_src = (ROOT / "domains" / "gridworld" / "domain.py").read_text(encoding="utf-8")
_src = _src.replace("from core.domain import Domain, Instance", "Domain = object\nInstance = object")
_helpers: Dict = {}
exec(compile(_src, str(ROOT / "domains" / "gridworld" / "domain.py"), "exec"), _helpers)
rule_vocabulary = _helpers["rule_vocabulary"]
expand_cycle = _helpers["expand_cycle"]
MOVES = _helpers["MOVES"]
SLIP_LEFT, SLIP_RIGHT = _helpers["SLIP_LEFT"], _helpers["SLIP_RIGHT"]
P_FWD, P_SLIP = _helpers["PROB_FORWARD"], _helpers["PROB_SLIP_LEFT"]
GOAL_T, SEQ_T, AVOID_T = _helpers["GOAL_THRESHOLD"], _helpers["SEQUENCE_THRESHOLD"], _helpers["AVOID_THRESHOLD"]

ACTIONS = ["up", "right", "down", "left"]
HIDDEN = {"hazard"}
DATA = ROOT / "domains" / "gridworld" / "data" / "families_v0.csv"
REQS = ["goal1", "goal2", "goal3", "seq_1_before_2", "seq_2_before_3", "complete_sequence",
        "avoid_moving_seg1", "avoid_moving_seg2", "avoid_moving_seg3"]
THRESH = {r: (AVOID_T if r.startswith("avoid") else GOAL_T if r.startswith("goal") else SEQ_T) for r in REQS}


def _parse(v, default):
    if isinstance(v, str):
        return ast.literal_eval(v)
    return default if v is None or (isinstance(v, float) and pd.isna(v)) else v


@dataclass
class Result:
    worst: Dict[str, float]
    best: Dict[str, float]
    uncovered: int = 0          # reachable (under the worst case's union of allowed actions) states no rule decides
    any_states: int = 0         # reachable states where the deciding rule leaves several actions open

    @property
    def certified(self) -> bool:
        return all(self.worst[r] >= THRESH[r] - 1e-9 for r in REQS)

    @property
    def margin(self) -> float:
        return min(self.worst[r] - THRESH[r] for r in REQS)

    def failing(self) -> List[str]:
        return [r for r in REQS if self.worst[r] < THRESH[r] - 1e-9]


class Member(VecEval):
    """One gridworld instance with its vocabulary, states and transition table."""

    def __init__(self, idx: int, row: pd.Series, hidden=None):
        self.id = idx
        self.family = row.get("family", "")
        self.n = n = int(row["n"])
        goals = _parse(row["goals"], {})
        self.goals = {int(k): tuple(v) for k, v in goals.items()}
        self.ks = sorted(self.goals)
        self.static = {tuple(c) for c in _parse(row["static"], [])}
        self.cycle = [tuple(c) for c in expand_cycle(_parse(row.get("moving"), []))]
        self.L = max(1, len(self.cycle))
        self.grid_text = str(row.get("grid", "")).replace("\\n", "\n")
        goal_list = [{"k": k, "row": self.goals[k][0], "col": self.goals[k][1]} for k in self.ks]
        voc = rule_vocabulary(n, goal_list, sorted(self.static), self.cycle, phase_visible=True)
        self.constants = [Constant(c["name"], c["value"], c["description"]) for c in voc["rule_constants"]]
        self.features = [Feature(f["name"], f["description"], per_action=f["per_action"])
                         for f in voc["rule_features"] if f["name"] not in (HIDDEN if hidden is None else set(hidden))]
        self.vocabulary = Vocabulary(self.constants, self.features, allow_any=True, general=True)
        self.variables = [Variable("x", "int", 0, n - 1, "agent row (0 = top)"),
                          Variable("y", "int", 0, n - 1, "agent column (0 = left)")]
        self.variables += [Variable(f"g{k}", "bool", description=f"goal {k} reached") for k in self.ks]
        if self.cycle:
            self.variables.append(Variable("obs_idx", "int", 0, self.L - 1, "moving obstacle's position in its cycle"))
        self._build_states()

    # ------------------------------------------------------------ states and transitions

    def _build_states(self):
        n, L, K = self.n, self.L, len(self.ks)
        x, y, seg, ob = np.meshgrid(np.arange(n), np.arange(n), np.arange(K + 1), np.arange(L), indexing="ij")
        self.sx, self.sy, self.sseg, self.sob = x.ravel(), y.ravel(), seg.ravel(), ob.ravel()
        self.S = len(self.sx)
        self.index = lambda xx, yy, ss, oo: ((xx * n + yy) * (K + 1) + ss) * L + oo
        self.state_vars = {"x": self.sx, "y": self.sy, "obs_idx": self.sob}
        for i, k in enumerate(self.ks):
            self.state_vars[f"g{k}"] = self.sseg > i
        # successor of each state under each direction (the PRISM model's outcome(d))
        succ_dir = {}
        for d, (dr, dc) in MOVES.items():
            r = np.clip(self.sx + dr, 0, n - 1)
            c = np.clip(self.sy + dc, 0, n - 1)
            blocked = np.zeros(self.S, bool)
            for (br, bc) in self.static:
                blocked |= (r == br) & (c == bc)
            nx = np.where(blocked, self.sx, r)
            ny = np.where(blocked, self.sy, c)
            nseg = self.sseg.copy()
            # g_k' = g_k | (g_{k-1} & hits_k): with seg = goals reached in order, seg advances by one at most
            for i, k in enumerate(self.ks):
                gr, gc = self.goals[k]
                hits = ~blocked & (r == gr) & (c == gc)
                nseg = np.where((self.sseg == i) & hits, i + 1, nseg)
            nob = (self.sob + 1) % L
            succ_dir[d] = self.index(nx, ny, nseg, nob)
        self.succ = np.stack([np.stack([succ_dir[a], succ_dir[SLIP_LEFT[a]], succ_dir[SLIP_RIGHT[a]]], axis=1)
                              for a in ACTIONS], axis=1)      # S x 4 x 3
        self.probs = np.array([P_FWD, P_SLIP, P_SLIP])
        self.init = int(self.index(0, 0, 0, 0))
        self.inits = np.array([self.init])
        at = {k: (self.sx == self.goals[k][0]) & (self.sy == self.goals[k][1]) for k in self.ks}
        on_obs = np.zeros(self.S, bool)
        for i, (r, c) in enumerate(self.cycle):
            on_obs |= (self.sob == i) & (self.sx == r) & (self.sy == c)
        g = {k: self.state_vars[f"g{k}"] for k in self.ks}
        k1, k2, k3 = self.ks
        self.req_sets = {
            "goal1": (at[k1], None), "goal2": (at[k2], None), "goal3": (at[k3], None),
            "seq_1_before_2": (g[k1], at[k2] & ~g[k1]),
            "seq_2_before_3": (g[k2], at[k3] & ~g[k2]),
            "complete_sequence": (g[k2], (at[k2] & ~g[k1]) | (at[k3] & ~g[k2])),
        }
        segs = {k1: ~g[k1], k2: g[k1] & ~g[k2], k3: g[k2] & ~g[k3]}
        self.bad = {f"avoid_moving_seg{k}": segs[k] & on_obs for k in self.ks}
        self.on_obs = on_obs
        self.at = at

    # ------------------------------------------------------------ vectorized rule-language evaluation

    @cached_property
    def scope(self):
        from core.rules import _Scope
        return _Scope.build(self.variables, ACTIONS, self.vocabulary)

    @cached_property
    def feature_values(self) -> Dict[Tuple[str, str], np.ndarray]:
        cache: Dict = {}
        out = {}
        for name, per in self.scope.action_features.items():
            for a, e in per.items():
                out[(name, a)] = np.broadcast_to(self.veval(FeatureCall(name, a, e), cache), (self.S,)).astype(bool)
        return out

    def policy(self, rules: Sequence[Tuple[str, str]]) -> SymbolicPolicy:
        return SymbolicPolicy.from_raw(self.variables, ACTIONS, rules, self.vocabulary)

    def allowed(self, rules: Sequence[Tuple[str, str]], order: Optional[Sequence[str]] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
        """(S x 4 bool allowed actions, S int deciding rule or -1). With `order`, an `any` rule takes the first
        action in `order` that it allows (the verifier resolves it) instead of leaving them all open."""
        pol = self.policy(rules)
        cache = {}
        for (name, a), v in self.feature_values.items():
            cache[(name, a)] = v
        allowed = np.zeros((self.S, 4), bool)
        decider = np.full(self.S, -1)
        undecided = np.ones(self.S, bool)
        for i, rule in enumerate(pol.rules):
            per = np.zeros((self.S, 4), bool)
            for a, e in rule.allows.items():
                per[:, ACTIONS.index(a)] = np.broadcast_to(self.veval(e, cache), (self.S,))
            if order is not None and rule.action == ANY:
                first = np.zeros_like(per)
                taken = np.zeros(self.S, bool)
                for a in order:
                    j = ACTIONS.index(a)
                    first[:, j] = per[:, j] & ~taken
                    taken |= per[:, j]
                per = first
            fires = undecided & per.any(axis=1)
            allowed[fires] = per[fires]
            decider[fires] = i
            undecided &= ~fires
        allowed[undecided] = True
        return allowed, decider

    # ------------------------------------------------------------ model checking

    def _reach(self, allowed: np.ndarray, target: np.ndarray, avoid: Optional[np.ndarray], maximize: bool,
               return_policy: bool = False):
        """P{min,max} of reaching `target` before `avoid` (both absorbing), over the allowed actions, by policy
        iteration after the graph precomputations (Prob0A for max, Prob0E for min), so every linear system solved
        is nonsingular: min removes every end component, and max starts from a proper attractor strategy and
        switches actions only on strict improvement, which keeps it proper."""
        from scipy.sparse import csr_matrix, identity
        from scipy.sparse.linalg import spsolve
        succ, p, S = self.succ, self.probs, self.S
        stop = target | (avoid if avoid is not None else np.zeros(S, bool))
        live = ~stop
        if maximize:
            # Prob0A: states that cannot reach the target under any choice; attractor distance for the start policy
            dist = np.where(target, 0, -1)
            d = 0
            while True:
                hit = (dist[succ] == d).any(axis=2) & allowed           # S x 4: action may reach layer d
                new = live & (dist < 0) & hit.any(axis=1)
                if not new.any():
                    break
                d += 1
                dist[new] = d
            zero = live & (dist < 0)
            hit_prev = (dist[succ] == (dist - 1)[:, None, None]).any(axis=2) & allowed
            act = np.where(hit_prev.any(axis=1), np.argmax(hit_prev, axis=1), np.argmax(allowed, axis=1))
        else:
            # Prob0E: the adversary can avoid the target forever with probability 1
            Z = ~target
            while True:
                ok = (Z[succ].all(axis=2) & allowed).any(axis=1)
                nz = Z & (~live | ok)
                if (nz == Z).all():
                    break
                Z = nz
            zero = live & Z
            act = np.argmax(allowed, axis=1)
        solve = live & ~zero
        idx = np.nonzero(solve)[0]
        v = target.astype(float)
        if len(idx) == 0:
            return (v, act) if return_policy else v
        pos = np.full(S, -1)
        pos[idx] = np.arange(len(idx))
        for _ in range(1000):
            nxt = succ[idx, act[idx]]                                    # m x 3
            rows = np.repeat(np.arange(len(idx)), 3)
            cols = pos[nxt.ravel()]
            pr = np.tile(p, len(idx))
            inside = cols >= 0
            A = identity(len(idx), format="csr") - csr_matrix((pr[inside], (rows[inside], cols[inside])),
                                                              shape=(len(idx), len(idx)))
            b = np.zeros(len(idx))
            np.add.at(b, rows[~inside], pr[~inside] * target[nxt.ravel()[~inside]])
            v = target.astype(float)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                try:
                    v[idx] = spsolve(A.tocsc(), b)
                except Exception:      # singular (numerical ties in the improvement step): value iteration instead
                    return self._vi(allowed, target, stop, maximize, act, return_policy)
            if not np.isfinite(v).all():
                return self._vi(allowed, target, stop, maximize, act, return_policy)
            q = (v[succ] * p).sum(axis=2)
            q = np.where(allowed, q, -np.inf if maximize else np.inf)
            cur = q[np.arange(S), act]
            better = (q.max(axis=1) > cur + 1e-12) if maximize else (q.min(axis=1) < cur - 1e-12)
            better &= solve
            if not better.any():
                break
            act = np.where(better, q.argmax(axis=1) if maximize else q.argmin(axis=1), act)
        return (v, act) if return_policy else v

    def _vi(self, allowed, target, stop, maximize, act, return_policy, tol: float = 1e-13, max_iter: int = 200000):
        """Value iteration from below (the fallback when a policy's linear system is singular)."""
        succ, p = self.succ, self.probs
        v = target.astype(float)
        neg = -np.inf if maximize else np.inf
        live = ~stop
        for _ in range(max_iter):
            q = (v[succ] * p).sum(axis=2)
            q = np.where(allowed, q, neg)
            nv = np.where(live, q.max(axis=1) if maximize else q.min(axis=1), v)
            if np.abs(nv - v).max() < tol:
                v = nv
                break
            v = nv
        if return_policy:
            q = np.where(allowed, (v[succ] * p).sum(axis=2), neg)
            return v, (q.argmax(axis=1) if maximize else q.argmin(axis=1))
        return v

    def check(self, rules: Sequence[Tuple[str, str]], best: bool = True,
              allowed: Optional[np.ndarray] = None, order: Optional[Sequence[str]] = None) -> Result:
        """Worst (and best) case of every requirement; with several initial states (`self.inits`), the smallest
        value over them."""
        if allowed is None:
            allowed, decider = self.allowed(rules, order)
        else:
            decider = None
        inits = self.inits
        worst, bestv = {}, {}
        self.worst_init = {}
        for r in REQS:
            if r in self.req_sets:
                t, av = self.req_sets[r]
                w = self._reach(allowed, t, av, maximize=False)[inits]
                b = self._reach(allowed, t, av, maximize=True)[inits] if best else None
            else:
                bad = self.bad[r]
                w = 1.0 - self._reach(allowed, bad, None, maximize=True)[inits]
                b = 1.0 - self._reach(allowed, bad, None, maximize=False)[inits] if best else None
            worst[r] = float(w.min())
            self.worst_init[r] = int(inits[int(w.argmin())])
            if best:
                bestv[r] = float(b.min())
        res = Result(worst, bestv)
        if decider is not None:
            reach = self.reachable(allowed)
            res.uncovered = int((reach & (decider < 0)).sum())
            res.any_states = int((reach & (decider >= 0) & (allowed.sum(axis=1) > 1)).sum())
        return res

    def reachable(self, allowed: np.ndarray) -> np.ndarray:
        seen = np.zeros(self.S, bool)
        seen[self.inits] = True
        frontier = seen.copy()
        while frontier.any():
            nxt = np.zeros(self.S, bool)
            idx = np.nonzero(frontier)[0]
            s = self.succ[idx][allowed[idx]]
            nxt[s.ravel()] = True
            frontier = nxt & ~seen
            seen |= nxt
        return seen

    def all_phases(self) -> "Member":
        """Check from the start cell with the moving obstacle at every phase (not only obs_idx = 0)."""
        self.inits = np.array([int(self.index(0, 0, 0, ob)) for ob in range(self.L)])
        return self

    def optimum(self) -> Dict[str, float]:
        """Best value of each requirement on the bare MDP (each optimized separately)."""
        return self.check([], best=True).best


def load(ids: Sequence[int], data=DATA, hidden=None) -> List[Member]:
    data = Path(data)
    if not data.exists():
        data = DATA.parent / data
    df = pd.read_csv(data)
    return [Member(i, df.loc[i], hidden) for i in ids]


FAMILIES = {"rooms": [0, 4, 8], "crossing": [60, 64, 68], "adjacent": [120, 124, 128]}
