"""Direct-solve re-check of every fallback check (item 3, Oct 21) on PRISM's own model: PRISM builds the composed model
of each check (as recertify.check_policy) and exports it (.tra/.lab at 17 digits); each requirement's deterministic
monitor (ltl_exact.Monitor) is multiplied in, and its worst case is computed by the graph precomputation (Prob0E for
a minimum, Prob0A for a maximum) and policy iteration with sparse direct solves, so no value comes from iterating to a
tolerance. Per-state policies (PPO) use the explicit model under the policy. Non-gridworld domains (Pac-Man): PRISM's
exact engine (exact_recheck.exact_check). Float64: round-off only; checks within NEAR of a threshold are flagged.

Usage (from PRISM-Guided-Learning/): python direct_recheck.py <fallback csv> <out csv> [--workers 6]"""
import argparse, json, re, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).parent))
import exact_recheck as E
from core.prism import PrismCall, PrismRunner, run_call
from core.verifier import PolicyVerifier
from ltl_exact import Monitor, labels_of, parse

NEAR = 1e-6
COLUMNS = ["file", "source", "target", "episode", "old_solver", "old_outcome", "old_margin", "states", "outcome",
           "margin", "binding", "values", "bound", "method", "seconds", "agrees", "near", "error"]
LOCK = threading.Lock()


class Explicit:
    """An MDP as CSR arrays: per state its choices (`cstart`), per choice its transitions (`tstart`, `tgt`, `prob`)."""

    def __init__(self, cstart, tstart, tgt, prob, labels, init):
        self.cstart, self.tstart, self.tgt, self.prob = cstart, tstart, tgt, prob
        self.labels, self.init = labels, int(init)
        self.S = len(cstart) - 1

    @classmethod
    def read(cls, tra: Path, lab: Path) -> "Explicit":
        lines = lab.read_text().splitlines()
        names = {int(k): v for k, v in re.findall(r'(\d+)="([^"]*)"', lines[0])}
        rows = np.loadtxt(tra, skiprows=1, usecols=(0, 1, 2, 3), ndmin=2)
        S = int(tra.read_text().split("\n", 1)[0].split()[0])
        s, c, t, p = rows[:, 0].astype(int), rows[:, 1].astype(int), rows[:, 2].astype(int), rows[:, 3]
        order = np.lexsort((t, c, s))
        s, c, t, p = s[order], c[order], t[order], p[order]
        new_choice = np.r_[True, (s[1:] != s[:-1]) | (c[1:] != c[:-1])]
        choice_id = np.cumsum(new_choice) - 1
        choice_state = s[new_choice]
        tstart = np.r_[np.flatnonzero(new_choice), len(s)]
        cstart = np.searchsorted(choice_state, np.arange(S + 1))
        labels = {n: np.zeros(S, dtype=bool) for n in names.values()}
        init = None
        for line in lines[1:]:
            st, _, ids = line.partition(":")
            for k in ids.split():
                labels[names[int(k)]][int(st)] = True
                if names[int(k)] == "init":
                    init = int(st)
        sums = np.add.reduceat(p, tstart[:-1])
        if np.abs(sums - 1).max() > 1e-12:
            raise ValueError("choice probabilities do not sum to 1")
        return cls(cstart, tstart, t, p, labels, init)

    @classmethod
    def from_model(cls, m) -> "Explicit":       # core.explicit.Model
        return cls(m.choice_start, m.trans_start, m.target, m.prob, m.labels, m.initial)


def worst_values(model: Explicit, reqs) -> dict:
    """Each requirement's worst case from the initial state, by its monitor product."""
    used = set().union(*(labels_of(parse(r.formula)) for r in reqs))
    letter_of = [frozenset(n for n in used if model.labels[n][s]) for s in range(model.S)]
    letters = sorted(set(letter_of), key=sorted)
    out = {}
    for r in reqs:
        if r.reward is not None:
            raise ValueError("reward requirement")
        mon = Monitor(r.formula, letters, letter_of[model.init])
        if mon.target is None:
            out[r.name] = (1.0 if mon.safety else 0.0, 0.0)
            continue
        # product: states (s, q); the monitor's target is absorbing
        index, states = {(model.init, 0): 0}, [(model.init, 0)]
        cst, tst, tg, pr = [0], [0], [], []
        k = 0
        while k < len(states):
            s, q = states[k]
            if q == mon.target:
                tg.append(k); pr.append(1.0); tst.append(len(tg)); cst.append(len(tst) - 1)
            else:
                for ch in range(model.cstart[s], model.cstart[s + 1]):
                    for t in range(model.tstart[ch], model.tstart[ch + 1]):
                        s2 = int(model.tgt[t])
                        nxt = (s2, mon.delta[q, letter_of[s2]])
                        if nxt not in index:
                            index[nxt] = len(states); states.append(nxt)
                        tg.append(index[nxt]); pr.append(model.prob[t])
                    tst.append(len(tg))
                cst.append(len(tst) - 1)
            k += 1
        goal = np.array([q == mon.target for _, q in states])
        # co-safe: worst = the least chance of reaching the target (lower bound) / most (upper bound);
        # safety: the target is the violation, worst = 1 - the most chance of it (lower bound on the value)
        maximize = (not r.maximize) if not mon.safety else r.maximize
        vals, bound = reach(np.array(cst), np.array(tst), np.array(tg), np.array(pr), goal, maximize)
        out[r.name] = (1.0 - vals[0] if mon.safety else vals[0], bound)
    return out


def reach(cstart, tstart, tgt, prob, goal, maximize):
    """P{max,min} of reaching `goal` (absorbing) in an MDP: graph precomputation, then policy iteration with direct
    sparse solves (as baselines/gridworld_check.GridChecker._reach, on CSR choices)."""
    S, C = len(cstart) - 1, len(tstart) - 1
    choice_state = np.repeat(np.arange(S), np.diff(cstart))
    t_choice = np.repeat(np.arange(C), np.diff(tstart))
    live = ~goal
    if maximize:        # Prob0A: no scheduler reaches the goal; start from an attractor (proper) strategy
        dist, d = np.where(goal, 0, -1), 0
        act = np.full(S, -1)
        while True:
            hits = np.zeros(C, dtype=bool)
            np.logical_or.at(hits, t_choice, dist[tgt] == d)
            cand = np.flatnonzero(hits & (dist[choice_state] < 0) & live[choice_state])
            if not cand.size:
                break
            d += 1
            st = choice_state[cand]
            first = np.unique(st, return_index=True)
            dist[first[0]] = d
            act[first[0]] = cand[first[1]]
        zero = live & (dist < 0)
    else:               # Prob0E: some scheduler avoids the goal forever
        Z = ~goal
        while True:
            stays = np.ones(C, dtype=bool)
            np.logical_and.at(stays, t_choice, Z[tgt])
            ok = np.zeros(S, dtype=bool)
            np.logical_or.at(ok, choice_state, stays)
            nz = Z & ok
            if (nz == Z).all():
                break
            Z = nz
        zero = live & Z
        act = np.full(S, -1)
    one = goal.copy()
    if (np.diff(cstart) == 1).all():   # a chain: probability 1 where no path leads to a state that misses the goal
        bad = zero.copy()
        frontier = np.flatnonzero(bad)
        T = sp.csr_matrix((np.ones(len(tgt)), (choice_state[t_choice], tgt)), shape=(S, S)).T.tocsr()
        while frontier.size:
            prev = np.unique(T[frontier].indices)
            prev = prev[~bad[prev] & ~goal[prev]]
            bad[prev] = True
            frontier = prev
        one = goal | (live & ~bad)
    solve = live & ~zero & ~one
    default = cstart[:-1]
    act = np.where(act >= 0, act, default)
    idx = np.flatnonzero(solve)
    v = one.astype(float)
    if not idx.size:
        return v, 0.0
    goal = one                          # states fixed at 1 count as reached
    pos = np.full(S, -1)
    pos[idx] = np.arange(idx.size)
    for _ in range(10000):
        ch = act[idx]
        lens = tstart[ch + 1] - tstart[ch]
        tr = np.concatenate([np.arange(tstart[c], tstart[c + 1]) for c in ch])
        rows = np.repeat(np.arange(idx.size), lens)
        cols = pos[tgt[tr]]
        inside = cols >= 0
        A = sp.identity(idx.size, format="csr") - sp.csr_matrix((prob[tr][inside], (rows[inside], cols[inside])),
                                                                 shape=(idx.size, idx.size))
        b = np.zeros(idx.size)
        np.add.at(b, rows[~inside], prob[tr][~inside] * goal[tgt[tr][~inside]])
        v = goal.astype(float)
        v[idx] = spsolve(A.tocsc(), b)
        q = np.add.reduceat(prob * v[tgt], tstart[:-1])                 # per choice
        cur = q[act]
        best = (np.maximum if maximize else np.minimum).reduceat(q, cstart[:-1])
        better = ((best > cur + 1e-12) if maximize else (best < cur - 1e-12)) & solve
        if not better.any():
            # a-posteriori bound: |x - x^| <= ||A^-1||_inf ||r||_inf, and for this M-matrix ||A^-1||_inf is the
            # largest expected time to leave the solved states (A^-1 1); unreliable when that time is huge
            Ac = A.tocsc()
            steps = spsolve(Ac, np.ones(idx.size))
            r = Ac @ v[idx] - b
            tmax = float(np.abs(steps).max())
            bound = np.inf if (tmax > 1e8 or not np.isfinite(tmax)) else 4 * tmax * (float(np.abs(r).max()) + 1e-15)
            return v, bound
        for s in np.flatnonzero(better):
            seg = q[cstart[s]:cstart[s + 1]]
            act[s] = cstart[s] + (int(np.argmax(seg)) if maximize else int(np.argmin(seg)))
    raise RuntimeError("policy iteration did not stop")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fallback"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    todo = pd.read_csv(a.fallback)
    out = Path(a.out)
    key = lambda f, s, t, e: (f, str(s), int(t), int(e) if pd.notna(e) else -1)
    if out.exists():                               # resume: keep the decided rows, redo the rest
        prev = pd.read_csv(out)
        prev = prev[prev.outcome.isin(["certified", "failed"])]
        prev.reindex(columns=COLUMNS).to_csv(out, index=False)
        done = {key(*r) for r in zip(prev.file, prev.source, prev.target, prev.episode)}
        todo = todo[[key(*r) not in done for r in zip(todo.file, todo.source, todo.target, todo.episode)]]
    contexts, locks = {}, {f: threading.Lock() for f in todo.file.unique()}

    def ctx(file):
        with locks[file]:
            if file not in contexts:
                contexts[file] = E.context(file)
            return contexts[file]

    def one(row):
        res = dict.fromkeys(COLUMNS, "")
        res.update(file=row.file, source=str(row.source), target=int(row.target), episode=row.episode,
                   old_solver=row.solver, old_outcome=row.outcome, old_margin=row.margin, states=row.states)
        start = time.time()
        try:
            domain, cfg, inst, controller = ctx(row.file)
            instance = inst[int(row.target)]
            if isinstance(controller, tuple):          # per-state policy on the explicit model
                from core.explicit import ExplicitMDP
                mdp = ExplicitMDP(domain, instance, cfg)
                spec = mdp.spec
                m = mdp.under(controller[1].policies(mdp)[str(row.source)])
                values = worst_values(Explicit.from_model(m.part(m.reachable())), spec.requirements)
                method = "explicit model under the policy; monitor product, direct solves"
            elif cfg.domain.name != "gridworld":       # Pac-Man: PRISM's exact engine
                spec, vals, _ = E.exact_check(domain, cfg, instance, controller(int(row.target))[str(row.source)],
                                              7200)
                values, method = {k: float(v) for k, v in vals.items()}, "PRISM exact engine"
            else:
                runner = PrismRunner(cfg.prism)
                v = PolicyVerifier(domain, instance, cfg, runner)
                spec = v.spec
                policy = spec.policy(controller(int(row.target))[str(row.source)], spec.vocabulary(cfg.rules))
                with tempfile.TemporaryDirectory(prefix="direct_") as tmp:
                    model = Path(tmp) / "m.prism"
                    model.write_text(v.compose(policy), encoding="utf-8")
                    log = run_call(PrismCall((runner.prism_path, str(model), "-exportmodel",
                                              str(Path(tmp) / "out.tra,lab"), "-exportmodelprecision", "17"),
                                             1800, cfg.prism.max_concurrent))
                    if not (Path(tmp) / "out.tra").exists():
                        raise RuntimeError("export failed: " + log[-300:])
                    values = worst_values(Explicit.read(Path(tmp) / "out.tra", Path(tmp) / "out.lab"),
                                          spec.requirements)
                method = "PRISM's model; monitor product, direct solves"
            bounds = {k: (v[1] if isinstance(v, tuple) else 0.0) for k, v in values.items()}
            values = {k: float(v[0] if isinstance(v, tuple) else v) for k, v in values.items()}
            m = {r.name: r.margin(values[r.name]) for r in spec.requirements}
            failed = [k for k in m if m[k] < -bounds[k]]
            decided = bool(failed) or all(m[k] > bounds[k] for k in m)
            binding = min(m, key=m.get)
            # undecided: the bound does not separate a value from its threshold (an ill-conditioned chain)
            outcome = ("failed" if failed else "certified") if decided else "undecided"
            if failed:
                binding = min(failed, key=m.get)
            res.update(outcome=outcome, margin=m[binding], binding=binding, values=json.dumps(values), method=method,
                       agrees=outcome == row.outcome, near=abs(m[binding]) < NEAR,
                       bound=max(bounds.values()))
        except Exception as e:
            res.update(outcome="error", error=f"{type(e).__name__}: {str(e)[:300]}")
        res["seconds"] = time.time() - start
        with LOCK:
            pd.DataFrame([res], columns=COLUMNS).to_csv(out, mode="a", header=not out.exists(), index=False)
            if res["outcome"] == "error" or not res["agrees"] or res["near"]:
                print(f"  {row.file} {res['source']} on {res['target']}: was {res['old_outcome']} {row.margin}, now "
                      f"{res['outcome']} {res['margin']} near={res['near']} {res['error']}", flush=True)
        return res

    print(f"{len(todo)} checks", flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(one, todo.itertuples(index=False)))
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
