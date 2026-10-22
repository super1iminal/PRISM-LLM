"""Joint solvability of a UUV instance: can one controller meet all three requirements at once?

PRISM's multi-objective query cannot decide UUV's requirements as written: LP rejects the step bound of
`done_in_time` (`F<=deadline`), state rewards and `R[F]`, and on the copy that removes those obstacles PRISM's LP
did not finish in 25+ minutes (Rajan's notes, item 8). This checks the same question with the standard
multi-objective LP solved by HiGHS (scipy), on a copy of the instance's model that PRISM builds and exports:

- a counter `t` (+1 on every transition, saturating at deadline + 1): `F<=D "done"` = `F ("done" & t<=D)`;
- a flag `ok`, false once a thruster-failure state has been left: since "done" is reached with probability 1 under
  every scheduler and no failure happens after it, `G !"thruster_failure"` = `F ("done" & ok)`;
- the energy reward kept as a state reward and charged, in the LP, on each choice taken from its state.

With "done" reached almost surely under every scheduler (checked: PRISM's Pmin of `F "done"` must be 1), the
expected number of times each (state, choice) is taken is finite, and the schedulers' achievable vectors of
(P safe, P on time, E energy) are exactly the feasible points of the LP over those visit counts y >= 0:
    sum_c y(s,c) - sum_{s',c'} P(s',c',s) y(s',c') = [s = initial]      for every state s not in "done"
    P safe = sum y(s,c) P(s,c, done & ok),  P on time = sum y(s,c) P(s,c, done & t<=D),  E = sum y(s,c) r(s,c)
The instance is jointly solvable iff max P on time s.t. P safe >= safe_threshold and E <= energy_threshold reaches
on_time_threshold. The check validates itself: the LP's single-objective optima must equal PRISM's
(`validate=True`), and both copies' equivalence is in out/paper/rajan/item8/.

Usage (from PRISM-Guided-Learning/): python src/uuv_joint.py --data uuv_paper.csv [--ids 0 1] [--validate]
"""
import argparse
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog

from config import load_config
from core.domain import load_domain
from core.verifier import PolicyVerifier

LABELS = ("step", "low", "med", "high")
FAILURE = "(s=recover_high | s=recover_med | s=recover_low | s=recover_following)"


def joint_copy(model: str, deadline: int) -> str:
    """The instance's model with the step counter `t` and the flag `ok`, and the energy reward as its only reward
    structure (the "time" structure dropped, so PRISM exports energy alone)."""
    m = re.sub(r'rewards "time".*?endrewards\n?', "", model, flags=re.S)
    counter = (f"\nmodule stepcount\n  t : [0..{deadline + 1}] init 0;\n  ok : bool init true;\n"
               + "".join(f"  [{a}] true -> (t'=min(t+1, {deadline + 1})) & (ok'=ok & !{FAILURE});\n" for a in LABELS)
               + "endmodule\n\n")
    i = m.index('label "done"')
    return m[:i] + counter + m[i:]


def export(prism: str, model: str, heap: str, timeout: float) -> Dict[str, object]:
    """Build the copy with PRISM's explicit engine and read states, transitions, transition rewards and labels."""
    with tempfile.TemporaryDirectory(prefix="uuvjoint_") as tmp:
        tmp = Path(tmp)
        (tmp / "m.prism").write_text(model, encoding="utf-8")
        (tmp / "p.props").write_text('Pmin=? [ F "done" ];\n', encoding="utf-8")
        cmd = [prism, str(tmp / "m.prism"), str(tmp / "p.props"), "-explicit", "-javamaxmem", heap,
               "-exportstates", str(tmp / "m.sta"), "-exporttrans", str(tmp / "m.tra"),
               "-exportstaterewards", str(tmp / "m.srew"), "-exportlabels", str(tmp / "m.lab")]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
        if "Error" in out:
            raise RuntimeError(out[-800:])
        pmin_done = float(re.search(r"^Result: (\S+)", out, re.M).group(1))
        header = (tmp / "m.sta").read_text().splitlines()
        names = header[0].strip("()").split(",")
        vals = np.array([[_val(v) for v in l.split(":", 1)[1].strip("()").split(",")] for l in header[1:]])
        tra = pd.read_csv(tmp / "m.tra", sep=" ", skiprows=1, header=None, usecols=[0, 1, 2, 3]).to_numpy()
        srew = np.zeros(len(vals))
        rows = [l for l in (tmp / "m.srew").read_text().splitlines() if l.strip() and not l.startswith("#")]
        for line in rows[1:]:                           # rows[0]: the header (states, non-zero entries)
            i, w = line.split()
            srew[int(i)] = float(w)
        lab_lines = (tmp / "m.lab").read_text().splitlines()
    label_ids = {name: int(i) for i, name in re.findall(r'(\d+)="([^"]+)"', lab_lines[0])}
    labels = {name: np.zeros(len(vals), bool) for name in label_ids}
    for line in lab_lines[1:]:
        s, ids = line.split(":")
        for i in ids.split():
            for name, j in label_ids.items():
                if int(i) == j:
                    labels[name][int(s)] = True
    return {"names": names, "vals": vals, "tra": tra, "srew": srew, "labels": labels, "pmin_done": pmin_done}


def _val(v: str):
    return 1 if v == "true" else 0 if v == "false" else int(v)


def joint_lp(m: Dict[str, object], deadline: int, thresholds: Dict[str, float]) -> Dict[str, float]:
    """The multi-objective LP (module docstring): optima of each objective alone and the joint answer."""
    names, vals, tra, srew = m["names"], m["vals"], m["tra"], m["srew"]
    S = len(vals)
    done = m["labels"]["done"]
    init = np.flatnonzero(m["labels"]["init"])[0]
    t, ok = vals[:, names.index("t")], vals[:, names.index("ok")].astype(bool)
    src, ch, dst, p = tra[:, 0].astype(int), tra[:, 1].astype(int), tra[:, 2].astype(int), tra[:, 3]
    live = ~done[src]                                   # "done" is absorbing: its choices are not variables
    src, ch, dst, p = src[live], ch[live], dst[live], p[live]
    pairs = np.unique(np.stack([src, ch], 1), axis=0)   # one variable per (state, choice)
    index = {(int(a), int(b)): i for i, (a, b) in enumerate(pairs)}
    var = np.array([index[(int(a), int(b))] for a, b in zip(src, ch)])
    n = len(pairs)
    # flow: out(s) - in(s) = [s = init], for every non-done state
    rows_out = pairs[:, 0]
    A = sp.csr_matrix((np.ones(n), (rows_out, np.arange(n))), shape=(S, n)) \
        - sp.csr_matrix((p * ~done[dst], (dst, var)), shape=(S, n))
    keep = ~done
    A_eq, b_eq = A[keep], (np.arange(S) == init)[keep].astype(float)
    on_time = np.bincount(var, weights=p * (done[dst] & (t[dst] <= deadline)), minlength=n)
    safe = np.bincount(var, weights=p * (done[dst] & ok[dst]), minlength=n)
    energy = srew[pairs[:, 0]]                          # a state's reward, charged on every choice taken from it

    def solve(c, A_ub=None, b_ub=None):
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        return res.fun if res.status == 0 else None, res.status

    out = {"states": S, "variables": n, "pmin_done": m["pmin_done"]}
    out["max_safe"] = -solve(-safe)[0]
    out["max_on_time"] = -solve(-on_time)[0]
    out["min_energy"] = solve(energy)[0]
    out["max_energy"] = -solve(-energy)[0]
    A_ub = sp.vstack([sp.csr_matrix(-safe), sp.csr_matrix(energy)])
    b_ub = np.array([-thresholds["safe"], thresholds["energy"]])
    val, status = solve(-on_time, A_ub, b_ub)
    out["joint_max_on_time"] = -val if val is not None else None
    out["jointly_feasible"] = bool(val is not None and -val >= thresholds["on_time"])
    out["joint_margin"] = (-val - thresholds["on_time"]) if val is not None else None
    out["lp_status"] = status
    return out


def check(instance, domain, cfg, prism: str, heap: str = "8g", timeout: float = 1800) -> Dict[str, object]:
    v = PolicyVerifier(domain, instance, cfg)
    d = instance.data
    th = {"safe": d["safe_threshold"], "on_time": d["on_time_threshold"], "energy": d["energy_threshold"]}
    start = time.time()
    m = export(prism, joint_copy(v.model, d["deadline"]), heap, timeout)
    built = time.time() - start
    res = joint_lp(m, d["deadline"], th)
    return {"instance": instance.id, "name": d.get("name"), **{f"threshold_{k}": x for k, x in th.items()},
            **res, "build_s": built, "lp_s": time.time() - start - built}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--ids", nargs="*", type=int, default=None)
    p.add_argument("--validate", action="store_true", help="also print PRISM's single-objective optima")
    a = p.parse_args()
    cfg = load_config("U1", [f"domain.dataset={a.data}"])
    domain = load_domain("uuv")
    from settings import get_prism_path
    prism = get_prism_path()
    for inst in domain.load_instances(a.data):
        if a.ids is not None and int(inst.id) not in a.ids:
            continue
        r = check(inst, domain, cfg, prism)
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        if a.validate:
            v = PolicyVerifier(domain, inst, cfg)
            props = [f"Pmax=? [ G !\"thruster_failure\" ]", f"Pmax=? [ F<={inst.data['deadline']} \"done\" ]",
                     'R{"energy"}min=? [ F "done" ]', 'R{"energy"}max=? [ F "done" ]']
            print("  PRISM:", dict(zip(["max_safe", "max_on_time", "min_energy", "max_energy"],
                                       [round(x, 6) for x in v.runner.run(v.model, props).initial_values])), flush=True)


if __name__ == "__main__":
    main()
