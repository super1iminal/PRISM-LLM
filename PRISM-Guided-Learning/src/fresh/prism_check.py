"""Exact PRISM check of a rule list on a training member (to validate gw.py and to confirm finalists)."""
import os
import re
import subprocess
import tempfile
from typing import Dict, Sequence, Tuple

import jinja2

from fresh.gw import ACTIONS, MOVES, P_FWD, P_SLIP, REQS, ROOT, SLIP_LEFT, SLIP_RIGHT, Member

_ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(str(ROOT / "domains" / "gridworld")),
                          trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)

FORMULAS = {
    "goal1": 'F "at_goal1"', "goal2": 'F "at_goal2"', "goal3": 'F "at_goal3"',
    "seq_1_before_2": '!"at_goal2" U "at_goal1"',
    "seq_2_before_3": '!"at_goal3" U ("at_goal1" & (!"at_goal3" U "at_goal2"))',
    "complete_sequence": '(!"at_goal2" U "at_goal1") & (!"at_goal3" U ("at_goal1" & (!"at_goal3" U "at_goal2")))',
    "avoid_moving_seg1": 'G ("in_seg1" => !"at_moving_obs")',
    "avoid_moving_seg2": 'G ("in_seg2" => !"at_moving_obs")',
    "avoid_moving_seg3": 'G ("in_seg3" => !"at_moving_obs")',
}


def model_text(m: Member) -> str:
    goals = [{"k": k, "row": m.goals[k][0], "col": m.goals[k][1]} for k in m.ks]
    ctx = {"n": m.n, "goals": goals, "static": sorted(m.static), "cycle": m.cycle,
           "prob_forward": P_FWD, "prob_slip_left": P_SLIP, "prob_slip_right": P_SLIP,
           "moves": [{"name": a, "dr": dr, "dc": dc} for a, (dr, dc) in MOVES.items()],
           "outcomes": {a: [(a, P_FWD), (SLIP_LEFT[a], P_SLIP), (SLIP_RIGHT[a], P_SLIP)] for a in MOVES}}
    return _ENV.get_template("model.prism.j2").render(**ctx)


def prism_check(m: Member, rules: Sequence[Tuple[str, str]], timeout: float = 600) -> Dict[str, float]:
    """Worst case of every requirement (Pmin, or Pmin of the G formulas), by interval iteration."""
    pol = m.policy(rules)
    model = model_text(m) + "\n" + pol.to_prism_module(200000) + "\n"
    props = [f"Pmin=? [ {FORMULAS[r]} ]" for r in REQS]
    with tempfile.TemporaryDirectory(prefix="fresh_prism_") as tmp:
        mp, pp = os.path.join(tmp, "m.prism"), os.path.join(tmp, "p.props")
        open(mp, "w").write(model)
        open(pp, "w").write("\n".join(p + ";" for p in props) + "\n")
        cmd = ["prism.bat" if os.name == "nt" else "prism", mp, pp, "-explicit", "-intervaliter", "-topological",
               "-epsilon", "1e-9", "-maxiters", "100000000", "-javamaxmem", "4g"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    vals = re.findall(r"^Result: (\S+)", out, re.M)
    if len(vals) != len(REQS):
        raise RuntimeError(out[-3000:])
    return {r: float(v) for r, v in zip(REQS, vals)}
