"""Where a rule list loses probability on a member: flux into failure states under the worst-case strategy."""
from typing import Dict, List, Sequence, Tuple

import numpy as np

from fresh.gw import ACTIONS, MOVES, REQS, THRESH, Member

SIDE = {"up": ("left", "right"), "right": ("up", "down"), "down": ("right", "left"), "left": ("down", "up")}


def _fail_sets(m: Member, req: str):
    """(target, avoid, maximize-for-adversary): the adversary minimizes reaching `target` before `avoid`, or
    maximizes reaching the bad set."""
    if req in m.req_sets:
        t, av = m.req_sets[req]
        return t, av, False
    return m.bad[req], None, True


def sources(m: Member, rules, req: str, allowed=None, decider=None, horizon: int = 0, top: int = 3):
    """The states where the worst-case run of `req` fails most: for avoid-type failures, the states from which
    the failure set is entered (flux); for runs that never reach the target, where they stall."""
    if allowed is None:
        allowed, decider = m.allowed(rules)
    target, avoid, adv_max = _fail_sets(m, req)
    _, act = m._reach(allowed, target, avoid, maximize=adv_max, return_policy=True)
    S = m.S
    horizon = horizon or 8 * m.n * m.n
    stop = target | (avoid if avoid is not None else np.zeros(S, bool))
    dist = np.zeros(S)
    dist[getattr(m, 'worst_init', {}).get(req, m.init)] = 1.0
    flux = np.zeros(S)
    succ = m.succ[np.arange(S), act]          # S x 3 under the adversary's choice
    fail_set = target if adv_max else avoid
    for _ in range(horizon):
        live = dist * ~stop
        if fail_set is not None:
            flux += live * (m.probs[None, :] * fail_set[succ]).sum(axis=1)
        nd = np.zeros(S)
        np.add.at(nd, succ.ravel(), (live[:, None] * m.probs[None, :]).ravel())
        dist = nd + dist * stop
    out = []
    if fail_set is not None and flux.sum() > 1e-9:
        for s in np.argsort(-flux)[:top]:
            if flux[s] < 1e-4:
                break
            out.append(("enters", int(s), float(flux[s]), int(act[s])))
    stalled = dist * ~stop
    if not adv_max and stalled.sum() > 1e-3:
        for s in np.argsort(-stalled)[:top]:
            if stalled[s] < 1e-4:
                break
            out.append(("stalls", int(s), float(stalled[s]), int(act[s])))
    return out, allowed, decider


def state_text(m: Member, s: int) -> str:
    x, y, seg, ob = int(m.sx[s]), int(m.sy[s]), int(m.sseg[s]), int(m.sob[s])
    flags = ", ".join(f"g{k}={'true' if seg > i else 'false'}" for i, k in enumerate(m.ks))
    obs = f"obs_idx={ob} (obstacle at {m.cycle[ob]}, next at {m.cycle[(ob + 1) % m.L]})" if m.cycle else ""
    return f"agent at ({x},{y}), {flags}, {obs}"


def action_effects(m: Member, s: int) -> Dict[str, str]:
    """Per action: the features true there and what the move risks in one step."""
    fv = m.feature_values
    out = {}
    for a in ACTIONS:
        feats = [name for (name, act), v in fv.items() if act == a and v[s]]
        nxt = m.succ[s, ACTIONS.index(a)]
        p_coll = sum(m.probs[k] for k in range(3) if m.on_obs[nxt[k]])
        p_early = 0.0
        for k in range(3):
            t = nxt[k]
            for i, kk in enumerate(m.ks):
                if m.at[kk][t] and m.sseg[s] < i and not (m.sx[s] == m.sx[t] and m.sy[s] == m.sy[t]):
                    p_early += m.probs[k]
        risk = []
        if p_coll > 0:
            risk.append(f"collision {p_coll:.2f}")
        if p_early > 0:
            risk.append(f"enters a later goal early {p_early:.2f}")
        out[a] = f"[{', '.join(feats) or 'no feature true'}]" + (f" risk: {'; '.join(risk)}" if risk else "")
    return out


def describe(m: Member, rules, results, top_reqs: int = 3, per_req: int = 2, order=None,
             action_lines: bool = True) -> str:
    """Feedback text for one member: the failing requirements and where they fail (with `order`, for the list
    whose `any` rules take the first allowed action in that order; without `action_lines`, no per-action
    sensor and risk lines)."""
    res = results
    failing = sorted(res.failing(), key=lambda r: res.worst[r] - THRESH[r])
    if not failing:
        return ""
    allowed, decider = m.allowed(rules, order)
    lines = []
    seen = set()
    for req in failing[:top_reqs]:
        srcs, _, _ = sources(m, rules, req, allowed, decider, top=per_req)
        lines.append(f"- {req} = {res.worst[req]:.3f} (needs >= {THRESH[req]}).")
        for kind, s, mass, a in srcs:
            if (s, kind) in seen:
                lines.append(f"    * same state as above: {state_text(m, s)} ({mass:.3f} of runs)")
                continue
            seen.add((s, kind))
            rule = int(decider[s])
            opts = [ACTIONS[i] for i in range(4) if allowed[s, i]]
            who = (f"rule {rule + 1} `{rules[rule][0]} -> {rules[rule][1]}`" if rule >= 0
                   else "NO RULE (uncovered: the worst case may take any action)")
            verb = "fails from here" if kind == "enters" else "is still stuck here at the end"
            lines.append(f"    * {mass:.3f} of runs {verb}: {state_text(m, s)}; decided by {who}; "
                         f"allowed {opts}, worst case takes {ACTIONS[a]}.")
            if action_lines:
                eff = action_effects(m, s)
                lines.append("      actions here: " + "; ".join(f"{a}: {e}" for a, e in eff.items()))
    return "\n".join(lines)
