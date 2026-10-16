"""The final list of a training set chosen from every round of its loops (`planner.pick: candidates`), after the
fresh-transfer session's method:

1. candidates: the `planner.pick_top` best distinct lists of every round of every chain, by the loop's values on the
   shown members (members passing, total shortfall, smallest margin);
2. repair (`planner.repair` steps, 0 = none): first-improvement hill climbing on the members over small edits (drop a
   rule, add or drop a conjunct, change an action, swap neighbours, specialize a rule with a conjunct), scored by the
   members' margins summed, each capped at 0.05 (so fixing failures counts first), with the domain's fast checker;
3. the list whose `any` rules resolved by the best action order on the members passes on most members, then the most
   members with `any` open, then the largest smallest margin (open, then resolved), then the shortest list.
The chosen list is frozen with `any` open; src/determinize.py resolves it for the other transfer variant.
"""
import random
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.domain import Requirement
from core.rules import ANY, RuleError, action_orders, fixed_order

Rules = List[Tuple[str, str]]
CAP = 0.05                       # the repair's per-requirement margin cap


def score(values: Sequence[Dict[str, float]], reqs: Sequence[Sequence[Requirement]]) -> Tuple[int, float, float]:
    """Higher is better: members passing, then minus the total shortfall, then the smallest margin."""
    passed = sum(all(r.satisfied(v[r.name]) for r in rs) for v, rs in zip(values, reqs))
    short = sum(r.shortfall(v[r.name]) for v, rs in zip(values, reqs) for r in rs)
    low = min(r.margin(v[r.name]) for v, rs in zip(values, reqs) for r in rs)
    return passed, -round(short, 9), low


def candidates(results: Sequence[Dict[str, Any]], reqs: Dict[str, Sequence[Requirement]], top: int) -> List[Dict]:
    """The `top` best distinct lists of every round of the chains' results (valid answers only), with where they came
    from and their score on the shown members by the loop's values."""
    found = {}
    for chain, result in enumerate(results):
        for it in result.get("iterations", []):
            if not it["rules"] or (it["invalid_answers"] and len(it["invalid_answers"]) >= it["llm_calls"]):
                continue
            rows = [m for m in it.get("members", []) if not m.get("hidden")]
            if not rows:
                rows = [{"instance": next(iter(reqs)), "worst": it["worst"]}]
            sc = score([m["worst"] for m in rows], [reqs[m["instance"]] for m in rows])
            key = tuple((r["condition"], r["action"]) for r in it["rules"])
            if key not in found or sc > found[key]["score"]:
                found[key] = {"rules": list(key), "score": sc, "chain": chain, "iteration": it["iteration"]}
    return sorted(found.values(), key=lambda c: c["score"], reverse=True)[:top]


def conjuncts(condition: str) -> List[str]:
    """The top-level `&` parts of a condition."""
    parts, depth, cur = [], 0, ""
    for ch in condition:
        depth += (ch == "(") - (ch == ")")
        if ch == "&" and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    return [p for p in parts + [cur.strip()] if p]


def neighbours(rules: Rules, rng: random.Random, k: int, literals: Sequence[str], actions: Sequence[str]) -> List[Rules]:
    out, n = [], len(rules)
    for _ in range(k):
        i = rng.randrange(n)
        c, a = rules[i]
        kind = rng.choice(["drop", "add", "remove", "action", "swap", "specialize", "specialize"])
        new = list(rules)
        if kind == "drop" and n > 1 and i < n - 1:
            del new[i]
        elif kind == "add":
            new[i] = (f"{c} & {rng.choice(literals)}" if c != "true" else rng.choice(literals), a)
        elif kind == "remove":
            parts = conjuncts(c)
            if len(parts) < 2:
                continue
            del parts[rng.randrange(len(parts))]
            new[i] = (" & ".join(parts), a)
        elif kind == "action":
            new[i] = (c, rng.choice([x for x in list(actions) + [ANY] if x != a]))
        elif kind == "swap" and i < n - 2:
            new[i], new[i + 1] = new[i + 1], new[i]
        elif kind == "specialize":
            lit = rng.choice(literals)
            new.insert(i, (f"{c} & {lit}" if c != "true" else lit, rng.choice(list(actions) + [ANY])))
        else:
            continue
        out.append(new)
    return out


def repair(rules: Rules, evaluate: Callable[[Rules], Optional[List[Dict[str, float]]]],
           reqs: Sequence[Sequence[Requirement]], literals: Sequence[str], actions: Sequence[str], steps: int,
           tries: int, seed: int, valid: Callable[[Rules], bool] = lambda rules: True) -> Rules:
    """Hill climbing from `rules`: `evaluate` gives the members' worst-case values (None if the rules do not parse),
    `valid` rejects lists the run does not allow (e.g. `rules.strict`)."""
    rng = random.Random(seed)

    def value(rs: Rules) -> Optional[float]:
        if not valid(rs):
            return None
        values = evaluate(rs)
        if values is None:
            return None
        return sum(min(r.margin(v[r.name]), CAP) for v, rq in zip(values, reqs) for r in rq)

    cur, best = list(rules), value(rules)
    if best is None:
        return cur
    target = CAP * sum(len(rq) for rq in reqs)
    for _ in range(steps):
        if best >= target - 1e-9:
            break
        for nb in neighbours(cur, rng, tries, literals, actions):
            v = value(nb)
            if v is not None and v > best + 1e-9:
                cur, best = nb, v
                break
        else:
            break
    return cur


def size(rules: Rules) -> int:
    """Names and numbers in the conditions (the last tie-break)."""
    return sum(len(re.findall(r"[A-Za-z_]\w*|\d+", c)) for c, _ in rules)


def resolved(rules: Rules, evaluate, reqs, actions: Sequence[str]) -> Tuple[Optional[Tuple], Optional[Tuple[str, ...]]]:
    """The best score of `rules` with every `any` resolved by one action order (all orders for up to four actions,
    else the rotations), and that order; the open score's order None if the rules have no `any`."""
    if not any(a == ANY for _, a in rules):
        values = evaluate(rules)
        return (score(values, reqs) if values is not None else None), None
    best = (None, None)
    for order in action_orders(list(actions), "all" if len(actions) <= 4 else "rotations"):
        values = evaluate(fixed_order(rules, order))
        if values is None:
            continue
        sc = score(values, reqs)
        if best[0] is None or (sc[0], sc[2]) > (best[0][0], best[0][2]):
            best = (sc, order)
    return best


def choose(finals: List[Dict]) -> Dict:
    """The final list among the repaired candidates (module docstring, step 3)."""
    def key(f):
        det = f["resolved_score"] or (-1, float("-inf"), float("-inf"))
        return det[0], f["final_score"][0], round(f["final_score"][2], 9), round(det[2], 9), -size(f["rules"])
    return max(finals, key=key)


def evaluator(checkers: Sequence[Any]) -> Callable[[Rules], Optional[List[Dict[str, float]]]]:
    """The members' worst-case values of a list by their fast checkers (None if it does not parse)."""
    def evaluate(rules: Rules) -> Optional[List[Dict[str, float]]]:
        try:
            return [c.worst(rules) for c in checkers]
        except RuleError:
            return None
    return evaluate
