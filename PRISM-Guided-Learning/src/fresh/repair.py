"""Verifier-guided local repair of a rule list on the training members (first-improvement hill climbing over
small edits: drop a rule, add or drop a conjunct, change an action, swap neighbours, specialize a rule)."""
import random
from typing import List, Sequence, Tuple

from core.rules import RuleError
from fresh.gw import ACTIONS, REQS, THRESH, Member

SENSORS = ["wall", "toward", "closer", "patrol", "early", "risky", "obstacle"]
PHASES = ["obs_idx = 0", "obs_idx = 1", "obs_idx = 1 + 1", "obs_idx = 1 + 1 + 1"]
LITERALS = SENSORS + [f"!{s}" for s in SENSORS] + ["g1", "!g1", "g2", "!g2", "g3", "!g3"] + PHASES \
    + [f"!({p})" for p in PHASES]
Rules = List[Tuple[str, str]]


def objective(results, cap: float = 0.05) -> float:
    """Sum over members and requirements of the margin, capped at `cap` (rewards fixing failures first)."""
    return sum(min(res.worst[r] - THRESH[r], cap) for res in results for r in REQS)


def conjuncts(cond: str) -> List[str]:
    """Top-level `&` parts of a condition."""
    parts, depth, cur = [], 0, ""
    for ch in cond:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "&" and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return [p for p in parts if p]


def literals(hidden=()) -> List[str]:
    """The repair's conjuncts, without the hidden sensors."""
    hidden = set(hidden)
    return [lit for lit in LITERALS if lit.lstrip("!") not in hidden]


def neighbours(rules: Rules, rng: random.Random, k: int, lits: Sequence[str] = LITERALS) -> List[Rules]:
    out = []
    n = len(rules)
    for _ in range(k):
        i = rng.randrange(n)
        c, a = rules[i]
        kind = rng.choice(["drop", "add", "remove", "action", "swap", "specialize", "specialize"])
        new = list(rules)
        if kind == "drop" and n > 1 and i < n - 1:
            del new[i]
        elif kind == "add":
            new[i] = (f"{c} & {rng.choice(lits)}" if c != "true" else rng.choice(lits), a)
        elif kind == "remove":
            parts = conjuncts(c)
            if len(parts) < 2:
                continue
            del parts[rng.randrange(len(parts))]
            new[i] = (" & ".join(parts), a)
        elif kind == "action":
            new[i] = (c, rng.choice([x for x in ACTIONS + ["any"] if x != a]))
        elif kind == "swap" and i < n - 2:
            new[i], new[i + 1] = new[i + 1], new[i]
        elif kind == "specialize":
            lit = rng.choice(lits)
            new.insert(i, (f"{c} & {lit}" if c != "true" else lit, rng.choice(ACTIONS + ["any"])))
        else:
            continue
        out.append(new)
    return out


def repair(rules: Rules, members: Sequence[Member], steps: int = 40, tries: int = 60, seed: int = 0,
           log=None, hidden=()) -> Tuple[Rules, float]:
    rng = random.Random(seed)

    def value(rs):
        try:
            return objective([m.check(rs, best=False) for m in members])
        except RuleError:
            return None

    cur, best = list(rules), value(rules)
    if best is None:
        return cur, float("-inf")
    target = 0.05 * len(REQS) * len(members)
    for step in range(steps):
        if best >= target - 1e-9:
            break
        improved = False
        for nb in neighbours(cur, rng, tries, literals(hidden)):
            v = value(nb)
            if v is not None and v > best + 1e-9:
                cur, best, improved = nb, v, True
                if log:
                    log(f"step {step}: {best:.4f}")
                break
        if not improved:
            break
    return cur, best
