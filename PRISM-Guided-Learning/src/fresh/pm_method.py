"""The same method on Pac-Man: one rule list for every game length, trained on horizons 14, 16, 18.

    python src/fresh/pm_method.py --chains 8 --rounds 6 --out out/fresh/pm/<name>
"""
import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.rules import RuleError, fixed_order  # noqa: E402
from fresh import llm  # noqa: E402
from fresh.pm import (ACTIONS, CROSSINGS, DIRECTIONS, GHOST_START, ROWS, START, TRAIN, PacInstance,  # noqa: E402
                      load, uniform_regret)

THRESHOLD = 0.56
HEADING = {0: "right", 1: "up", 2: "left", 3: "down"}


def maze_text() -> str:
    rows = []
    for y, row in ROWS.items():
        cells = []
        for x, c in enumerate(row, start=1):
            cells.append("#" if c == "#" else "P" if (x, y) == START else "G" if (x, y) == GHOST_START
                         else "+" if (x, y) in CROSSINGS else ".")
        rows.append(f"  y={y}   {''.join(cells)}")
    return "     x: 1234567890\n" + "\n".join(rows)


PROBLEM = f"""Pac-Man walks a small maze for MAXSTEPS moves while two ghosts move around it at random. You write ONE
controller for Pac-Man, as an ordered list of rules, that must work for EVERY game length: it is trained on games of
14, 16 and 18 moves and will be checked, unchanged, on games of other lengths (shorter and longer, up to 30 moves).

Maze (# wall, . corridor, + crossing where Pac-Man chooses, P Pac-Man's start, G the ghosts' start, also a crossing):
{maze_text()}

- Columns x = 1..10, rows y = 1..6 (y grows upwards). Pac-Man starts at {START} heading right; both ghosts start at
  {GHOST_START} heading right.
- Each round ghost 0 moves, then ghost 1, then Pac-Man. A crash is Pac-Man and a ghost on the same cell after any
  move. The game ends after MAXSTEPS Pac-Man moves.
- In a corridor Pac-Man keeps going in his heading and turns at corners (forced; rules have no effect there). At a
  crossing {", ".join(str(c) for c in CROSSINGS)} the rules choose right, up, left or down; a direction with a wall
  keeps him in place for that move.
- Ghosts keep going in corridors; at a crossing a ghost picks a direction at random with learned probabilities
  (they tend to turn towards Pac-Man and rarely reverse). A ghost that can no longer reach Pac-Man in the moves
  left (distance in cells > 2 * moves left) is removed: its position becomes (0,0). The two ghosts are sorted after
  each Pac-Man move (ghost 0 has the smaller x, or the same x and a y no larger), so their numbering can swap.

## Requirement (checked exactly, in the WORST case)
The probability that Pac-Man crashes at some point must be <= {THRESHOLD}. Wherever no rule applies, or a rule
with action `any` allows several directions, an adversary picks among the allowed directions. Avoiding crashes
completely is impossible: even the best possible controller crashes with probability about 0.551 in the training
games, so the rules must choose (nearly) the best direction in every situation that matters.

## Rule language
An ordered list of rules {{"condition": ..., "action": ...}}; at a crossing the FIRST rule whose condition holds
decides. Actions: right, up, left, down, or `any` (allows every direction for which the condition holds, with the
per-direction sensors evaluated for that direction). Conditions use ! & | => ( ), = != < <= > >=, + -, and only
the numbers 0 and 1, which may not be added together (no `1+1`): rules cannot name particular cells, so compare
positions with each other instead (e.g. `xG0 < xP`, `yG1 = yP`, `xP = xG0 + 1`) and use the sensors.
Names:
- `xP`, `yP`, `dP`: Pac-Man's column, row and heading (0 = right, 1 = up, 2 = left, 3 = down; e.g. `dP > 1` is
  left or down).
- `xG0`, `yG0`, `dG0`, `xG1`, `yG1`, `dG1`: the ghosts' columns, rows and headings (a removed ghost is at (0,0)).
- `steps`: moves made so far; `MAXSTEPS`: the game's length; `moves_left` (= MAXSTEPS - steps).
- per-direction sensors (for the rule's direction):
    - `wall`: this direction is a wall at Pac-Man's crossing (he would stay put)
    - `nearer_g0`, `nearer_g1`: this direction goes towards ghost 0 / 1 by its row or column (while it is in the game)
    - `closer_g0`, `closer_g1`: this direction's first step shortens Pac-Man's distance to ghost 0 / 1 along the maze
End the list with a catch-all rule (condition `true`)."""


def initial_prompt() -> str:
    return PROBLEM + """

## Answer
Think about where the danger is (the ghosts start next to Pac-Man and turn towards him), how to keep away from both
ghosts at each crossing, and how the rules can stay correct for short and long games. Then give the complete list
as JSON: {"rules": [{"condition": "...", "action": "..."}, ...]}"""


def revise_prompt(rules, report: str, history: str) -> str:
    listing = "\n".join(f'{i}. {{"condition": {json.dumps(c)}, "action": {json.dumps(a)}}}'
                        for i, (c, a) in enumerate(rules, start=1))
    return PROBLEM + f"""

## Your current list
{listing}

## Its exact worst-case results on the training games
{report}
{history}
Revise the list so that the crash probability is <= {THRESHOLD} in every training game, keeping the rules general
(they must also work for other game lengths). The situations above show where the rules lose the most: the
state, the rule that decided, and the crash probability after each direction (if the rules decide afterwards).
Give the complete revised list as JSON: {{"rules": [{{"condition": "...", "action": "..."}}, ...]}}"""


def uniform_prompt(rules, insts, history: str) -> str:
    listing = "\n".join(f'{i}. {{"condition": {json.dumps(c)}, "action": {json.dumps(a)}}}'
                        for i, (c, a) in enumerate(rules, start=1))
    reports = "\n".join(regret_report(p, rules) for p in insts)
    return PROBLEM + f"""

## Your current list
{listing}

It meets the requirement in every training game (crash probability 0.551 from the start, the best possible).
Longer games give the ghosts more time, so the list must also evade well from every situation, not only from the
start. Below, for each training game, are the situations where the rules crash most often compared with the best
possible controller (from any configuration of Pac-Man and the ghosts):
{reports}
{history}
Revise the list so that it evades well in these situations too, without losing the result from the start (keep
the rules general: no particular cells). Give the complete revised list as JSON:
{{"rules": [{{"condition": "...", "action": "..."}}, ...]}}"""


# ------------------------------------------------------------------ evaluation and feedback

def evaluate(rules, insts: List[PacInstance]) -> List[float]:
    return [p.worst(rules) for p in insts]


def score(crash: List[float]) -> Tuple:
    margins = [THRESHOLD - c for c in crash]
    return (sum(m >= -1e-12 for m in margins), -round(sum(max(0.0, -m) for m in margins), 6), min(margins))


def blame(p: PacInstance, rules, top: int = 4) -> str:
    allowed, decider = p.allowed(rules)
    mask = p.choice_mask(allowed)
    v = p.values(mask)
    q = np.zeros(len(p.ch_state))
    np.add.at(q, p.tr_choice, p.tr_prob * v[p.tr_target])
    # the adversary's choice per state (largest crash probability among the allowed choices)
    qm = np.where(mask, q, -1.0)
    best_choice = np.full(p.S, -1)
    order = np.lexsort((-qm, p.ch_state))
    first = np.ones(len(order), bool)
    first[1:] = p.ch_state[order][1:] != p.ch_state[order][:-1]
    best_choice[p.ch_state[order][first]] = order[first]
    # visit probabilities under that strategy
    visit = np.zeros(p.S)
    dist = np.zeros(p.S)
    dist[p.init] = 1.0
    tr_sel = best_choice[p.ch_state[p.tr_choice]] == p.tr_choice
    for _ in range(3 * p.H + 6):
        live = dist * ~p.crash
        visit += live
        nd = np.zeros(p.S)
        np.add.at(nd, p.tr_target[tr_sel], live[p.ch_state[p.tr_choice[tr_sel]]] * p.tr_prob[tr_sel])
        dist = nd
    # regret at decision states: worst allowed direction vs the best direction (same continuation)
    pol = p.ch_action >= 0
    qa = np.full((p.S, 4), np.nan)
    qa[p.ch_state[pol], p.ch_action[pol]] = q[pol]
    worst_allowed = np.where(allowed, qa, -1).max(axis=1)
    best_any = np.nanmin(np.where(np.isnan(qa), np.inf, qa), axis=1)
    regret = np.where(p.decision, (worst_allowed - best_any) * visit, 0.0)
    lines = []
    for s in np.argsort(-regret)[:top]:
        if regret[s] < 1e-4:
            break
        sv = {k: int(p.state_vars[k][s]) for k in ("xP", "yP", "dP", "xG0", "yG0", "dG0", "xG1", "yG1", "dG1",
                                                    "steps")}
        g = [f"ghost {i} " + (f"at ({sv[f'xG{i}']},{sv[f'yG{i}']}) heading {HEADING[sv[f'dG{i}']]}"
                              if sv[f"xG{i}"] else "removed") for i in (0, 1)]
        rule = int(decider[s])
        who = f"rule {rule + 1} `{rules[rule][0]} -> {rules[rule][1]}`" if rule >= 0 else "NO RULE (uncovered)"
        acts = ", ".join(f"{a} {qa[s, i]:.3f}" + ("*" if allowed[s, i] else "") for i, a in enumerate(ACTIONS))
        lines.append(f"    * reached with probability {visit[s]:.3f}, costs {regret[s]:.3f}: Pac-Man at "
                     f"({sv['xP']},{sv['yP']}) heading {HEADING[sv['dP']]}, {g[0]}, {g[1]}, steps {sv['steps']} "
                     f"(moves_left {p.H - sv['steps']}); decided by {who}. Crash probability after each direction "
                     f"(* = allowed): {acts}")
    return "\n".join(lines)


def report(rules, insts: List[PacInstance], crash: List[float]) -> str:
    out = []
    for p, c in zip(insts, crash):
        ok = c <= THRESHOLD + 1e-12
        out.append(f"### {p.H}-move game: crash probability {c:.4f} ({'OK' if ok else 'TOO HIGH'}; needs <= "
                   f"{THRESHOLD}, best possible 0.551)")
        if not ok:
            out.append(blame(p, rules))
    return "\n".join(out)


OPT: Dict[int, np.ndarray] = {}


def optimum_values(p: PacInstance) -> np.ndarray:
    if p.H not in OPT:
        OPT[p.H] = p.values(np.ones(len(p.ch_state), bool), maximize=False)
    return OPT[p.H]


def regret_report(p: PacInstance, rules, top: int = 5) -> str:
    """States of the model (any configuration, not only those the start leads to) where the rules crash most
    often compared with the best controller: what makes a list hold up in longer or shorter games."""
    opt = optimum_values(p)
    allowed, decider = p.allowed(rules)
    v = p.values(p.choice_mask(allowed))
    q = np.zeros(len(p.ch_state))
    np.add.at(q, p.tr_choice, p.tr_prob * v[p.tr_target])
    pol = p.ch_action >= 0
    qa = np.full((p.S, 4), np.nan)
    qa[p.ch_state[pol], p.ch_action[pol]] = q[pol]
    regret = np.where(p.decision, v - opt, 0.0)
    lines, seen = [f"### {p.H}-move game: average excess crash probability over all decision situations "
                   f"{regret[p.decision].mean():.4f}"], set()
    for s in np.argsort(-regret):
        if len(seen) >= top or regret[s] < 1e-3:
            break
        sv = {k: int(p.state_vars[k][s]) for k in ("xP", "yP", "dP", "xG0", "yG0", "dG0", "xG1", "yG1", "dG1",
                                                    "steps")}
        key = (sv["xP"], sv["yP"], sv["dP"], int(decider[s]))
        if key in seen:
            continue
        seen.add(key)
        g = [f"ghost {i} " + (f"at ({sv[f'xG{i}']},{sv[f'yG{i}']}) heading {HEADING[sv[f'dG{i}']]}"
                              if sv[f"xG{i}"] else "removed") for i in (0, 1)]
        rule = int(decider[s])
        who = f"rule {rule + 1} `{rules[rule][0]} -> {rules[rule][1]}`" if rule >= 0 else "NO RULE (uncovered)"
        acts = ", ".join(f"{a} {qa[s, i]:.3f}" + ("*" if allowed[s, i] else "") for i, a in enumerate(ACTIONS))
        lines.append(f"    * Pac-Man at ({sv['xP']},{sv['yP']}) heading {HEADING[sv['dP']]}, {g[0]}, {g[1]}, "
                     f"moves_left {p.H - sv['steps']}: crashes {v[s]:.3f} (best possible {opt[s]:.3f}); decided by "
                     f"{who}. Crash probability after each direction (* = allowed): {acts}")
    return "\n".join(lines)


def mean_regret(rules, insts) -> float:
    return float(np.mean([uniform_regret(p, rules, optimum_values(p)) for p in insts]))


LITERAL_SUM = re.compile(r"(?<!\w)\d+\s*[-+]\s*\d+(?!\w)")


def validate(rules, insts) -> Optional[str]:
    if not rules:
        return "no rules found in the answer"
    bad = [f"rule {i} ({c} -> {a})" for i, (c, a) in enumerate(rules, start=1) if LITERAL_SUM.search(c)]
    if bad:
        return ("adding or subtracting number literals (e.g. `1+1`) is not allowed; compare positions, headings and "
                "steps with each other, with 0 or 1, or use the sensors: " + "; ".join(bad))
    try:
        insts[0].policy(rules)
    except RuleError as e:
        return str(e)
    return None


UNIFORM = False   # --uniform: after the training games pass, keep revising to lower the regret everywhere


def chain(insts, rounds: int, log: Path, tag: str) -> List[Dict]:
    attempts, best, history_lines = [], None, []
    prompt = initial_prompt()
    for rnd in range(rounds + 1):
        text, usage = llm.chat([{"role": "user", "content": prompt}])
        rules = llm.parse_rules(text)
        err = validate(rules, insts)
        entry = {"chain": tag, "round": rnd, "rules": rules, "error": err, "usage": usage}
        if err is None:
            crash = evaluate(rules, insts)
            entry["score"], entry["crash"] = score(crash), crash
            if UNIFORM and entry["score"][0] == len(insts):
                entry["regret"] = mean_regret(rules, insts)
                entry["score"] = entry["score"] + (-round(entry["regret"], 6),)
            else:
                entry["score"] = entry["score"] + (-1.0,)
            if best is None or entry["score"] > best["score"]:
                best = entry
            history_lines.append(f"- attempt {rnd + 1}: crash " + ", ".join(f"{c:.3f}" for c in crash)
                                 + (f", average excess {entry['regret']:.4f}" if "regret" in entry else ""))
        else:
            history_lines.append(f"- attempt {rnd + 1}: rejected ({err.splitlines()[0][:200]})")
        attempts.append(entry)
        with open(log / f"{tag}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({**entry, "prompt": prompt, "answer": text}) + "\n")
        if not UNIFORM and best is not None and best["score"][0] == len(insts) and best["score"][2] > 0.002:
            break
        if best is None:
            prompt = initial_prompt() + (f"\n\nYour previous answer was rejected:\n{err}\nFix this." if err else "")
            continue
        history = "\n## Your attempts so far\n" + "\n".join(history_lines) + "\n" if len(history_lines) > 1 else ""
        if UNIFORM and best["score"][0] == len(insts):
            prompt = uniform_prompt(best["rules"], insts, history)
        else:
            prompt = revise_prompt(best["rules"], report(best["rules"], insts, best["crash"]), history)
    return attempts


# ------------------------------------------------------------------ repair, determinization, selection

LITERALS = (["wall", "!wall", "nearer_g0", "!nearer_g0", "nearer_g1", "!nearer_g1", "closer_g0", "!closer_g0",
             "closer_g1", "!closer_g1"]
            + [f"{a} {op} {b}" for a, b in (("xG0", "xP"), ("yG0", "yP"), ("xG1", "xP"), ("yG1", "yP"))
               for op in ("<", ">", "=")]
            + ["dP = 0", "dP = 1", "dP > 1", "dP < 1", "xG0 = 0", "xG1 = 0", "xG0 != 0", "xG1 != 0"])


def repair(rules, insts, steps: int = 25, tries: int = 40, seed: int = 0):
    rng = random.Random(seed)

    def objective(rs):
        try:
            first = insts[0].worst(rs)
        except RuleError:
            return None
        return first

    def full(rs):
        return sum(min(THRESHOLD - c, 0.005) for c in evaluate(rs, insts))

    cur, cur_first, cur_full = list(rules), objective(rules), full(rules)
    for _ in range(steps):
        if cur_full >= 0.005 * len(insts) - 1e-12:
            break
        improved = False
        for _ in range(tries):
            i = rng.randrange(len(cur))
            c, a = cur[i]
            kind = rng.choice(["drop", "add", "action", "specialize", "specialize"])
            new = list(cur)
            if kind == "drop" and i < len(cur) - 1:
                del new[i]
            elif kind == "add":
                new[i] = (f"{c} & {rng.choice(LITERALS)}" if c != "true" else rng.choice(LITERALS), a)
            elif kind == "action":
                new[i] = (c, rng.choice([x for x in ACTIONS + ["any"] if x != a]))
            elif kind == "specialize":
                lit = rng.choice(LITERALS)
                new.insert(i, (f"{c} & {lit}" if c != "true" else lit, rng.choice(ACTIONS + ["any"])))
            else:
                continue
            f1 = objective(new)
            if f1 is None or f1 > cur_first + 1e-12:
                continue
            ff = full(new)
            if ff > cur_full + 1e-9:
                cur, cur_first, cur_full, improved = new, f1, ff, True
                break
        if not improved:
            break
    return cur


def size(rules) -> int:
    """Names and numbers in the conditions: the tie-break between lists with equal training margins."""
    return sum(len(re.findall(r"[A-Za-z_]\w*|\d+", c)) for c, _ in rules)


def selection_key(f: Dict) -> Tuple:
    """Certified on the most training members (deterministic, then with `any`), then the largest smallest margin
    (with `any`, then deterministic), then the simplest list."""
    return (f["det_score"][0], f["score"][0], round(f["score"][2], 9), round(f["det_score"][2], 9),
            -round(f.get("det_regret", 0.0), 6), -size(f["rules"]))


def determinize(rules, insts):
    """The global action order whose deterministic list does best on the training games (in uniform mode, then
    with the lowest regret in the longest training game)."""
    if not any(a == "any" for _, a in rules):
        return list(rules), None, score(evaluate(rules, insts))
    best = None
    for order in permutations(ACTIONS):
        det = fixed_order(rules, order)
        sc = score(evaluate(det, insts))
        key = (sc[0], sc[2])
        if UNIFORM and sc[0] == len(insts):
            key += (-uniform_regret(insts[-1], det, optimum_values(insts[-1])),)
        if best is None or key > best[3]:
            best = (det, list(order), sc, key)
    return best[:3]


def run(chains: int, rounds: int, out: Path, top: int = 6, repair_steps: int = 25):
    out.mkdir(parents=True, exist_ok=True)
    insts = load(TRAIN)
    start = time.time()
    with ThreadPoolExecutor(max_workers=chains) as pool:
        futures = [pool.submit(chain, insts, rounds, out, f"pacman_c{i}") for i in range(chains)]
        attempts = [a for f in futures for a in f.result()]
    llm_seconds = time.time() - start
    valid = sorted((a for a in attempts if a.get("score")), key=lambda a: a["score"], reverse=True)
    seen, cands = set(), []
    for a in valid:
        key = tuple(map(tuple, a["rules"]))
        if key not in seen:
            seen.add(key)
            cands.append(a)
    finals = []
    for i, a in enumerate(cands[:top]):
        rules = [tuple(r) for r in a["rules"]]
        if repair_steps:
            rules = repair(rules, insts, steps=repair_steps, seed=i)
        crash = evaluate(rules, insts)
        det, order, det_score = determinize(rules, insts)
        finals.append({"from": [a["chain"], a["round"]], "llm_score": a["score"], "rules": rules,
                       "score": score(crash), "crash": crash, "det_rules": det, "det_order": order,
                       "det_score": det_score})
        if UNIFORM and det_score[0] == len(insts):
            finals[-1]["det_regret"] = mean_regret(det, insts)
    finals.sort(key=selection_key, reverse=True)
    summary = {"chains": chains, "rounds": rounds, "uniform": UNIFORM, "llm_seconds": round(llm_seconds, 1),
               "seconds": round(time.time() - start, 1), "usage": dict(llm.USAGE), "attempts": len(attempts),
               "valid": len(valid), "best": finals[0] if finals else None, "finals": finals}
    (out / "pacman_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--repair-steps", type=int, default=25)
    ap.add_argument("--uniform", action="store_true", help="keep improving the regret over all situations")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    UNIFORM = args.uniform
    s = run(args.chains, args.rounds, Path(args.out), args.top, args.repair_steps)
    print(json.dumps({k: v for k, v in s.items() if k not in ("best", "finals")}, default=str))
    b = s["best"]
    if b:
        print("best", b["score"], b["crash"], "det", b["det_score"], b["det_order"])
        for c, a in b["rules"]:
            print(f"  {c} -> {a}")
