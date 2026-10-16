"""Family-level rule lists: Haiku writes one list for all training members at once, revised from exact worst-case
feedback on every member; the list that passes all members with the largest margin is kept.

Run from PRISM-Guided-Learning/:
    python src/fresh/method.py rooms --chains 6 --rounds 6 --out out/fresh/try1
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.rules import RuleError  # noqa: E402
from fresh import llm  # noqa: E402
from fresh.explain import describe  # noqa: E402
from fresh.gw import ACTIONS, FAMILIES, REQS, THRESH, Member, Result, load  # noqa: E402

POSITION_NAMES = re.compile(r"\b(x|y|N|G\d[RC])\b")


@dataclass
class Settings:
    """Everything a run may change; the defaults are the frozen family method."""
    kind: str = "family"                 # family | random (prompt wording)
    data: str = "families_v0.csv"
    hidden: tuple = ("hazard",)          # sensors left out of the vocabulary
    resolve_in_loop: bool = False        # the verifier resolves each candidate's `any` before scoring and feedback
    action_lines: bool = True            # feedback shows each action's sensors and one-step risks


CFG = Settings()

FEATURE_TEXT = {
    "wall": "the move would leave the grid or hit a static obstacle (the agent then stays where it is)",
    "toward": "the move goes towards the next goal not yet reached (reduces the row or column difference)",
    "closer": "the move's target cell is closer to the next goal along a shortest path through free cells "
              "(avoiding static obstacles and later goals, and the patrol cells unless every route crosses them)",
    "patrol": "the move's target cell is on the moving obstacle's patrol path",
    "early": "the move's target cell is a goal that must not be entered yet",
    "risky": "a slip of this move (to either side) could land on the patrol path or on a goal that must not be "
             "entered yet",
    "obstacle": "the moving obstacle will be on the move's target cell right after this move",
}


PATROL = {
    "family": """- A moving obstacle patrols a short path of cells, advancing one position per agent step through a cycle of 4
  positions (obs_idx 0 -> 1 -> 2 -> 3 -> 0 ...). Its cells may be entered, but being on the same cell as the
  obstacle at the same time is a collision.""",
    "random": """- A moving obstacle patrols a short path of cells back and forth, advancing one position per agent step through
  its cycle (obs_idx 0 -> 1 -> ... -> 0; the cycle's length differs between grids, see each grid below). Its cells
  may be entered, but being on the same cell as the obstacle at the same time is a collision.""",
}


def world_text(position: bool) -> str:
    names = ["- `g1`, `g2`, `g3` (bool): goal k has been reached in order (g2 only counts after g1, g3 after g2).",
             "- `obs_idx` (int): the moving obstacle's current position in its patrol cycle (0, 1, 2, 3; see each "
             "member below)." if CFG.kind == "family" else
             "- `obs_idx` (int): the moving obstacle's current position in its patrol cycle (see each grid below; "
             "the same index means different cells in different grids)."]
    if position:
        names += ["- `x`, `y` (int): the agent's row and column.",
                  "- constants: `N` (grid size), `G1R`, `G1C`, `G2R`, `G2C`, `G3R`, `G3C` (goal rows and columns)."]
    names += ["- per-action sensors (true/false for the rule's action; with `any`, for each action separately):"]
    names += [f"    - `{k}`: {v}" for k, v in FEATURE_TEXT.items() if k not in CFG.hidden]
    restrict = ("Conditions may only use these names, `true`/`false` and the numbers 0 and 1 (write e.g. "
                "`obs_idx = 1 + 1` for 2)." if position else
                "Conditions may only use the names above, `true`/`false` and the numbers 0 and 1 (write e.g. "
                "`obs_idx = 1 + 1` for 2). The agent's coordinates and the grid size are NOT available: the rules "
                "must work from the sensors, the flags and the obstacle phase alone, which is what makes them carry "
                "over to every member.")
    WHAT = "member" if CFG.kind == "family" else "grid"
    return f"""## The world (the same in every {WHAT})
- An agent moves on an NxN grid: x = row (0 = top), y = column (0 = left). It starts at (0,0).
- Goals 1, 2, 3 must be visited in this order. Flag g1 becomes true when the agent enters goal 1's cell; g2 when it
  enters goal 2's cell while g1 is true; g3 when it enters goal 3's cell while g2 is true.
- Each step the agent picks up, right, down or left. The move succeeds with probability 0.7 and slips 90 degrees to
  either side with probability 0.15 each (e.g. `down` goes down 0.7, left 0.15, right 0.15).
- Moving off the grid or into a static obstacle leaves the agent where it is.
{PATROL[CFG.kind]}

## Requirements (checked exactly, in the WORST case)
- goal1, goal2, goal3: the agent eventually enters goal k's cell; probability >= 0.8 each.
- seq_1_before_2 / seq_2_before_3 / complete_sequence: goal 2's cell is never entered before goal 1, and goal 3's
  cell never before goals 1 and 2 (each, and both together); probability >= 0.8.
- avoid_moving_seg1/2/3: while heading to goal k, the agent never collides with the moving obstacle;
  probability >= 0.7 for each k.
"Worst case": wherever no rule applies, or a rule with action `any` allows several actions, an adversary picks
among the allowed actions to make the requirement fail. So every state must be covered, and every action a rule
allows must be safe. The adversary can also stall: if a rule allows a move that does not make progress (e.g. a
sideways or backwards move), the worst case may take it every time it can, and the agent may circle forever and
never reach the goal. Waiting is only safe when it ends: the obstacle's phase changes every step.

## Rule language
An ordered list of rules {{"condition": ..., "action": ...}}; in each state the FIRST rule whose condition holds
decides. Actions: up, right, down, left, or `any`. A rule with `any` allows every action for which its condition
holds (sensors evaluated for that action) and applies if at least one action qualifies. Conditions use
! & | => ( ), = != < <= > >=, + -. Names:
{chr(10).join(names)}
{restrict}
End the list with a catch-all rule (condition `true`) so that no state is left uncovered."""


def member_text(m: Member) -> str:
    lines = [f"### Member {m.n}x{m.n}"]
    grid = [["." for _ in range(m.n)] for _ in range(m.n)]
    for r, c in m.cycle:
        grid[r][c] = "M"
    for k in m.ks:
        r, c = m.goals[k]
        grid[r][c] = str(k)
    for r, c in m.static:
        grid[r][c] = "X"
    grid[0][0] = "S"
    lines.append("```")
    lines.append("    " + " ".join(f"{c % 10}" for c in range(m.n)) + "   <- y")
    for r in range(m.n):
        lines.append(f"{r:2d}  " + " ".join(grid[r]))
    lines.append("```")
    lines.append("Goals: " + ", ".join(f"{k} at {m.goals[k]}" for k in m.ks) + ". Patrol cycle: "
                 + ", ".join(f"obs_idx {i} = {c}" for i, c in enumerate(m.cycle)) + ".")
    return "\n".join(lines)


INTRO_RANDOM = """You write ONE controller for random gridworlds, as an ordered list of rules. The same list, unchanged, will be
checked on the training grids below ({sizes}) and on LARGER random grids (12x12 and 13x13) with other goals,
obstacles and patrols. Rules that only work because of one grid's exact layout will fail there; describe a
strategy that holds in every grid."""


def initial_prompt(fam: str, members: List[Member], position: bool) -> str:
    sizes = ", ".join(f"{m.n}x{m.n}" for m in members)
    if CFG.kind == "random":
        return f"""{INTRO_RANDOM.format(sizes=sizes)}

{world_text(position)}

## Training grids (S = start, 1/2/3 = goals, X = static obstacle, M = patrol cells, . = free)
{chr(10).join(member_text(m) for m in members)}

## Answer
Think about a strategy that works in every grid: how to make progress towards the next goal, how to cross or
avoid the patrol without collisions, how slips can hurt, and what to do when the preferred move is unsafe. Then
give the complete list as JSON:
{{"rules": [{{"condition": "...", "action": "..."}}, ...]}}"""
    return f"""You write ONE controller for a whole family of gridworlds ("{fam}"), as an ordered list of rules. The same list,
unchanged, will be checked on every member: the training members below ({sizes}) and LARGER members of the same
family (up to about 14x14) that have the same structure (same kind of walls, doors or gaps, goal placement and
patrol) but other sizes, offsets and scattered obstacles. Rules that only work because of one member's exact
layout will fail there; describe the strategy so that it holds in every member.

{world_text(position)}

## Training members (S = start, 1/2/3 = goals, X = static obstacle, M = patrol cells, . = free)
{chr(10).join(member_text(m) for m in members)}

## Answer
Think about a strategy that works in every member: how to make progress towards the next goal, how to cross or
avoid the patrol without collisions (use obs_idx, `obstacle`, `patrol`), how slips can hurt (`risky`, `early`),
and what to do when the preferred move is unsafe. Then give the complete list as JSON:
{{"rules": [{{"condition": "...", "action": "..."}}, ...]}}"""


def revise_prompt(fam: str, members: List[Member], position: bool, rules, report: str, history: str) -> str:
    listing = "\n".join(f'{i}. {{"condition": {json.dumps(c)}, "action": {json.dumps(a)}}}'
                        for i, (c, a) in enumerate(rules, start=1))
    return f"""{initial_prompt(fam, members, position)}

## Your current list
{listing}

## Its exact worst-case results on the training members
{report}
{history}
Revise the list so that every requirement holds on every training member, while keeping the rules general (they
must also work on the larger members). Use the failure locations above: they show the state, the rule that decided
it, what the worst case did, and what each action would do there. Give the complete revised list as JSON:
{{"rules": [{{"condition": "...", "action": "..."}}, ...]}}"""


# ------------------------------------------------------------------ evaluation

def validate(rules, members: List[Member], position: bool) -> Optional[str]:
    if not rules:
        return "no rules found in the answer"
    errors = []
    for i, (c, a) in enumerate(rules, start=1):
        if not position and POSITION_NAMES.search(c):
            errors.append(f"rule {i} ({c} -> {a}): uses coordinates or grid constants "
                          f"({', '.join(sorted(set(POSITION_NAMES.findall(c))))}), which are not available")
    if errors:
        return "\n".join(errors)
    try:
        members[0].policy(rules)
    except RuleError as e:
        return str(e)
    return None


def score(results: List[Result]) -> Tuple:
    """Higher is better: members certified, then -total shortfall, then the smallest margin."""
    shortfall = sum(max(0.0, THRESH[r] - res.worst[r]) for res in results for r in REQS)
    return (sum(r.certified for r in results), -round(shortfall, 6), min(r.margin for r in results))


def evaluate(rules, members: List[Member], order=None) -> List[Result]:
    return [m.check(rules, best=False, order=order) for m in members]


def best_order(rules, members: List[Member]):
    """The action order that resolves the list's `any` rules best on the training members (None if it has none),
    with the results under it."""
    from itertools import permutations
    if not any(a == "any" for _, a in rules):
        return None, evaluate(rules, members)
    best = None
    for order in permutations(ACTIONS):
        results = evaluate(rules, members, list(order))
        sc = score(results)
        if best is None or (sc[0], sc[2]) > (best[2][0], best[2][2]):
            best = (list(order), results, sc)
    return best[0], best[1]


def report(rules, members: List[Member], results: List[Result], order=None) -> str:
    out = []
    for m, res in zip(members, results):
        if res.certified:
            out.append(f"### {m.n}x{m.n}: all requirements hold (smallest margin {res.margin:+.3f}).")
        else:
            out.append(f"### {m.n}x{m.n}: FAILS {', '.join(res.failing())}"
                       + (f"; {res.uncovered} reachable states are uncovered" if res.uncovered else ""))
            out.append(describe(m, rules, res, order=order, action_lines=CFG.action_lines))
    if order is not None:
        out.insert(0, f"(Your `any` rules are resolved by the verifier: each takes the first action it allows in the "
                      f"order {', '.join(order)}, the order that does best on the training members.)")
    return "\n".join(out)


# ------------------------------------------------------------------ search

def chain(fam: str, members: List[Member], position: bool, rounds: int, log: Path, tag: str) -> List[Dict]:
    """One independent LLM chain: an initial list, then `rounds` revisions of the best list so far."""
    attempts = []
    best = None
    prompt = initial_prompt(fam, members, position)
    history_lines = []
    for rnd in range(rounds + 1):
        text, usage = llm.chat([{"role": "user", "content": prompt}])
        rules = llm.parse_rules(text)
        err = validate(rules, members, position)
        entry = {"chain": tag, "round": rnd, "rules": rules, "error": err, "usage": usage}
        if err is None:
            if CFG.resolve_in_loop:
                entry["order"], results = best_order(rules, members)
            else:
                results = evaluate(rules, members)
            entry["score"] = score(results)
            entry["worst"] = [r.worst for r in results]
            entry["certified"] = [r.certified for r in results]
            if best is None or entry["score"] > best["score"]:
                best = {**entry, "results": results}
            history_lines.append(f"- attempt {rnd + 1}: certified on {entry['score'][0]}/3 members, total shortfall "
                                 f"{-entry['score'][1]:.3f}")
        else:
            history_lines.append(f"- attempt {rnd + 1}: rejected ({err.splitlines()[0][:200]})")
        attempts.append(entry)
        with open(log / f"{tag}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({**entry, "prompt": prompt, "answer": text}) + "\n")
        if best is not None and best["score"][0] == len(members) and best["score"][2] > 0.05:
            break
        if best is None:
            prompt = initial_prompt(fam, members, position) + (
                f"\n\nYour previous answer was rejected:\n{err}\nFix this and answer again." if err else "")
            continue
        history = ""
        if len(history_lines) > 1:
            history = "\n## Your attempts so far\n" + "\n".join(history_lines) + "\n"
            if err is not None:
                history += f"Your last answer was rejected: {err}\n"
        prompt = revise_prompt(fam, members, position, best["rules"],
                               report(best["rules"], members, best["results"], best.get("order")), history)
    return attempts


def size(rules) -> int:
    """Names and numbers in the conditions: the tie-break between lists with equal training margins."""
    return sum(len(re.findall(r"[A-Za-z_]\w*|\d+", c)) for c, _ in rules)


def selection_key(f: Dict) -> Tuple:
    """Certified on the most training members (deterministic, then with `any`), then the largest smallest margin
    (with `any`, then deterministic), then the simplest list."""
    return (f["det_score"][0], f["score"][0], round(f["score"][2], 9), round(f["det_score"][2], 9), -size(f["rules"]))


def determinize(rules, members: List[Member]):
    """Resolve every `any` by one global action order (the first allowed action in the order is taken): the order
    whose deterministic list does best on the training members. A list certified with `any` stays certified under
    every order, since the worst case already covers each resolution."""
    from itertools import permutations
    from core.rules import fixed_order
    if not any(a == "any" for _, a in rules):
        return list(rules), None, score(evaluate(rules, members))
    best = None
    for order in permutations(ACTIONS):
        det = fixed_order(rules, order)
        sc = score(evaluate(det, members))
        if best is None or (sc[0], sc[2]) > (best[2][0], best[2][2]):
            best = (det, list(order), sc)
    return best


def finalize(members: List[Member], attempts: List[Dict], top: int, steps: int) -> Tuple[Dict, List[Dict]]:
    """Repair the `top` best distinct lists on the training members; keep the one certified on the most members
    with the largest smallest margin."""
    from fresh.repair import repair
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
        if a.get("order"):
            from core.rules import fixed_order
            rules = fixed_order(rules, a["order"])   # in-loop resolution: the candidate is the resolved list
        if steps:
            rules, _ = repair(rules, members, steps=steps, seed=i, hidden=CFG.hidden)
        results = evaluate(rules, members)
        det, order, det_score = determinize(rules, members)
        finals.append({"from": [a["chain"], a["round"]], "llm_score": a["score"], "rules": rules,
                       "score": score(results), "worst": [r.worst for r in results],
                       "det_rules": det, "det_order": order, "det_score": det_score})
    finals.sort(key=selection_key, reverse=True)
    return (finals[0] if finals else None), finals


def run(fam: str, chains: int, rounds: int, out: Path, position: bool, ids: Optional[Sequence[int]] = None,
        repair_top: int = 8, repair_steps: int = 40, phases: bool = False):
    out.mkdir(parents=True, exist_ok=True)
    members = load(ids or FAMILIES[fam], CFG.data, CFG.hidden)
    for m in members:
        if phases:
            m.all_phases()
        m.feature_values   # build once, before threads share the members
    start = time.time()
    with ThreadPoolExecutor(max_workers=chains) as pool:
        futures = [pool.submit(chain, fam, members, position, rounds, out, f"{fam}_c{i}") for i in range(chains)]
        attempts = [a for f in futures for a in f.result()]
    llm_seconds = time.time() - start
    usage = {k: sum(a["usage"].get(k, 0) for a in attempts) for k in ("calls", "prompt_tokens", "output_tokens", "cost")}
    final, finals = finalize(members, attempts, repair_top, repair_steps)
    valid = [a for a in attempts if a.get("score")]
    summary = {"family": fam, "chains": chains, "rounds": rounds, "position": position, "phases": phases,
               "settings": asdict(CFG),
               "ids": [m.id for m in members], "repair_top": repair_top, "repair_steps": repair_steps,
               "llm_seconds": round(llm_seconds, 1), "seconds": round(time.time() - start, 1),
               "usage": usage, "process_usage": dict(llm.USAGE), "attempts": len(attempts), "valid": len(valid),
               "certified_llm_lists": sum(1 for a in valid if a["score"][0] == len(members)),
               "certified_final_lists": sum(1 for f in finals if f["score"][0] == len(members)),
               "best": final, "finals": finals}
    (out / f"{fam}_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", help="a family name (rooms, crossing, adjacent) or a label with --ids")
    ap.add_argument("--chains", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--position", action="store_true", help="allow x, y and the grid constants in conditions")
    ap.add_argument("--ids", default="", help="training member ids (default: the family's three)")
    ap.add_argument("--repair-top", type=int, default=8)
    ap.add_argument("--repair-steps", type=int, default=40)
    ap.add_argument("--phases", action="store_true", help="check from every initial obstacle phase")
    ap.add_argument("--kind", choices=["family", "random"], default="family")
    ap.add_argument("--data", default="families_v0.csv")
    ap.add_argument("--hide", default="hazard", help="comma-separated sensors to leave out")
    ap.add_argument("--resolve-in-loop", action="store_true")
    ap.add_argument("--no-action-lines", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    CFG.kind, CFG.data, CFG.hidden = args.kind, args.data, tuple(h for h in args.hide.split(",") if h)
    CFG.resolve_in_loop, CFG.action_lines = args.resolve_in_loop, not args.no_action_lines
    ids = [int(i) for i in args.ids.split(",")] if args.ids else None
    s = run(args.family, args.chains, args.rounds, Path(args.out), args.position, ids, args.repair_top,
            args.repair_steps, args.phases)
    b = s["best"]
    print(json.dumps({k: v for k, v in s.items() if k not in ("best", "finals")}, default=str))
    if b:
        print("best score", b["score"], "from", b["from"], "llm score", b["llm_score"])
        for c, a in b["rules"]:
            print(f"  {c} -> {a}")


if __name__ == "__main__":
    main()
