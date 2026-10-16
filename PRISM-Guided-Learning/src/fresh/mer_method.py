"""The same method on MER: one scheduler rule list for all training instances (three values of x, or of n), revised
from exact worst-case feedback, repaired, the largest-margin list kept, `any` resolved by an action order.

    python src/fresh/mer_method.py --dataset mer_transfer.csv --chains 8 --rounds 6 --out out/fresh/mer/<name>
"""
import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jinja2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.rules import RuleError, fixed_order  # noqa: E402
from fresh import llm  # noqa: E402
from fresh.mer import ACTIONS, CODES, ROOT, MerInstance, load  # noqa: E402

TRAIN = [0, 1, 2]
LITERAL_SUM = re.compile(r"(?<!\w)\d+\s*[-+]\s*\d+(?!\w)")


def spec_text(n: int, x: float) -> Dict:
    env = jinja2.Environment(loader=jinja2.DictLoader({
        "spec": (ROOT / "domains" / "mer" / "spec.yaml.j2").read_text(encoding="utf-8"), "_vocabulary.yaml.j2": ""}),
        trim_blocks=True, lstrip_blocks=True)
    return yaml.safe_load(env.get_template("spec").render(n=n, x=x, err_threshold=0))


def problem(insts: List[MerInstance], dataset: str) -> str:
    first = insts[0]
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(ROOT / "domains" / "mer")), trim_blocks=True,
                             lstrip_blocks=True)
    desc = env.get_template("description.md.j2").render(x=first.x, n=first.n)
    visual = env.get_template("visual.txt.j2").render()
    spec = spec_text(first.n, first.x)
    varying = ("x (the probability that a user keeps a resource after using it)" if "n_" not in dataset
               else "N (how many of user 1's comm requests may take the arbiter's shortcut)")
    train = ", ".join(f"{p.name} (n = {p.n}, x = {p.x}; required >= {p.threshold}, best possible {p.optimum_value:.4f})"
                      for p in insts)
    codes = ", ".join(f"`{name}` = {value}" for name, value, _ in CODES)
    return f"""You write ONE scheduler, as an ordered list of rules, for a family of instances of the system below that
differ in {varying}. The same list, unchanged, is checked on the training instances {train} and on other
values (smaller and larger) of the same parameter, so the rules must not depend on the exact value.

{desc}

{visual}

## Actions (every step of the system is one of them; an action that is not possible leaves the state unchanged)
{chr(10).join(f"- `{a}`: {d}" for a, d in spec["actions"].items())}

## Requirement (checked exactly, in the WORST case)
The probability that the monitor eventually reaches e = 3 must be at least the instance's threshold (95% of the best
possible). Wherever no rule applies, or a rule with action `any` allows several actions, an adversary picks among
the allowed actions, including ones that are not possible (which stall the system forever). So rules must name
possible actions (use `enabled`), and every action a rule allows must lead towards the error.

## Rule language
An ordered list of rules {{"condition": ..., "action": ...}}; in each state the FIRST rule whose condition holds
decides. Actions: the names above, or `any` (allows every action for which the condition holds, with the
per-action sensor evaluated for that action). Conditions use ! & | => ( ), = != < <= > >=, + -, and only the numbers
0 and 1, which may not be added together (use the named codes instead). Names:
- variables (compare them with the codes, never with numbers above 1):
{chr(10).join(f"  - {line}" for line in VARIABLES)}
- codes: {codes}; `N` and `X_PCT` (the instance's n and x in percent)
- `shortcuts_left` (= N - k); per-action sensor `enabled`: the action is possible here
End the list with a catch-all rule (condition `true`)."""


VARIABLES = [
    "`s`: the arbiter: 0 idle, `USER1` or `USER2` serving that user's request, -1 about to grant comm to user 1 at once",
    "`r`: the user the arbiter asked to give up drive (0 = nobody, `USER1`, `USER2`)",
    "`k`: user 1's comm requests made while drive was held (at most `N`; `shortcuts_left` = N - k)",
    "`commUser`, `driveUser`: the user holding comm / drive (0 = free, `USER1`, `USER2`)",
    "`gc`, `gd`, `dc`, `dd` (bool): a comm grant / drive grant / comm denial / drive denial is pending",
    "`s1`, `s2`: user 1's / user 2's status: `IDLE`, `READY`, `WAITING`, `HOLDING`, `MUST_CANCEL`",
    "`r1`, `r2`: the resource user 1 / user 2 wants or holds: 0 none, `COMM`, `DRIVE`",
    "`rescind1`, `rescind2` (bool): user 1 / user 2 has been asked to give up drive",
    "`e`: the monitor: 0 nothing granted, `COMM` comm granted, `DRIVE` drive granted, `BOTH` both at once (the error)",
    "`u`: the monitor's user holding the watched resource (0 = none, `USER1`, `USER2`)",
]


def initial_prompt(head: str) -> str:
    return head + """

## Answer
Think about which order of user and arbiter steps makes the error (both resources granted at once) most likely,
in every instance. Keep your explanation short (a few lines), then give the complete list as JSON:
{"rules": [{"condition": "...", "action": "..."}, ...]}"""


def revise_prompt(head: str, rules, report: str, history: str) -> str:
    listing = "\n".join(f'{i}. {{"condition": {json.dumps(c)}, "action": {json.dumps(a)}}}'
                        for i, (c, a) in enumerate(rules, start=1))
    return head + f"""

## Your current list
{listing}

## Its exact worst-case results on the training instances
{report}
{history}
Revise the list so that the requirement holds on every training instance, keeping the rules general. Keep your
explanation short, then give the complete revised list as JSON:
{{"rules": [{{"condition": "...", "action": "..."}}, ...]}}"""


def evaluate(rules, insts) -> List[float]:
    return [p.worst(rules) for p in insts]


def score(values: List[float], insts) -> Tuple:
    margins = [v - p.threshold for v, p in zip(values, insts)]
    return (sum(m >= -1e-12 for m in margins), -round(sum(max(0.0, -m) for m in margins), 6), min(margins))


OPT: Dict[str, np.ndarray] = {}


def enabled_matrix(p: MerInstance) -> np.ndarray:
    """S x actions: whether each action is possible in each state (the `enabled` sensor)."""
    if not hasattr(p, "_enabled"):
        cache: Dict = {}
        p._enabled = np.stack([np.broadcast_to(p.veval(p.scope.action_features["enabled"][a], cache), (p.S,))
                               for a in ACTIONS], axis=1).astype(bool)
    return p._enabled


def blame(p: MerInstance, rules, top: int = 4) -> str:
    """States where the worst case loses the most error probability compared with the best possible scheduler,
    weighted by how often the worst case reaches them: the rule that decided, the action the worst case takes
    there (and whether it is possible at all), and the best actions."""
    if p.name not in OPT:
        OPT[p.name] = p.values(np.ones(len(p.ch_state), bool), maximize=True)
    opt = OPT[p.name]
    allowed, decider = p.allowed(rules)
    mask = p.choice_mask(allowed)
    v = p.values(mask)
    q = np.zeros(len(p.ch_state))
    np.add.at(q, p.tr_choice, p.tr_prob * v[p.tr_target])
    qo = np.zeros(len(p.ch_state))
    np.add.at(qo, p.tr_choice, p.tr_prob * opt[p.tr_target])
    qm = np.where(mask, q, 2.0)
    order = np.lexsort((qm, p.ch_state))
    first = np.ones(len(order), bool)
    first[1:] = p.ch_state[order][1:] != p.ch_state[order][:-1]
    pick = np.full(p.S, -1)
    pick[p.ch_state[order][first]] = order[first]
    sel = pick[p.ch_state[p.tr_choice]] == p.tr_choice
    visit, dist = np.zeros(p.S), np.zeros(p.S)
    dist[p.init] = 1.0
    for _ in range(300):
        live = dist * ~p.err
        visit += live
        nd = np.zeros(p.S)
        np.add.at(nd, p.tr_target[sel], live[p.ch_state[p.tr_choice[sel]]] * p.tr_prob[sel])
        dist = nd
    loss = np.minimum(visit, 1.0) * (opt - v)
    en = enabled_matrix(p)
    pol = p.ch_action >= 0
    qoa = np.full((p.S, len(ACTIONS)), np.nan)
    qoa[p.ch_state[pol], p.ch_action[pol]] = qo[pol]
    names = [var.name for var in p.variables]
    lines, seen = [], set()
    for s in np.argsort(-loss):
        if len(lines) >= top or loss[s] < 1e-4:
            break
        rule = int(decider[s])
        if (rule, int(p.ch_action[pick[s]])) in seen:
            continue
        seen.add((rule, int(p.ch_action[pick[s]])))
        state = ", ".join(f"{nm}={int(p.state_vars[nm][s])}" for nm in names)
        who = f"rule {rule + 1} `{rules[rule][0]} -> {rules[rule][1]}`" if rule >= 0 else "NO RULE (uncovered)"
        a = int(p.ch_action[pick[s]])
        took = f"{ACTIONS[a]}" + ("" if en[s, a] else " (not possible here: nothing happens, and it can repeat forever)")
        best = sorted((i for i in range(len(ACTIONS)) if en[s, i]), key=lambda i: -qoa[s, i])[:3]
        lines.append(f"    * state {state}: error probability {v[s]:.3f} with your rules, {opt[s]:.3f} at best; decided "
                     f"by {who}; the worst case takes {took}; best actions here: "
                     + ", ".join(f"{ACTIONS[i]} ({qoa[s, i]:.3f})" for i in best))
    return chr(10).join(lines)


def report(rules, insts, values) -> str:
    out = []
    for p, v in zip(insts, values):
        ok = v >= p.threshold - 1e-12
        out.append(f"### {p.name}: error probability {v:.4f} ({'OK' if ok else 'TOO LOW'}; needs >= {p.threshold})")
        if not ok:
            out.append(blame(p, rules))
    return "\n".join(out)


def validate(rules, insts) -> Optional[str]:
    if not rules:
        return "no rules found in the answer"
    bad = [f"rule {i} ({c} -> {a})" for i, (c, a) in enumerate(rules, start=1) if LITERAL_SUM.search(c)]
    if bad:
        return "adding or subtracting number literals (e.g. `1+1`) is not allowed; use the named codes: " + "; ".join(bad)
    try:
        insts[0].policy(rules)
    except RuleError as e:
        return str(e)
    return None


def chain(insts, head: str, rounds: int, log: Path, tag: str) -> List[Dict]:
    attempts, best, history_lines = [], None, []
    prompt = initial_prompt(head)
    for rnd in range(rounds + 1):
        text, usage = llm.chat([{"role": "user", "content": prompt}], max_tokens=32000)
        rules = llm.parse_rules(text)
        err = validate(rules, insts)
        entry = {"chain": tag, "round": rnd, "rules": rules, "error": err, "usage": usage}
        if err is None:
            values = evaluate(rules, insts)
            entry["score"], entry["values"] = score(values, insts), values
            if best is None or entry["score"] > best["score"]:
                best = entry
            history_lines.append(f"- attempt {rnd + 1}: error probability " + ", ".join(f"{v:.4f}" for v in values))
        else:
            history_lines.append(f"- attempt {rnd + 1}: rejected ({err.splitlines()[0][:200]})")
        attempts.append(entry)
        with open(log / f"{tag}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({**entry, "prompt": prompt, "answer": text}) + "\n")
        if best is not None and best["score"][0] == len(insts) and best["score"][2] > 0.002:
            break
        if best is None:
            prompt = initial_prompt(head) + (f"\n\nYour previous answer was rejected:\n{err}\nFix this." if err else "")
            continue
        history = "\n## Your attempts so far\n" + "\n".join(history_lines) + "\n" if len(history_lines) > 1 else ""
        prompt = revise_prompt(head, best["rules"], report(best["rules"], insts, best["values"]), history)
    return attempts


LITERALS = (["enabled", "!enabled", "shortcuts_left = 0", "shortcuts_left > 0", "s = 0", "s = USER1", "s = USER2",
             "s = -1", "r = 0", "r != 0", "commUser = 0", "commUser != 0", "driveUser = 0", "driveUser != 0",
             "gc", "!gc", "gd", "!gd", "dc", "!dc", "dd", "!dd", "e = 0", "e = COMM", "e = DRIVE", "rescind1",
             "!rescind1", "rescind2", "!rescind2"]
            + [f"s{i} = {c}" for i in (1, 2) for c in ("IDLE", "READY", "WAITING", "HOLDING", "MUST_CANCEL")]
            + [f"r{i} = {c}" for i in (1, 2) for c in ("0", "COMM", "DRIVE")])


def repair(rules, insts, steps: int = 25, tries: int = 40, seed: int = 0):
    rng = random.Random(seed)

    def objective(rs):
        try:
            return sum(min(v - p.threshold, 0.005) for v, p in zip(evaluate(rs, insts), insts))
        except RuleError:
            return None

    cur, cur_v = list(rules), objective(rules)
    if cur_v is None:
        return cur
    for _ in range(steps):
        if cur_v >= 0.005 * len(insts) - 1e-12:
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
            v = objective(new)
            if v is not None and v > cur_v + 1e-9:
                cur, cur_v, improved = new, v, True
                break
        if not improved:
            break
    return cur


def determinize(rules, insts, orders: int = 24, seed: int = 0):
    """`any` resolved by the best of `orders` action orders (the declared one and random ones; 27! are too many)."""
    if not any(a == "any" for _, a in rules):
        return list(rules), None, score(evaluate(rules, insts), insts)
    rng = random.Random(seed)
    cands = [list(ACTIONS)] + [rng.sample(ACTIONS, len(ACTIONS)) for _ in range(orders - 1)]
    best = None
    for order in cands:
        det = fixed_order(rules, order)
        sc = score(evaluate(det, insts), insts)
        if best is None or (sc[0], sc[2]) > (best[2][0], best[2][2]):
            best = (det, order, sc)
    return best


def size(rules) -> int:
    return sum(len(re.findall(r"[A-Za-z_]\w*|\d+", c)) for c, _ in rules)


def run(dataset: str, chains: int, rounds: int, out: Path, top: int = 6, repair_steps: int = 25):
    out.mkdir(parents=True, exist_ok=True)
    insts = load(TRAIN, dataset)
    head = problem(insts, dataset)
    start = time.time()
    with ThreadPoolExecutor(max_workers=chains) as pool:
        futures = [pool.submit(chain, insts, head, rounds, out, f"mer_c{i}") for i in range(chains)]
        attempts = [a for f in futures for a in f.result()]
    llm_seconds = time.time() - start
    usage = {k: sum(a["usage"].get(k, 0) for a in attempts) for k in ("calls", "prompt_tokens", "output_tokens", "cost")}
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
        values = evaluate(rules, insts)
        det, order, det_score = determinize(rules, insts)
        finals.append({"from": [a["chain"], a["round"]], "llm_score": a["score"], "rules": rules,
                       "score": score(values, insts), "values": values, "det_rules": det, "det_order": order,
                       "det_score": det_score})
    finals.sort(key=lambda f: (f["det_score"][0], f["score"][0], round(f["score"][2], 9),
                               round(f["det_score"][2], 9), -size(f["rules"])), reverse=True)
    summary = {"dataset": dataset, "chains": chains, "rounds": rounds, "llm_seconds": round(llm_seconds, 1),
               "seconds": round(time.time() - start, 1), "usage": usage, "process_usage": dict(llm.USAGE), "attempts": len(attempts),
               "valid": len(valid), "best": finals[0] if finals else None, "finals": finals}
    (out / "mer_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mer_transfer.csv")
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s = run(args.dataset, args.chains, args.rounds, Path(args.out))
    print(json.dumps({k: v for k, v in s.items() if k not in ("best", "finals")}, default=str))
    if s["best"]:
        print("best", s["best"]["score"], s["best"]["values"], "det", s["best"]["det_score"])
