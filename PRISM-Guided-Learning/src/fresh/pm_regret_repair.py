"""Pac-Man, no LLM: hill-climb a kept list to lower its uniform regret (mean excess crash probability over every
decision state of the longest training game) while it stays certified on all training games.

    python src/fresh/pm_regret_repair.py out/fresh/pm/<run>/pacman_summary.json --out <file.json>
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.rules import RuleError  # noqa: E402
import fresh.pm_method as pmm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("summary")
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--tries", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

insts = pmm.load(pmm.TRAIN)
longest = insts[-1]
opt = pmm.optimum_values(longest)
best = json.loads(Path(args.summary).read_text(encoding="utf-8"))["best"]
rules = [tuple(r) for r in best["det_rules"]]
rng = random.Random(args.seed)


def regret(rs):
    try:
        return pmm.uniform_regret(longest, rs, opt)
    except RuleError:
        return None


def certified(rs):
    return all(c <= pmm.THRESHOLD + 1e-12 for c in pmm.evaluate(rs, insts))


cur, cur_r = rules, regret(rules)
print("start", round(cur_r, 5), flush=True)
for step in range(args.steps):
    improved = False
    for _ in range(args.tries):
        i = rng.randrange(len(cur))
        c, a = cur[i]
        kind = rng.choice(["drop", "add", "action", "specialize", "specialize"])
        new = list(cur)
        if kind == "drop" and i < len(cur) - 1:
            del new[i]
        elif kind == "add":
            new[i] = (f"{c} & {rng.choice(pmm.LITERALS)}" if c != "true" else rng.choice(pmm.LITERALS), a)
        elif kind == "action":
            new[i] = (c, rng.choice([x for x in pmm.ACTIONS if x != a]))
        elif kind == "specialize":
            lit = rng.choice(pmm.LITERALS)
            new.insert(i, (f"{c} & {lit}" if c != "true" else lit, rng.choice(pmm.ACTIONS)))
        else:
            continue
        r = regret(new)
        if r is not None and r < cur_r - 1e-6 and certified(new):
            cur, cur_r, improved = new, r, True
            print(f"step {step}: {cur_r:.5f}", flush=True)
            break
    if not improved:
        break
final = {"rules": cur, "regret_longest": cur_r, "crash": pmm.evaluate(cur, insts),
         "regret": [pmm.uniform_regret(p, cur, pmm.optimum_values(p)) for p in insts], "from": best["from"]}
Path(args.out).write_text(json.dumps(final, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in final.items() if k != "rules"}))
