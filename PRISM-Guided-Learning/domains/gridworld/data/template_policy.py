"""How far does one hand-written policy over relative features get? Certifies it on every grid of a dataset.

The template is the same on every grid. Its decision depends only on features that move with the instance:
the direction to the next unreached goal (sign of the row and column offsets, and which offset is larger)
and whether each neighbouring cell is blocked. Static obstacles, the moving obstacle's patrol cells and
goals that must not be entered yet all count as blocked. Variant `greedy` takes the first free move in
the order: towards the goal along the longer offset, along the shorter offset, sideways, away. Variant
`cautious` first prefers free moves whose slip cells are not hazards (patrol cells, future goals), using
the same order. Both are ~10-rule decision lists over those features; for verification each grid's
instance of the template is compiled to one rule per (segment, action) over x, y and the goal flags.

Run from PRISM-Guided-Learning/:
    python domains/gridworld/data/template_policy.py grid_20_balanced.csv grid_20_large.csv
Prints one line per grid and a summary per dataset and variant, and writes <dataset stem>_template.csv
next to this file.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from config import load_config  # noqa: E402
from core.domain import load_domain  # noqa: E402
from core.rules import SymbolicPolicy  # noqa: E402
from core.verifier import PolicyVerifier  # noqa: E402
from domains.gridworld.domain import MOVES, SLIP_LEFT, SLIP_RIGHT, expand_cycle  # noqa: E402

OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
VARIANTS = ("greedy", "cautious")
NEAR = 0.03   # re-check with interval iteration when a worst case is this close to its threshold


def _step(cell, action):
    return cell[0] + MOVES[action][0], cell[1] + MOVES[action][1]


def template_action(n, cell, target, blocked, hazards, variant):
    """The template's move from `cell` towards `target`, given the blocked and hazardous cells."""
    dr, dc = target[0] - cell[0], target[1] - cell[1]
    vert = "down" if dr > 0 else "up" if dr < 0 else None
    hor = "right" if dc > 0 else "left" if dc < 0 else None
    toward = [a for a in ((vert, hor) if abs(dr) >= abs(dc) else (hor, vert)) if a]
    rest = [a for a in ("down", "right", "up", "left") if a not in toward]
    sideways = [a for a in rest if OPPOSITE[a] not in toward]
    order = toward + sideways + [a for a in rest if a not in sideways]

    def free(a):
        r, c = _step(cell, a)
        return 0 <= r < n and 0 <= c < n and (r, c) not in blocked

    def safe(a):   # the move's slip outcomes do not enter a hazard
        return all(_step(cell, s) not in hazards for s in (SLIP_LEFT[a], SLIP_RIGHT[a]))

    candidates = [a for a in order if free(a)] or order
    if variant == "cautious":
        candidates = [a for a in candidates if safe(a)] or candidates
    return candidates[0]


def template_rules(data, variant):
    """The template instantiated on one grid: one (condition, action) rule per segment and action."""
    n, goals = data["n"], data["goals"]
    ks = sorted(goals)
    patrol = set(expand_cycle(data["moving"]))
    rules = []
    for i, k in enumerate(ks):
        future = {goals[j] for j in ks[i + 1:]}
        blocked = set(map(tuple, data["static"])) | patrol | future
        hazards = patrol | future
        segment = " & ".join([f"g{j}" for j in ks[:i]] + [f"!g{k}"])
        cells = {}
        for r in range(n):
            for c in range(n):
                if (r, c) not in set(map(tuple, data["static"])):
                    a = template_action(n, (r, c), goals[k], blocked, hazards, variant)
                    cells.setdefault(a, []).append(f"(x={r} & y={c})")
        rules += [(f"{segment} & ({' | '.join(cs)})", a) for a, cs in cells.items()]
    return rules + [("true", "up")]   # every goal reached: any move


def main(*datasets):
    domain = load_domain("gridworld", ["obs_idx"])
    cfg = load_config()
    for dataset in datasets:
        rows = []
        for inst in domain.load_instances(dataset):
            verifier = PolicyVerifier(domain, inst, cfg)
            reqs = verifier.spec.requirements
            for variant in VARIANTS:
                policy = SymbolicPolicy.from_raw(verifier.spec.variables, list(verifier.spec.actions),
                                                 template_rules(inst.data, variant))
                v = verifier.verify(policy, analysis=False)
                if any(abs(v.worst[r.name] - r.threshold) < NEAR for r in reqs):   # verdict could flip: exact check
                    v = verifier.verify_exact(policy)[0] or v
                met = [r.name for r in reqs if r.satisfied(v.worst[r.name])]
                rows.append({"dataset": dataset, "grid": int(inst.id), "n": inst.data["n"], "variant": variant,
                             "met": len(met), "requirements": len(reqs), "solved": len(met) == len(reqs),
                             **{f"worst_{r.name}": v.worst[r.name] for r in reqs}})
                print(f"{dataset} grid {inst.id} ({inst.data['n']}x{inst.data['n']}) {variant}: {len(met)}/{len(reqs)} met"
                      + ("" if len(met) == len(reqs) else "; fails " + ", ".join(
                          f"{r.name}={v.worst[r.name]:.3f}" for r in reqs if r.name not in met)), flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(Path(__file__).parent / f"{Path(dataset).stem}_template.csv", index=False)
        for variant, g in df.groupby("variant"):
            print(f"== {dataset} {variant}: solved {int(g.solved.sum())}/{len(g)}, requirements met {g.met.mean():.2f}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["grid_20_balanced.csv", "grid_20_large.csv"]))
