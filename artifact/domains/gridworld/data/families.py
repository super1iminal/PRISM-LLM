"""Gridworld families: parameterized layouts whose structure scales with the grid size n.

A family is a function (n, rng) -> layout (goals, static obstacles, moving-obstacle waypoints). The size gives the
structure (walls, doors, goal corners); the seed gives small variations (a door's offset, scattered obstacles), so
a family has many members per size. Every member keeps the start at (0, 0), three goals and one moving obstacle.
Rows use datagen.py's dataset format (n, grid, goals, static, moving, BFS_steps), so domains/gridworld reads
them unchanged; extra columns `family` and `seed` say where a row came from.

Families:
  rooms     a wall splits the grid into a left room and two right rooms, each passage a one-cell door; the
            moving obstacle patrols in front of the left door. Goals: top-right room, bottom-right room, then
            back to the bottom-left. Moving straight at a goal runs into a wall.
  crossing  a wall across the middle with a one-cell gap; the moving obstacle patrols along the gap's upper side.
            Goals: bottom-right, top-right, bottom-left, so the agent crosses three times.
  adjacent  goals 1 and 3 next to each other on the bottom row, goal 2 on the right edge behind a patrol: leaving
            goal 1 upwards can slip into goal 3 before its turn.

Run from artifact/:
    python domains/gridworld/data/families.py rooms crossing adjacent --sizes 6-20 --seeds 0-2 --out families.csv
"""
import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datagen import is_solvable_sequence_with_path  # noqa: E402

Cell = Tuple[int, int]
Layout = Dict[str, object]   # goals {k: cell}, static [cell], moving [waypoint]
START = (0, 0)


def rooms(n: int, rng: random.Random) -> Layout:
    col, row = n // 2, n // 2
    door_v = max(1, n // 4 + rng.choice([0, 1]))
    door_h = col + 1 + (n - col - 1) // 2 + rng.choice([-1, 0])
    static = [(r, col) for r in range(n) if r != door_v]
    static += [(row, c) for c in range(col + 1, n) if c != door_h]
    goals = {1: (1, n - 2), 2: (n - 2, n - 2), 3: (n - 2, 1)}
    moving = [(door_v - 1, col - 1), (door_v, col - 1), (door_v + 1, col - 1)]
    return {"goals": goals, "static": static, "moving": moving, "keep_clear": {(door_v, col - 1), (door_v, col + 1),
            (row - 1, door_h), (row + 1, door_h)}}


def crossing(n: int, rng: random.Random) -> Layout:
    wall = n // 2
    gap = n // 2 + rng.choice([-1, 0, 1])
    static = [(wall, c) for c in range(n) if c != gap]
    goals = {1: (n - 1, n - 1), 2: (0, n - 1), 3: (n - 1, 0)}
    moving = [(wall - 1, gap - 1), (wall - 1, gap), (wall - 1, gap + 1)]
    return {"goals": goals, "static": static, "moving": moving, "keep_clear": {(wall + 1, gap), (wall - 2, gap)}}


def adjacent(n: int, rng: random.Random) -> Layout:
    m = n // 2 + rng.choice([-1, 0])
    edge = n // 2 + rng.choice([-1, 0])
    goals = {1: (n - 1, m), 2: (edge, n - 1), 3: (n - 1, m + 1)}
    moving = [(edge - 1, n - 2), (edge, n - 2), (edge + 1, n - 2)]
    return {"goals": goals, "static": [], "moving": moving, "keep_clear": {(n - 1, m - 1), (n - 2, m)}}


FAMILIES: Dict[str, Callable[[int, random.Random], Layout]] = {"rooms": rooms, "crossing": crossing,
                                                                "adjacent": adjacent}


def patrol_cells(moving: List[Cell]) -> List[Cell]:
    """The moving obstacle's cells (it walks the waypoints back and forth)."""
    return list(dict.fromkeys(moving))


def scatter(n: int, layout: Layout, rng: random.Random, density: float) -> List[Cell]:
    """Random extra static obstacles (about `density` of the free cells) that keep every goal reachable in order
    and leave the start, goals, patrol and the layout's `keep_clear` cells free."""
    reserved = {START} | set(layout["goals"].values()) | set(layout["static"]) | set(patrol_cells(layout["moving"]))
    reserved |= set(layout.get("keep_clear", ())) | {(0, 1), (1, 0)}
    free = [(r, c) for r in range(n) for c in range(n) if (r, c) not in reserved]
    rng.shuffle(free)
    extra: List[Cell] = []
    for cell in free[:round(density * len(free))]:
        ok, _ = is_solvable_sequence_with_path(n, START, layout["goals"], layout["static"] + extra + [cell])
        if ok:
            extra.append(cell)
    return extra


def grid_text(n: int, goals: Dict[int, Cell], static: List[Cell], moving: List[Cell]) -> str:
    """The dataset's ASCII grid (as datagen.py writes it)."""
    marks = {c: "M" for c in moving}
    marks.update({c: "X" for c in static})
    marks.update({c: str(k) for k, c in goals.items()})
    marks[START] = "S"
    rows = ["  " + " ".join(str(c) for c in range(n))]
    rows += [f"{r} " + " ".join(marks.get((r, c), ".") for c in range(n)) for r in range(n)]
    return "\n".join(rows)


def member(family: str, n: int, seed: int, density: float) -> dict:
    """One instance of `family` at size `n`, varied by `seed`, as a dataset row."""
    rng = random.Random(f"{family}-{n}-{seed}")
    layout = FAMILIES[family](n, rng)
    static = sorted(set(layout["static"]) | set(scatter(n, layout, rng, density)))
    goals = layout["goals"]
    ok, path = is_solvable_sequence_with_path(n, START, goals, static)
    if not ok:
        raise ValueError(f"{family} n={n} seed={seed}: goals not reachable in order")
    if set(goals.values()) & (set(static) | set(layout["moving"])) or START in static:
        raise ValueError(f"{family} n={n} seed={seed}: a goal or the start is blocked")
    return {"n": n, "grid": grid_text(n, goals, static, layout["moving"]).replace("\n", "\\n"), "goals": goals,
            "static": static, "moving": layout["moving"], "BFS_steps": len(path) - 1, "family": family, "seed": seed}


def ids(spec: str) -> List[int]:
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("families", nargs="+", choices=sorted(FAMILIES))
    parser.add_argument("--sizes", default="6-20", help='e.g. "6-20" or "6,8,10"')
    parser.add_argument("--seeds", default="0-2")
    parser.add_argument("--density", type=float, default=0.08, help="share of free cells given extra obstacles")
    parser.add_argument("--out", required=True, help="CSV file name (written next to this script)")
    args = parser.parse_args(argv)
    rows = [member(f, n, s, args.density) for f in args.families for n in ids(args.sizes) for s in ids(args.seeds)]
    path = Path(__file__).parent / args.out
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} instances -> {path}")


if __name__ == "__main__":
    main()
