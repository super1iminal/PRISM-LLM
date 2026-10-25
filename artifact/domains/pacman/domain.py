"""Pac-Man case study: avoid two randomly moving ghosts for a fixed number of moves.

The MDP is the Quantitative Verification Benchmark Set's Pac-Man model (benchmarks/mdp/pacman,
version 2; Junges and Koenighofer, from "Safe Reinforcement Learning via Probabilistic Shields",
arXiv:1807.06096), unchanged except for the planning horizon MAXSTEPS, which each instance sets.
Pac-Man only decides at the eight crossings; corridors and corners are forced moves (label `[p]`,
which the policy module does not synchronize on). The ghosts move after Pac-Man, with learned
probabilities at crossings that depend on their heading and on where Pac-Man is.

Dataset rows (CSV): name, max_steps, crash_threshold.
"""
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from core.domain import Domain, Instance

# Walkable cells per row, top (y = 6) to bottom (y = 1), as in the model's corridor commands
ROWS = {6: "...#......", 5: "##.#.####.", 4: "..........", 3: ".#.##.#.#.", 2: ".#.##.#.#.", 1: ".........."}
CROSSINGS = [(3, 1), (6, 1), (8, 1), (3, 4), (5, 4), (6, 4), (8, 4), (10, 4)]
START = (1, 1)
GHOST_START = (3, 4)
DIRECTIONS = {0: "right", 1: "up", 2: "left", 3: "down"}   # the model's dP/dG codes
HEADINGS = ("dP", "dG0", "dG1")


class PacMan(Domain):

    def load_instances(self, dataset: str) -> List[Instance]:
        df = self.read_dataset(dataset)
        return [Instance(id=str(idx), data={"name": str(row["name"]), "max_steps": int(row["max_steps"]),
                                            "crash_threshold": float(row["crash_threshold"])})
                for idx, row in df.iterrows()]

    def horizon(self, instance: Instance) -> Optional[int]:
        """Mass-analysis horizon: three transitions (Pac-Man, ghost 0, ghost 1) per Pac-Man move."""
        return 3 * instance.data["max_steps"] + 3

    def strict(self, instance: Instance) -> Dict[str, Any]:
        """`rules.strict`: number literals may not be added, so conditions cannot name particular cells."""
        return {"hide": [], "literal_sums": False,
                "note": "Number literals may not be added or subtracted (no `1 + 1`): the rules cannot name particular "
                        "cells, so compare positions with each other (e.g. `xG0 < xP`, `yG1 = yP`) and use the "
                        "features."}

    def repair_literals(self, instance: Instance, vocabulary) -> List[str]:
        """The default and comparisons of Pac-Man's position with the ghosts', headings and removed ghosts."""
        return super().repair_literals(instance, vocabulary) + [
            f"{a} {op} {b}" for a, b in (("xG0", "xP"), ("yG0", "yP"), ("xG1", "xP"), ("yG1", "yP"))
            for op in ("<", ">", "=")] + ["dP = 0", "dP = 1", "dP > 1", "dP < 1", "xG0 = 0", "xG1 = 0", "xG0 != 0",
                                          "xG1 != 0"]

    def optimal_strategy(self, instance: Instance, config):
        """The crash requirement's optimal actions on the explicit model, what 1-2-3-Go!'s trees learn from."""
        from core.explicit import optimal_strategy
        return optimal_strategy(self, instance, config)

    def context(self, instance: Instance) -> Dict[str, Any]:
        d = dict(instance.data)
        d.update({
            "crossings": CROSSINGS,
            "start": START,
            "ghost_start": GHOST_START,
            "directions": DIRECTIONS,
            "map_rows": [(y, "".join(self._symbol(x + 1, y, c) for x, c in enumerate(row))) for y, row in ROWS.items()],
            **rule_vocabulary(d["max_steps"]),
        })
        return d

    @staticmethod
    def _symbol(x: int, y: int, cell: str) -> str:
        if cell == "#":
            return "#"
        if (x, y) == START:
            return "P"
        if (x, y) == GHOST_START:
            return "G"
        return "+" if (x, y) in CROSSINGS else "."

    @staticmethod
    def crossings() -> List[tuple]:
        """The cells where Pac-Man chooses a direction."""
        return list(CROSSINGS)

    @staticmethod
    def walkable() -> List[tuple]:
        """Every cell Pac-Man or a ghost can occupy."""
        return [(x + 1, y) for y, row in ROWS.items() for x, c in enumerate(row) if c == "."]

    def format_state(self, valuation: Dict[str, Any]) -> str:
        """Rule syntax, with heading names added."""
        return " & ".join(f"{k}={v}" + (f" ({DIRECTIONS[v]})" if k in HEADINGS else "") for k, v in valuation.items())


# ---------------------------------------------------------------- extended rule vocabulary

# At these crossings a direction is a wall: choosing it keeps Pac-Man in place (from the model's crossing commands)
WALLS = {"right": [(10, 4)], "up": [(6, 4), (8, 4)], "left": [], "down": [(3, 1), (6, 1), (8, 1), (5, 4)]}
TOWARDS = {"right": "xG{g} > xP", "left": "xG{g} < xP", "up": "yG{g} > yP", "down": "yG{g} < yP"}
STEP = {"right": (1, 0), "up": (0, 1), "left": (-1, 0), "down": (0, -1)}


def maze_distances() -> Dict[Tuple[int, int], Dict[Tuple[int, int], int]]:
    """Shortest number of moves between walkable cells, through the maze (BFS from each cell)."""
    cells = set(PacMan.walkable())
    dist = {}
    for start in cells:
        d, queue = {start: 0}, deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in STEP.values():
                nxt = (x + dx, y + dy)
                if nxt in cells and nxt not in d:
                    d[nxt] = d[(x, y)] + 1
                    queue.append(nxt)
        dist[start] = d
    return dist


def _closer(g: int, a: str, dist) -> str:
    """Condition: at a crossing, the first step in direction `a` shortens the maze distance to ghost g."""
    terms = []
    for x, y in CROSSINGS:
        step = (x + STEP[a][0], y + STEP[a][1])
        if step not in dist:
            continue
        cells = [q for q in dist[(x, y)] if dist[step][q] < dist[(x, y)][q]]
        if cells:
            terms.append(f"(xP = {x} & yP = {y} & (" + " | ".join(f"(xG{g} = {qx} & yG{g} = {qy})" for qx, qy in sorted(cells)) + "))")
    return " | ".join(terms) or "false"


def rule_vocabulary(max_steps: int) -> Dict[str, Any]:
    """Constant MAXSTEPS, the state feature moves_left and per-action features wall, nearer_g0, nearer_g1 (by rows
    and columns) and closer_g0, closer_g1 (along the maze)."""
    def wall(a):
        return " | ".join(f"(xP = {x} & yP = {y})" for x, y in WALLS[a]) or "false"

    features = [
        {"name": "moves_left", "description": "moves Pac-Man has left (MAXSTEPS - steps)", "expr": "MAXSTEPS - steps"},
        {"name": "wall", "description": "true if this direction is a wall at Pac-Man's crossing (he would stay put)",
         "per_action": {a: wall(a) for a in DIRECTIONS.values()}},
    ]
    for g in (0, 1):
        features.append({"name": f"nearer_g{g}",
                         "description": f"true if this direction goes towards ghost {g} (by its row or column), "
                                        f"while the ghost is in the game",
                         "per_action": {a: f"xG{g} > 0 & " + TOWARDS[a].format(g=g) for a in DIRECTIONS.values()}})
    dist = maze_distances()
    for g in (0, 1):
        features.append({"name": f"closer_g{g}",
                         "description": f"true if this direction's first step shortens Pac-Man's distance to ghost "
                                        f"{g} along the maze (the shortest path through the corridors)",
                         "per_action": {a: _closer(g, a, dist) for a in DIRECTIONS.values()}})
    return {"rule_constants": [{"name": "MAXSTEPS", "value": max_steps, "description": "the game's length in moves"}],
            "rule_features": features}
