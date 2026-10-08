"""Pac-Man case study: avoid two randomly moving ghosts for a fixed number of moves.

The MDP is the Quantitative Verification Benchmark Set's Pac-Man model (benchmarks/mdp/pacman,
version 2; Junges and Koenighofer, from "Safe Reinforcement Learning via Probabilistic Shields",
arXiv:1807.06096), unchanged except for the planning horizon MAXSTEPS, which each instance sets.
Pac-Man only decides at the eight crossings; corridors and corners are forced moves (label `[p]`,
which the policy module does not synchronize on). The ghosts move after Pac-Man, with learned
probabilities at crossings that depend on their heading and on where Pac-Man is.

Dataset rows (CSV): name, max_steps, crash_threshold.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from core.domain import Domain, Instance

# Walkable cells per row, top (y = 6) to bottom (y = 1), as in the model's corridor commands
ROWS = {6: "...#......", 5: "##.#.####.", 4: "..........", 3: ".#.##.#.#.", 2: ".#.##.#.#.", 1: ".........."}
CROSSINGS = [(3, 1), (6, 1), (8, 1), (3, 4), (5, 4), (6, 4), (8, 4), (10, 4)]
START = (1, 1)
GHOST_START = (3, 4)
DIRECTIONS = {0: "right", 1: "up", 2: "left", 3: "down"}   # the model's dP/dG codes


class PacMan(Domain):

    def load_instances(self, dataset: str) -> List[Instance]:
        path = Path(dataset) if Path(dataset).is_absolute() else self.root / "data" / dataset
        df = pd.read_csv(path, index_col=False)
        return [Instance(id=str(idx), data={"name": str(row["name"]), "max_steps": int(row["max_steps"]),
                                            "crash_threshold": float(row["crash_threshold"])})
                for idx, row in df.iterrows()]

    def horizon(self, instance: Instance) -> Optional[int]:
        """Mass-analysis horizon: three transitions (Pac-Man, ghost 0, ghost 1) per Pac-Man move."""
        return 3 * instance.data["max_steps"] + 3

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
    def walkable() -> List[tuple]:
        """Every cell Pac-Man or a ghost can occupy."""
        return [(x + 1, y) for y, row in ROWS.items() for x, c in enumerate(row) if c == "."]

    def format_state(self, valuation: Dict[str, Any]) -> str:
        """Rule syntax, with heading names added."""
        return " & ".join(f"{k}={v}" + (f" ({DIRECTIONS[v]})" if k.startswith("d") and v in DIRECTIONS else "")
                          for k, v in valuation.items())


# ---------------------------------------------------------------- extended rule vocabulary

# At these crossings a direction is a wall: choosing it keeps Pac-Man in place (from the model's crossing commands)
WALLS = {"right": [(10, 4)], "up": [(6, 4), (8, 4)], "left": [], "down": [(3, 1), (6, 1), (8, 1), (5, 4)]}
TOWARDS = {"right": "xG{g} > xP", "left": "xG{g} < xP", "up": "yG{g} > yP", "down": "yG{g} < yP"}


def rule_vocabulary(max_steps: int) -> Dict[str, Any]:
    """Constant MAXSTEPS, the state feature moves_left and per-action features wall, nearer_g0, nearer_g1."""
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
    return {"rule_constants": [{"name": "MAXSTEPS", "value": max_steps, "description": "the game's length in moves"}],
            "rule_features": features}
