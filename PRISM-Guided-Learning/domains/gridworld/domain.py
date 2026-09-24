"""Gridworld case study: sequential goals, static obstacles, a moving obstacle and slippery moves.

Dataset rows (CSV): n, goals {k: (row, col)}, static [(row, col)], moving [(row, col)] (optional),
BFS_steps. Mirrors the legacy model in src/legacy/prism_model.py, but as a PRISM MDP whose
actions are left open for the policy.
"""
import ast
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from core.domain import Domain, Instance

PROB_FORWARD = 0.7
PROB_SLIP_LEFT = 0.15
PROB_SLIP_RIGHT = 0.15

# Action label -> (row delta, col delta); slips are 90 degrees left/right of the intended move
MOVES = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
SLIP_LEFT = {"up": "left", "right": "up", "down": "right", "left": "down"}
SLIP_RIGHT = {"up": "right", "right": "down", "down": "left", "left": "up"}

GOAL_THRESHOLD = 0.8
SEQUENCE_THRESHOLD = 0.8
AVOID_THRESHOLD = 0.7


def expand_cycle(path: List) -> List:
    """Moving obstacle path -> repeating cycle (same rule as legacy GridWorld._expand_path)."""
    if len(path) <= 1:
        return list(path)
    if path[-1] == path[0]:
        return list(path[:-1])
    return list(path) + list(path[-2:0:-1])


def _parse(value, default):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return default if value is None or (isinstance(value, float) and pd.isna(value)) else value


class GridWorld(Domain):

    def load_instances(self, dataset: str) -> List[Instance]:
        path = Path(dataset) if Path(dataset).is_absolute() else self.root / "data" / dataset
        df = pd.read_csv(path, index_col=False)
        instances = []
        for idx, row in df.iterrows():
            instances.append(Instance(id=str(idx), data={
                "n": int(row["n"]),
                "goals": _parse(row["goals"], {}),
                "static": [tuple(o) for o in _parse(row["static"], [])],
                "moving": [tuple(o) for o in _parse(row.get("moving"), [])],
                "bfs_steps": int(row["BFS_steps"]),
            }))
        return instances

    def context(self, instance: Instance) -> Dict[str, Any]:
        d = instance.data
        n = d["n"]
        goal_nums = sorted(d["goals"])
        goals = [{"k": k, "row": d["goals"][k][0], "col": d["goals"][k][1]} for k in goal_nums]
        cycle = expand_cycle(d["moving"])

        # Visual: S start, 1..K goals, X static obstacle, M moving obstacle path, . free
        grid = [["." for _ in range(n)] for _ in range(n)]
        for r, c in cycle:
            grid[r][c] = "M"
        for g in goals:
            grid[g["row"]][g["col"]] = str(g["k"])
        for r, c in d["static"]:
            grid[r][c] = "X"
        if grid[0][0] == ".":
            grid[0][0] = "S"

        return {
            "n": n,
            "goals": goals,
            "static": d["static"],
            "cycle": cycle,
            "grid_rows": grid,
            "prob_forward": PROB_FORWARD,
            "prob_slip_left": PROB_SLIP_LEFT,
            "prob_slip_right": PROB_SLIP_RIGHT,
            "moves": [{"name": a, "dr": dr, "dc": dc} for a, (dr, dc) in MOVES.items()],
            "outcomes": {a: [(a, PROB_FORWARD), (SLIP_LEFT[a], PROB_SLIP_LEFT), (SLIP_RIGHT[a], PROB_SLIP_RIGHT)]
                         for a in MOVES},
            "requirements": self._requirements(goal_nums, bool(cycle)),
        }

    @staticmethod
    def _sequence_formula(before: List[int], exclude: int) -> str:
        """Goals `before` reached in order, and `exclude` not reached until they all are."""
        if len(before) == 1:
            return f'!"at_goal{exclude}" U "at_goal{before[0]}"'
        rest = GridWorld._sequence_formula(before[1:], exclude)
        return f'!"at_goal{exclude}" U ("at_goal{before[0]}" & ({rest}))'

    def _requirements(self, goal_nums: List[int], has_moving: bool) -> List[Dict[str, Any]]:
        """Same requirements, names and thresholds as the legacy verifier."""
        reqs = [{"name": f"goal{k}", "formula": f'F "at_goal{k}"', "threshold": GOAL_THRESHOLD,
                 "description": f"The agent eventually reaches goal {k}."} for k in goal_nums]
        sequences = []
        for i in range(len(goal_nums) - 1):
            before, nxt = goal_nums[:i + 1], goal_nums[i + 1]
            formula = self._sequence_formula(before, nxt)
            sequences.append(formula)
            reqs.append({"name": f"seq_{goal_nums[i]}_before_{nxt}", "formula": formula,
                         "threshold": SEQUENCE_THRESHOLD,
                         "description": f"Goal{'s' if len(before) > 1 else ''} {', '.join(map(str, before))} "
                                        f"{'are' if len(before) > 1 else 'is'} visited{' in order' if len(before) > 1 else ''} "
                                        f"before goal {nxt} is ever entered."})
        if len(goal_nums) > 1:
            reqs.append({"name": "complete_sequence", "formula": " & ".join(f"({s})" for s in sequences),
                         "threshold": SEQUENCE_THRESHOLD,
                         "description": "All goals are visited in the correct order (all sequence requirements at once)."})
        if has_moving:
            for k in goal_nums:
                reqs.append({"name": f"avoid_moving_seg{k}", "formula": f'G ("in_seg{k}" => !"at_moving_obs")',
                             "threshold": AVOID_THRESHOLD,
                             "description": f"While heading to goal {k}, the agent never shares a cell "
                                            f"with the moving obstacle."})
        return reqs
