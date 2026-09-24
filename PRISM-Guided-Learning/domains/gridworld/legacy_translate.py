"""Translate legacy per-state gridworld policies into the symbolic rule form (for regression)."""
from typing import Dict, Iterable, List, Sequence, Tuple

from core.rules import atomic_rules

LEGACY_ACTIONS = {0: "up", 1: "right", 2: "down", 3: "left"}


def legacy_policy_to_rules(policy: Iterable[Tuple[Sequence, int]], num_goals: int) -> List[Tuple[str, str]]:
    """`policy`: pairs (state, action) with state = (x, y, g1, ..., gN) and action 0-3.

    Every state becomes one exact-match rule, so the result covers exactly the states the
    legacy policy assigned and chooses the same actions.
    """
    assignments = []
    for state, action in policy:
        x, y, *flags = state
        valuation: Dict = {"x": x, "y": y}
        valuation.update({f"g{k + 1}": bool(flags[k]) for k in range(num_goals)})
        assignments.append((valuation, LEGACY_ACTIONS[int(action)]))
    return atomic_rules(assignments)
