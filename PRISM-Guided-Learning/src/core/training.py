"""Multi-instance training: one rule set written for, and checked on, several instances of a domain at once.

A run with `run.train_sets` solves training sets instead of single instances: each round, the LLM sees every
member, its rules are parsed for each member (instance constants differ) and verified on each, and the set
is solved only when every member meets every requirement in the worst case. The frozen result is then
certified on held-out instances with src/transfer.py. Semantics: docs/semantics.md ("Training sets").
"""
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from core.domain import Instance


@dataclass(frozen=True)
class TrainingSet:
    id: str
    members: Tuple[Instance, ...]


def training_sets(instances: Sequence[Instance], groups: Sequence[Sequence[int]]) -> List[TrainingSet]:
    """One training set per group of instance ids (dataset rows), named set0, set1, ..."""
    by_id = {inst.id: inst for inst in instances}
    sets = []
    for k, group in enumerate(groups):
        missing = [i for i in group if str(i) not in by_id]
        if missing or not group:
            raise ValueError(f"training set {k} ({list(group)}): no instances with ids {missing} in the dataset"
                             if missing else f"training set {k} is empty")
        sets.append(TrainingSet(f"set{k}", tuple(by_id[str(i)] for i in group)))
    return sets
