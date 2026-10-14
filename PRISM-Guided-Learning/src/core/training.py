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
    hidden: Tuple[Instance, ...] = ()     # checked and scored with the members, never shown to the LLM
    pool: Tuple[Instance, ...] = ()       # once the members pass, the first of these that fails joins them


def training_sets(instances: Sequence[Instance], groups: Sequence[Sequence[int]],
                  hidden: Sequence[Sequence[int]] = (), pool: Sequence[Sequence[int]] = ()) -> List[TrainingSet]:
    """One training set per group of instance ids (dataset rows), named set0, set1, ...; `hidden[k]` and `pool[k]`
    (if given) are set k's hidden members and its pool of counterexample candidates."""
    by_id = {inst.id: inst for inst in instances}
    for name, extra in (("hidden_sets", hidden), ("pool_sets", pool)):
        if extra and len(extra) != len(groups):
            raise ValueError(f"run.{name} has {len(extra)} groups for {len(groups)} training sets")
    sets = []
    for k, group in enumerate(groups):
        hide, cand = (list(hidden[k]) if hidden else []), (list(pool[k]) if pool else [])
        missing = [i for i in list(group) + hide + cand if str(i) not in by_id]
        if missing or not group:
            raise ValueError(f"training set {k} ({list(group)}): no instances with ids {missing} in the dataset"
                             if missing else f"training set {k} is empty")
        sets.append(TrainingSet(f"set{k}", tuple(by_id[str(i)] for i in group), tuple(by_id[str(i)] for i in hide),
                                tuple(by_id[str(i)] for i in cand)))
    return sets
