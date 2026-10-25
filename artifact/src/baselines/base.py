"""What every baseline is: a method that writes policies for a family of instances, fitted on training instances,
then frozen and certified on each target exactly as src/transfer.py certifies an LLM's rules.

A baseline gives, per target, named rule lists (`(condition, action)` pairs in the rule language of the run
condition it is configured with). Each name is a `source` in the results, so one baseline can report several
policies (e.g. one per training set, or variants of one list). Methods that do not produce rules in our language
must compile their policy into one (e.g. a decision tree into a decision list).
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import Config
from core.domain import Domain, Instance

Rules = List[Tuple[str, str]]


class Baseline(ABC):
    """Fitted once on the training instances (`fit`), then asked for its policies on each target (`rules`).

    Methods whose policies have no compact rules (`explicit = True`: a network, a table, a randomized policy) give
    per-state policies on the target's explicit model instead (`policies`), which core.explicit.ExplicitMDP.certify
    hands to PRISM; their `rules` are the same policies as one rule per state, for small targets only.
    """
    explicit = False
    config: Optional[Config] = None        # the run condition's config, set by src/run_baseline.py

    def __init__(self, domain: Domain, params: Dict[str, Any]):
        self.domain = domain
        self.params = params

    def fit(self, train: Sequence[Instance]) -> None:
        """Learn from the training instances; methods that are not fitted ignore them."""

    @abstractmethod
    def rules(self, target: Instance) -> Dict[str, Rules]:
        """Source name -> the frozen rule list to certify on `target`."""

    def policies(self, mdp) -> Dict[str, np.ndarray]:
        """Source name -> its per-state policy on `mdp` (an core.explicit.ExplicitMDP of the target), for `explicit`
        methods."""
        raise NotImplementedError

    def training(self, source: str) -> Optional[List[int]]:
        """The ids `source` was fitted on, when they are not all the training ids (None)."""
        return None

    def summary(self) -> Dict[str, Any]:
        """What the fit produced, per source (e.g. a tree's size and rules), for <results>/fit.json; {} for none."""
        return {}


def named_sets(sets, ids: Sequence[int]) -> Dict[str, List[int]]:
    """Training sets by name: `sets` as a mapping (name -> ids), a list (set0, set1, ...), or None (one set of all
    `ids`, named "")."""
    if sets is None:
        return {"": sorted(ids)}
    if isinstance(sets, dict):
        return {str(name): [int(i) for i in s] for name, s in sets.items()}
    return {f"set{k}": [int(i) for i in s] for k, s in enumerate(sets)}
