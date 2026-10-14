"""What every baseline is: a method that writes policies for a family of instances, fitted on training instances,
then frozen and certified on each target exactly as src/transfer.py certifies an LLM's rules.

A baseline gives, per target, named rule lists (`(condition, action)` pairs in the rule language of the run
condition it is configured with). Each name is a `source` in the results, so one baseline can report several
policies (e.g. one per training set, or variants of one list). Methods that do not produce rules in our language
must compile their policy into one (e.g. a decision tree into a decision list).
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple

from core.domain import Domain, Instance

Rules = List[Tuple[str, str]]


class Baseline(ABC):
    """Fitted once on the training instances (`fit`), then asked for its policies on each target (`rules`)."""

    def __init__(self, domain: Domain, params: Dict[str, Any]):
        self.domain = domain
        self.params = params

    def fit(self, train: Sequence[Instance]) -> None:
        """Learn from the training instances; methods that are not fitted ignore them."""

    @abstractmethod
    def rules(self, target: Instance) -> Dict[str, Rules]:
        """Source name -> the frozen rule list to certify on `target`."""
