"""What the refinement loop works with while it solves one instance or training set (core/planner.py).

An `Episode` holds what stays fixed (its members, the answer schema, the task factory); each round asks the
LLM (`Answer`) and checks the rule list on every member (`Check`); keep-best keeps the best check so far
(`Kept`). A single instance is an episode with one member.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.analysis import MassAnalyzer
from core.domain import Instance, Requirement, failing
from core.rules import SymbolicPolicy, Vocabulary
from core.tasks import LLMResult, TaskFactory
from core.verifier import PolicyVerifier, Verification


def score(v: Verification, reqs: List[Requirement]) -> Tuple:
    """Lower is better: worst-case failures, best-case failures, then total shortfalls."""
    return (len(failing(reqs, v.worst)),
            len(failing(reqs, v.best)),
            sum(r.shortfall(v.worst[r.name]) for r in reqs),
            sum(r.shortfall(v.best[r.name]) for r in reqs))


def pessimistic(values: List[Dict[str, float]], reqs: List[Requirement]) -> Dict[str, float]:
    """Per requirement, the least favourable of several members' values."""
    return {r.name: (min if r.maximize else max)(d[r.name] for d in values) for r in reqs}


@dataclass
class Member:
    """One instance the rules are written for, with what stays fixed while it is checked."""
    label: str                        # how prompts and logs name it in a training set
    instance: Instance
    verifier: PolicyVerifier
    analyzer: MassAnalyzer
    vocabulary: Vocabulary            # what rule conditions may name besides the policy variables
    context: Dict[str, Any]           # prompt context of this instance

    @property
    def requirements(self) -> List[Requirement]:
        return self.verifier.spec.requirements


@dataclass
class Episode:
    """What stays fixed while one instance, or one training set, is solved."""
    members: List[Member]             # one for a single instance
    schema: type
    log: Callable[[str], None]
    tasks: TaskFactory                # numbers the episode's LLM calls (their seeds)
    training: Dict[str, Any]          # members and variables as the prompt describes them; {} for one instance

    @property
    def primary(self) -> Member:
        return self.members[0]

    def on(self, member: Member) -> str:
        """Log suffix naming the member in a training set."""
        return f" on {member.label}" if len(self.members) > 1 else ""


@dataclass
class Answer:
    """What one round's questions to the LLM produced."""
    policies: Optional[List[SymbolicPolicy]]   # the rules parsed for each member; None if every answer was invalid
    errors: List[str]                 # invalid answers (and a failed verification), in order
    results: List[LLMResult]          # one per LLM call, re-asks included
    seconds: float = 0.0              # wall time, including queueing and, in lockstep, waiting for the batch


@dataclass
class Check:
    """One rule list checked on every member: parsed for each (instance constants differ) and verified."""
    policies: List[SymbolicPolicy]
    vs: List[Verification]

    @property
    def policy(self) -> SymbolicPolicy:
        """The rules as written, the same text for every member."""
        return self.policies[0]

    def score(self, members: List[Member]) -> Tuple:
        """The members' scores, summed."""
        return tuple(map(sum, zip(*(score(v, m.requirements) for m, v in zip(members, self.vs)))))

    def passes(self, members: List[Member], case: str = "worst") -> bool:
        return all(not failing(m.requirements, getattr(v, case)) for m, v in zip(members, self.vs))

    def focus(self, members: List[Member]) -> int:
        """The member the feedback is about: the one that fails worst (the first, on ties)."""
        scores = [score(v, m.requirements) for m, v in zip(members, self.vs)]
        return scores.index(max(scores))


@dataclass
class Kept:
    """The best rule set so far (keep-best), with its check and score."""
    check: Check
    score: Tuple
    focus: int                        # index of the member the feedback is about
    joint: Optional[bool] = None      # can one completion meet every threshold there? (None: undecided)
    joint_queried: bool = False
