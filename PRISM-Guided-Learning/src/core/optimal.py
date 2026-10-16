"""What an optimal strategy of an instance does, shown to the LLM: at the states the feedback names
(`feedback.optimal`) and along one optimal run in the initial prompt (`prompt.demonstration`).

The domain supplies the strategy (`Domain.optimal_strategy`): per state, its optimal actions and the chance of
success from there. It is the information 1-2-3-Go! learns from (gridworld: one objective for every requirement
at once, the trees' `gridworld_mdp.optimal_actions`; Pac-Man and MER: their requirement's optimal actions,
`explicit_mdp.ExplicitMDP.optimal_actions`), computed on a training member only.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from core.prism import PrismResult
from core.rules import Value
from core.verifier import PolicyVerifier

DEMO_STEPS = 30          # decisions shown in a demonstration at most
MAX_TRANSITIONS = 2000   # transitions followed while looking for them


@dataclass(frozen=True)
class Optimal:
    """The optimal strategy in one state."""
    value: float                    # chance of success from here under the optimal strategy
    actions: Tuple[str, ...]        # its optimal actions (policy actions, in the declared order)
    settled: bool = False           # nothing chosen from here on changes the outcome


# Domain.optimal_strategy returns one: a state (every model variable, by name) -> the strategy there, or None for a
# state outside its model
Strategy = Callable[[Dict[str, Value]], Optional[Optimal]]


def at(strategy: Strategy, result: PrismResult, s: int) -> Optional[Optimal]:
    """The strategy in state `s` of a PRISM result."""
    return strategy(dict(zip(result.variables, result.states[s])))


@dataclass
class DemoStep:
    valuation: Dict[str, Value]     # policy variables of the decision state
    actions: Tuple[str, ...]        # the optimal actions there


def demonstration(verifier: PolicyVerifier, strategy: Strategy, max_steps: int = DEMO_STEPS) -> List[DemoStep]:
    """One run of the optimal strategy on the verifier's bare model, from the initial state: at a decision state the
    first optimal action in the declared order, elsewhere the choice with the highest expected chance of success,
    and of each choice's outcomes the one most likely to succeed (its probability times the chance of success from
    there). Stops when the outcome is settled, a state repeats, or after `max_steps` decisions. Needs the optimum's
    transitions (`verifier.optimum`, cached by the loop)."""
    _, _, result = verifier.optimum()
    forced = verifier.forced_states()
    order = list(verifier.spec.actions)
    cache: Dict[int, Optional[Optimal]] = {}

    def value(t: int) -> float:
        if t not in cache:
            cache[t] = at(strategy, result, t)
        return cache[t].value if cache[t] else 0.0

    steps, s, seen = [], result.initial_state, set()
    for _ in range(MAX_TRANSITIONS):
        if len(steps) >= max_steps or s in seen:
            break
        seen.add(s)
        value(s)
        here = cache[s]
        if here is None or here.settled or not result.choices[s]:
            break
        choices = result.choices[s]
        if result.states[s] not in forced and here.actions:
            steps.append(DemoStep(verifier.policy_valuation(result, s), here.actions))
            first = next(a for a in order if a in here.actions)
            choices = [c for c in choices if c.action == first] or choices
        choice = max(choices, key=lambda c: sum(p * value(t) for t, p in c.successors))
        s = max(choice.successors, key=lambda tp: tp[1] * value(tp[0]))[0]
    return steps
