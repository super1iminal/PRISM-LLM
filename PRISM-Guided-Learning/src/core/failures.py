"""Where a rule list fails on a member, for the feedback `planner.feedback: failures` (after the fresh-transfer
session's method): per failing requirement, the states where the worst-case run is lost and where it is still stuck
at the end, with what decided them and what each action does there. No optimal actions are shown.

A requirement is *lost* in a state of the bare model when even the best scheduler can no longer meet it from there
(its optimum is 0 for a lower bound, 1 for an upper bound: a collision, a crash, a goal entered early). The worst-case
run follows the adversary's choices (greedy on the worst-case values of the rules' model) from the initial state for
`horizon` steps; a run stops when the requirement is lost or certain to hold. A state *fails* by the probability flux
from it into lost states; runs still going at the end are *stuck* where they are (an adversary stalling forever).
Per action, the features that hold for it and its one-step risk: the chance its next state is lost, per requirement,
on the bare model (so also for actions the rules do not allow there).
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.analysis import _greedy, _q_values, _signed
from core.domain import Requirement
from core.verifier import PolicyVerifier, Verification


@dataclass
class FailureState:
    kind: str                       # "fails" (flux into lost states) or "stuck" (still going at the end)
    state: int                      # index in the rules' model
    mass: float                     # share of runs
    choice: int                     # the adversary's choice there (index into the state's choices)


Met = Callable[[Requirement], Optional[Callable[[Dict], bool]]]


def lost_sets(verifier: PolicyVerifier, met: Optional[Met] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Per probability requirement, the bare model's states where it is lost (bool array), and the bare model's
    state index (state tuple -> index). Needs the optimum's vectors (`verifier.optimum`, cached by the loop). With
    `met` (`feedback.history`: the domain's test per requirement), a state that shows the requirement met is not
    lost."""
    _, vectors, result = verifier.optimum()
    out = {}
    for r in verifier.spec.requirements:
        if r.reward is not None or vectors.get(r.name) is None:
            continue
        v = np.asarray(vectors[r.name], dtype=float)
        out[r.name] = v <= 1e-12 if r.maximize else v >= 1.0 - 1e-12
        done = met_states(verifier, result, met(r)) if met else None
        if done is not None:
            out[r.name] &= ~done
    return out, result.index_of()


def met_states(verifier: PolicyVerifier, result, test: Optional[Callable[[Dict], bool]]) -> Optional[np.ndarray]:
    """The states of `result` whose policy variables pass `test` (Domain.met), or None without a test."""
    if test is None:
        return None
    return np.array([test(verifier.policy_valuation(result, s)) for s in range(len(result.states))], dtype=bool)


def failure_states(v: Verification, req: Requirement, lost_here: np.ndarray, horizon: int,
                   top: int = 2, met: Optional[np.ndarray] = None) -> List[FailureState]:
    """The `top` states the worst-case run of `req` fails from (by flux into `lost_here`, per state of the rules'
    model), then the `top` states it is still stuck in after `horizon` steps. Runs stop where `met` (the states that
    show `req` met, `feedback.history`) holds, as where it is certain to hold."""
    result = v.result
    w = np.asarray(v.worst_vectors[req.name], dtype=float)
    success = (w >= 1.0 - 1e-12) if req.maximize else (w <= 1e-12)
    if met is not None:
        success = success | met
    stop = lost_here | success
    strategy = _greedy(_q_values(result, _signed(req, w)), maximize=False)
    n = len(result.states)
    src, dst, prob = [], [], []
    for s, c in enumerate(strategy):
        if c >= 0 and not stop[s]:
            for t, p in result.choices[s][c].successors:
                src.append(s), dst.append(t), prob.append(p)
    src, dst, prob = np.array(src, dtype=int), np.array(dst, dtype=int), np.array(prob)
    into_lost = np.bincount(src, weights=prob * lost_here[dst], minlength=n) if len(src) else np.zeros(n)
    mu = np.zeros(n)
    mu[result.initial_state] = 1.0
    flux = np.zeros(n)
    for _ in range(horizon):
        live = mu * ~stop
        if live.sum() < 1e-12:
            break
        flux += live * into_lost
        mu = np.bincount(dst, weights=live[src] * prob, minlength=n) if len(src) else np.zeros(n)
    out = [FailureState("fails", int(s), float(flux[s]), int(strategy[s])) for s in np.argsort(-flux)[:top]
           if flux[s] >= 1e-4]
    stuck = mu * ~stop
    if stuck.sum() > 1e-3:
        out += [FailureState("stuck", int(s), float(stuck[s]), int(strategy[s])) for s in np.argsort(-stuck)[:top]
                if stuck[s] >= 1e-4]
    return out


def action_lines(verifier: PolicyVerifier, v: Verification, s: int, lost: Dict[str, np.ndarray], bare_index: Dict,
                 features: Callable[[Dict], Dict], feature_names: Sequence[str],
                 words: Optional[Callable[[str], Optional[str]]] = None) -> List[Tuple[str, str]]:
    """Per policy action at state `s` of the rules' model: the action features true for it, and its one-step risk on
    the bare model (`loses r with p`, requirements with equal chances together, or with `words` (requirement name ->
    what losing it means, `feedback.words`) those words; `changes nothing` for a choice whose every outcome is the
    state itself)."""
    _, _, bare = verifier.optimum()
    i = bare_index.get(v.result.states[s])
    valuation = verifier.policy_valuation(v.result, s)
    values = features(valuation)
    out = []
    for a in verifier.spec.actions:
        held = [f for f in feature_names if values[(f, a)] is True] + \
               [f"{f}={values[(f, a)]}" for f in feature_names if not isinstance(values[(f, a)], bool)]
        text = f"[{', '.join(held) or 'no feature true'}]"
        choices = [c for c in bare.choices[i] if c.action == a] if i is not None else []
        if choices:
            c = choices[0]
            if all(t == i for t, _ in c.successors):
                text += " changes nothing"
            risk = defaultdict(list)
            for name, lost_set in lost.items():
                p = sum(q for t, q in c.successors if lost_set[t])
                if p > 1e-9:
                    risk[round(p, 2)].append(name)
            if risk:
                parts = []
                for p, names in sorted(risk.items(), reverse=True):
                    said = [words(n) if words else None for n in names]
                    parts += [f"{w} {p:.2f}" for w in dict.fromkeys(w for w in said if w)]
                    plain = [n for n, w in zip(names, said) if not w]
                    if plain:
                        parts.append(f"loses {', '.join(plain)} {p:.2f}")
                text += " risk: " + "; ".join(parts)
        out.append((a, text))
    return out


def allowed_here(verifier: PolicyVerifier, v: Verification, s: int) -> List[str]:
    """The policy actions the rules' model keeps at state `s`, in the declared order."""
    labels = {c.action for c in v.result.choices[s]}
    return [a for a in verifier.spec.actions if a in labels]


def choice_label(v: Verification, s: int, c: int) -> Optional[str]:
    return v.result.choices[s][c].action if 0 <= c < len(v.result.choices[s]) else None
