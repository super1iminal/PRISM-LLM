"""Search over the rule language without an LLM: beam search for a decision list of feature rules that passes as
many training grids as possible, the first non-LLM baseline of the transfer search (docs/transfer_plan.md).

A rule is a conjunction of feature literals (each of `features` required true, required false, or ignored) with an
action order; it allows the actions whose features match, and the policy takes the first allowed action in the
rule's order (as `c -> any` rewritten into a fixed order). The list ends with `true -> <fallback>`. Lists are grown
one rule at a time, each extension scored on every training grid with the numpy gridworld (gridworld_mdp.py):
(grids passed, requirements met, -shortfall), summed. The best list is frozen and certified on targets by PRISM.
Gridworld only.

params:
  features: [wall, hazard, risky, toward, closer, obstacle]   features the rules may use
  orders: rotations      action orders a rule may use: "rotations" (4) or a list of comma-separated orders
  max_rules: 6           rules before the catch-all
  beam: 4                lists kept per length
  workers: 8             processes scoring candidates
  sets: null             training sets as lists of ids (one list per set); null: one set of all training ids
"""
import itertools
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from baselines.base import Baseline, Rules
from baselines.gridworld_mdp import ACTIONS, FEATURES, GridMDP, first_allowed
from core.domain import Instance

ROTATIONS = [[2, 1, 0, 3], [1, 0, 3, 2], [0, 3, 2, 1], [3, 2, 1, 0]]   # down-right-up-left and its rotations
Rule = Tuple[Tuple[Tuple[str, bool], ...], Tuple[int, ...]]           # (literals, action order)

_GRIDS: List[Tuple[GridMDP, list, dict]] = []                          # per worker process: mdp, requirements, monitors
_CACHE: Dict[Tuple[int, bytes], Dict[str, float]] = {}                  # per worker: (grid, policy) -> values


def _init(datas: List[dict], requirements: List[list]):
    global _GRIDS
    _GRIDS = []
    for data, reqs in zip(datas, requirements):
        mdp = GridMDP(data)
        _GRIDS.append((mdp, reqs, mdp.monitors()))


def _score(rules: Sequence[Rule], fallback: int = 0) -> Tuple[int, int, float]:
    """(grids passed, requirements met, -shortfall) of a list on every grid of this process."""
    passed = met = 0
    shortfall = 0.0
    for g, (mdp, reqs, monitors) in enumerate(_GRIDS):
        actions = first_allowed(mdp, [(dict(lits), list(order)) for lits, order in rules], fallback)
        key = (g, actions.tobytes())
        if key not in _CACHE:
            _CACHE[key] = mdp.evaluate(actions, monitors)
        values = _CACHE[key]
        ok = [r.satisfied(values[r.name]) for r in reqs]
        passed += all(ok)
        met += sum(ok)
        shortfall += sum(r.shortfall(values[r.name]) for r in reqs)
    return passed, met, -round(shortfall, 9)


def _score_many(lists: List[Tuple[Rule, ...]]) -> List[Tuple[int, int, float]]:
    return [_score(lst) for lst in lists]


def rule_text(rules: Sequence[Rule], fallback: int = 0) -> Rules:
    """A list in the rule language: each rule as one `(condition, action)` per action of its order."""
    out = []
    for lits, order in rules:
        cond = " & ".join(name if want else f"!{name}" for name, want in lits) or "true"
        out += [(cond, ACTIONS[a]) for a in order]
    return out + [("true", ACTIONS[fallback])]


def from_text(rules: Rules) -> Tuple[Tuple[Rule, ...], int]:
    """The inverse of `rule_text`: consecutive rules with the same conjunction of feature literals form one rule
    with that action order, and a final `true -> a` is the fallback. Raises ValueError for anything else."""
    out: List[Rule] = []
    body, last = list(rules[:-1]), rules[-1] if rules else ("", "")
    if last[0].strip() != "true":
        raise ValueError("the list must end with a `true -> action` rule")
    for cond, action in body:
        lits = tuple((t.strip().lstrip("!").strip(), not t.strip().startswith("!")) for t in cond.split("&"))
        if any(name not in FEATURES for name, _ in lits) or action not in ACTIONS:
            raise ValueError(f"not a conjunction of feature literals with an action: {cond} -> {action}")
        if out and out[-1][0] == lits:
            out[-1] = (lits, out[-1][1] + (ACTIONS.index(action),))
        else:
            out.append((lits, (ACTIONS.index(action),)))
    return tuple(out), ACTIONS.index(last[1])


def pool(features: Sequence[str], orders: Sequence[Sequence[int]], mdps: Sequence[GridMDP] = ()) -> List[Rule]:
    """Every rule over `features` (each true, false or ignored; not all ignored), with each order. Given `mdps`,
    rules that match nothing on them are dropped, and of rules that match the same (state, action) pairs on every
    one of them only the shortest is kept (they give the same policies there)."""
    filters, seen = [], set()
    for values in itertools.product((True, False, None), repeat=len(features)):
        lits = tuple((f, v) for f, v in zip(features, values) if v is not None)
        if lits:
            filters.append(lits)
    filters.sort(key=len)
    kept = []
    for lits in filters:
        if mdps:
            masks = [np.logical_and.reduce([m.features[f] == v for f, v in lits]) for m in mdps]
            signature = b"".join(np.packbits(mask).tobytes() for mask in masks)
            if not any(mask.any() for mask in masks) or signature in seen:
                continue
            seen.add(signature)
        kept.append(lits)
    return [(lits, tuple(o)) for lits in kept for o in orders]


def beam_search(datas: List[dict], requirements: List[list], features: Sequence[str], orders, max_rules: int,
                beam: int, workers: int, log=print) -> Tuple[Tuple[Rule, ...], Tuple[int, int, float]]:
    """The best list found (and its score) for the grids `datas`."""
    candidates = pool(features, orders, [GridMDP(d) for d in datas])
    log(f"  {len(candidates)} candidate rules")
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(datas, requirements)) as ex:
        def score_all(lists):
            chunks = [lists[i::workers * 4] for i in range(workers * 4)]
            scores = {}
            for chunk, result in zip(chunks, ex.map(_score_many, chunks)):
                scores.update(zip(chunk, result))
            return scores

        frontier: List[Tuple[Rule, ...]] = [()]
        best, best_score = (), score_all([()])[()]
        for depth in range(1, max_rules + 1):
            lists = list(dict.fromkeys(lst + (r,) for lst in frontier for r in candidates if r not in lst))
            scores = score_all(lists)
            ranked = sorted(lists, key=lambda lst: scores[lst], reverse=True)
            frontier, seen = [], set()
            for lst in ranked:                       # keep lists with distinct scores, so the beam stays diverse
                if scores[lst] not in seen:
                    seen.add(scores[lst])
                    frontier.append(lst)
                if len(frontier) == beam:
                    break
            if scores[ranked[0]] > best_score:
                best, best_score = ranked[0], scores[ranked[0]]
            log(f"  depth {depth}: {len(lists)} lists, best {scores[ranked[0]]} (overall {best_score})")
            if best_score[0] == len(datas):
                break
    return best, best_score


class FeatureSearch(Baseline):

    def __init__(self, domain, params):
        super().__init__(domain, params)
        self.features = params.get("features") or list(FEATURES)
        orders = params.get("orders") or "rotations"
        self.orders = ROTATIONS if orders == "rotations" else [[ACTIONS.index(a) for a in o.split(",")] for o in orders]
        self.max_rules = params.get("max_rules", 6)
        self.beam = params.get("beam", 4)
        self.workers = params.get("workers", 8)
        self.sets: Optional[List[List[int]]] = params.get("sets")
        self.found: Dict[str, Rules] = {}

    def fit(self, train: Sequence[Instance]) -> None:
        by_id = {int(i.id): i for i in train}
        sets = self.sets or [sorted(by_id)]
        for k, ids in enumerate(sets):
            members = [by_id[i] for i in ids]
            reqs = [self.domain.spec(m).requirements for m in members]
            print(f"feature search, set {k} {ids}:", flush=True)
            rules, score = beam_search([m.data for m in members], reqs, self.features, self.orders, self.max_rules,
                                       self.beam, self.workers, log=lambda s: print(s, flush=True))
            print(f"  -> {score}: {rule_text(rules)}", flush=True)
            self.found[f"set{k}"] = rule_text(rules)

    def rules(self, target: Instance) -> Dict[str, Rules]:
        return self.found
