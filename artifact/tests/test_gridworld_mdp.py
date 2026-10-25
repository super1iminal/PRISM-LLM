"""src/baselines/gridworld_mdp.py: the numpy gridworld agrees with the core rule evaluator and with PRISM."""
import random
import shutil

import pytest

from baselines.gridworld_mdp import ACTIONS, FEATURES, OPTIONAL_FEATURES, GridMDP, RuleScorer, first_allowed
from config import load_config
from core.domain import load_domain
from core.rules import fixed_order
from core.verifier import PolicyVerifier

needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")
CFG = load_config("haiku_mi_grid_large_closer", ["rules.hidden_features=[]", "rules.extra_features=[collide]"])
RULES = [({"closer": True, "hazard": False, "risky": False}, [2, 1, 0, 3]),
         ({"closer": True, "hazard": False}, [1, 2, 3, 0]),
         ({"wall": False, "hazard": False}, None),
         ({"wall": False}, None)]


def _instance(dataset, i):
    domain = load_domain("gridworld", CFG.domain.visible_extra)
    return domain, domain.load_instances(dataset)[i]


@pytest.mark.parametrize("dataset, i", [("grid_20_balanced.csv", 12), ("families_v0.csv", 61)])
def test_features_match_the_core_rule_evaluator(dataset, i):
    domain, inst = _instance(dataset, i)
    mdp, spec = GridMDP(inst.data), domain.spec(inst)
    vocab = spec.vocabulary(CFG.rules)
    for f in FEATURES + OPTIONAL_FEATURES:
        policies = {a: spec.policy([(f, a)], vocab) for a in ACTIONS}
        for idx in random.Random(0).sample(range(mdp.S), 100):
            r, c, s, o = mdp.states[idx]
            state = {"x": r, "y": c, "obs_idx": o, **{f"g{k + 1}": k < s for k in range(mdp.K)}}
            for ai, a in enumerate(ACTIONS):
                assert (policies[a].allowed(state) == [a]) == bool(mdp.features[f][idx, ai]), (f, a, mdp.states[idx])


@needs_prism
@pytest.mark.parametrize("dataset, i", [("grid_20_balanced.csv", 12), ("families_v0.csv", 13)])
def test_rule_scorer_matches_prism_on_each_order(dataset, i):
    """RuleScorer.order_values: the values PRISM finds for the rules with `any` resolved in each order, or None where
    some reachable state is left open."""
    domain, inst = _instance(dataset, i)
    spec = domain.spec(inst)
    vocab = spec.vocabulary(CFG.rules)
    raw = [("closer & !hazard & !risky", "any"), ("closer & !hazard", "right"), ("!wall & !hazard", "any")]
    orders = [("up", "right", "down", "left"), ("left", "down", "right", "up")]
    scorer = RuleScorer(inst.data, spec.variables)
    values = scorer.order_values(spec.policy(raw + [("true", "any")], vocab), orders)
    assert values[0] != values[1]
    for order, (best, worst) in zip(orders, values):
        v, _ = PolicyVerifier(domain, inst, CFG).verify_exact(spec.policy(fixed_order(raw + [("true", "any")], order),
                                                                          vocab))
        assert best == worst == pytest.approx(v.worst, abs=1e-7)
    assert scorer.order_values(spec.policy([("g1", "any")], vocab), orders) == [None, None]   # the start is open


@needs_prism
@pytest.mark.parametrize("dataset, i", [("grid_20_balanced.csv", 12), ("families_v0.csv", 13)])
def test_requirement_values_match_prism(dataset, i):
    domain, inst = _instance(dataset, i)
    mdp, spec = GridMDP(inst.data), domain.spec(inst)
    values = mdp.evaluate(first_allowed(mdp, RULES))
    text = [(" & ".join(k if v else f"!{k}" for k, v in lits.items()), ACTIONS[a])
            for lits, order in RULES for a in (order or range(4))] + [("true", "up")]
    v, _ = PolicyVerifier(domain, inst, CFG).verify_exact(spec.policy(text, spec.vocabulary(CFG.rules)))
    assert set(values) == {r.name for r in spec.requirements}
    for name, value in values.items():
        assert value == pytest.approx(v.worst[name], abs=1e-7)
