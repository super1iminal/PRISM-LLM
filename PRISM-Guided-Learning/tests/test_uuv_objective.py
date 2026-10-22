"""Item 8 of Rajan's checklist: the UUV objective model for the baselines (UUV.objective) and its use by the
explicit-model baselines (decision trees, PPO)."""
from dataclasses import replace

import pytest

from config import load_config
from core.domain import load_domain
from core.prism import PrismRunner
from core.verifier import exact_solvers



@pytest.fixture(scope="module")
def setup():
    cfg = load_config("uuv_baseline")
    domain = load_domain("uuv")
    return cfg, domain, domain.load_instances("uuv_family_v0.csv")[0]


def test_the_objective_model_counts_steps_and_failures(setup):
    _, domain, inst = setup
    model, req = domain.objective(inst)
    assert "module stepcount" in model and f'label "goal" = s=done & t<={inst.data["deadline"]} & ok;' in model
    assert "rewards" not in model and req.formula == 'F "goal"' and req.bound == ">="


def test_numpy_and_prism_agree_on_the_objective(setup):
    from baselines.explicit_mdp import ExplicitMDP
    cfg, domain, inst = setup
    mdp = ExplicitMDP(domain, inst, cfg, objective=True)
    value, optimal = mdp.optimal_actions(mdp.objective_requirement)
    model, _ = domain.objective(inst)
    exact = replace(cfg.prism, method="", max_iters=cfg.prism.exact_max_iters)
    pmax = PrismRunner(exact, extra_args=exact_solvers(cfg.prism)[0][1]).run(model, ['Pmax=? [ F "goal" ]']).initial_values[0]
    assert abs(value[mdp.initial] - pmax) < 1e-6
    assert (mdp.decision & ~optimal.all(axis=1)).any()          # some states prefer an action


def test_the_objective_needs_the_domains_model(setup):
    from baselines.explicit_mdp import ExplicitMDP
    cfg, _, _ = setup
    mer = load_domain("mer")
    with pytest.raises(ValueError):
        ExplicitMDP(mer, mer.load_instances("mer_transfer.csv")[0], load_config("mer_baseline"), objective=True)


def test_ppo_can_be_certified_as_rules():
    from baselines.ppo import PPO
    domain = load_domain("uuv")
    assert PPO(domain, {"model": "explicit"}).explicit
    assert not PPO(domain, {"model": "explicit", "certify": "rules"}).explicit


def test_a_tree_learns_the_objective(setup):
    from baselines.decision_tree import DecisionTree
    cfg, domain, inst = setup
    tree = DecisionTree(domain, {"model": "explicit", "inputs": "raw", "requirement": "objective", "sets": {"t": [0]}})
    tree.config = cfg
    tree.fit([inst])
    rules = tree.rules(inst)["tree_raw_t"]
    domain.spec(inst).policy(rules, domain.spec(inst).vocabulary(cfg.rules))   # compiles in the baseline vocabulary
