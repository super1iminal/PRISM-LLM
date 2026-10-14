"""src/baselines/: the rule-list search, the decision trees, and the runner's output format."""
import shutil

import pytest
import yaml

from baselines.decision_tree import DecisionTree, policy_rules
from baselines.feature_search import ROTATIONS, from_text, pool, rule_text
from baselines.gridworld_mdp import GridMDP, first_allowed, optimal_actions
from config import load_config
from core.domain import load_domain

needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")


def _grid(i=12, dataset="grid_20_balanced.csv"):
    domain = load_domain("gridworld", ["obs_idx"])
    return domain, domain.load_instances(dataset)[i]


def test_rule_text_and_from_text_are_inverse():
    rules = ((("closer", True), ("hazard", False)), (2, 1, 0, 3)), ((("wall", True),), (0,))
    text = rule_text(rules, fallback=1)
    assert text[0] == ("closer & !hazard", "down") and text[-1] == ("true", "right")
    assert from_text(text) == (rules, 1)
    with pytest.raises(ValueError):
        from_text([("x = 1", "up"), ("true", "up")])


def test_the_pool_drops_rules_that_match_nothing_or_the_same_as_a_shorter_one():
    _, inst = _grid()
    mdp = GridMDP(inst.data)
    full = pool(["wall", "closer"], ROTATIONS[:1])
    pruned = pool(["wall", "closer"], ROTATIONS[:1], [mdp])
    assert len(full) == 8 and len(pruned) < len(full)


def test_the_optimal_strategy_passes_its_own_grid():
    domain, inst = _grid()
    mdp = GridMDP(inst.data)
    value, optimal = optimal_actions(mdp)
    values = mdp.evaluate(optimal.argmax(axis=1))
    assert value[mdp.initial] > 0.9
    assert all(r.satisfied(values[r.name]) for r in domain.spec(inst).requirements)


@needs_prism
def test_a_tree_policy_as_rules_has_the_fast_models_values():
    from core.verifier import PolicyVerifier
    domain, inst = _grid()
    tree = DecisionTree(domain, {"inputs": "features"})
    tree.fit([inst])
    mdp = GridMDP(inst.data)
    values = mdp.evaluate(tree.actions(inst))
    rules = tree.rules(inst)["tree_features"]
    assert rules == policy_rules(mdp, tree.actions(inst))
    cfg = load_config()
    v, _ = PolicyVerifier(domain, inst, cfg).verify_exact(domain.spec(inst).policy(rules))
    for name, value in values.items():
        assert value == pytest.approx(v.worst[name], abs=1e-7)


@needs_prism
def test_run_baseline_writes_the_transfer_format(tmp_path):
    import run_baseline
    config = tmp_path / "lists.yaml"
    config.write_text(yaml.safe_dump({
        "method": "rule_lists", "condition": "haiku_mi_grid_large_closer", "dataset": "grid_20_balanced.csv",
        "train": "12", "targets": "12-13", "workers": 2, "exact_near": 0.0,
        "params": {"lists": {"greedy": [["closer & !hazard", "down"], ["closer & !hazard", "right"],
                                        ["true", "up"]]}}}), encoding="utf-8")
    df = run_baseline.run(config, out_root=tmp_path)
    assert set(df.columns) >= {"source", "target", "certified", "met", "trained", "method"}
    assert df.trained.tolist() == [True, False] and (df.method == "rule_lists").all()
    assert (tmp_path / "lists" / "transfer_grid_20_balanced.csv").exists()


def test_first_allowed_takes_the_first_matching_action_of_the_first_rule_that_allows_one():
    _, inst = _grid()
    mdp = GridMDP(inst.data)
    acts = first_allowed(mdp, [({"wall": True}, [0, 1, 2, 3])], fallback=2)
    walls = mdp.features["wall"]
    assert all(walls[i, a] for i, a in enumerate(acts) if walls[i].any())
    assert all(a == 2 for i, a in enumerate(acts) if not walls[i].any())
