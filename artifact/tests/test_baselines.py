"""src/baselines/: the rule-list search, the decision trees, and the runner's output format."""
import shutil

import numpy as np
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
        "method": "rule_lists", "condition": "haiku_mi_grid_large_closer", "dataset": "grid_20_balanced.csv", "target_dataset": None,
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


@needs_prism
@pytest.mark.parametrize("inputs", ["raw", "features"])
def test_an_explicit_tree_compiles_into_rules_that_pick_its_action(inputs):
    """1-2-3-Go! on the explicit model: trained on Pac-Man horizon 8, its rules (one per action, the tree as an
    if-then-else) pick the tree's action in every decision state of horizon 10, and its leaves are as many rules."""
    from baselines.decision_tree import ExplicitDecisionTree
    from core.explicit import ExplicitMDP
    cfg = load_config("pacman_baseline")
    domain = load_domain("pacman")
    inst = {int(i.id): i for i in domain.load_instances("pacman_transfer.csv")}
    tree = DecisionTree(domain, {"model": "explicit", "inputs": inputs, "sets": {"h8": [0]},
                                 "features": ["moves_left", "wall", "nearer_g0", "nearer_g1"]})
    assert isinstance(tree, ExplicitDecisionTree)
    tree.config = cfg
    tree.fit([inst[0]])
    source = f"tree_{inputs}_h8"
    info = tree.summary()[source]
    assert info["leaves"] == len(info["rules"]) == tree.trees[source].get_n_leaves() and tree.training(source) == [0]
    mdp = ExplicitMDP(domain, inst[1], cfg)
    vocab = mdp.spec.vocabulary(cfg.rules)
    nested, leaves = (mdp.spec.policy(r, vocab) for r in (tree.rules(inst[1])[source], info["rules"]))
    acts = tree.actions_on(mdp, source)
    for s in np.flatnonzero(mdp.decision):
        valuation = mdp.valuation(mdp.obs[s])
        assert nested.allowed(valuation) == leaves.allowed(valuation) == [mdp.actions[acts[s]]]


def _pacman():
    domain = load_domain("pacman")
    return load_config("pacman_baseline"), domain, {int(i.id): i for i in domain.load_instances("pacman_transfer.csv")}


@needs_prism
def test_the_ppo_sampler_follows_the_explicit_model():
    """A fixed policy's simulated crash rate on Pac-Man horizon 10 is its exact value."""
    from core.explicit import ExplicitMDP
    from baselines.ppo import _Env
    cfg, domain, inst = _pacman()
    mdp = ExplicitMDP(domain, inst[1], cfg)
    rng = np.random.default_rng(0)
    policy = np.where(mdp.decision, rng.integers(0, 4, mdp.S), -1)
    env = _Env([mdp], None, lambda m, s: m.obs[s].astype(np.float32))
    n = 40000                                  # one episode each (stopping at a count of episodes would favour short ones)
    state, t, which = (np.zeros(n, dtype=np.int64) for _ in range(3))
    state[:], total, end = env.initial[0], np.zeros(n), np.zeros(n, dtype=bool)
    env._advance(np.arange(n), state, t, which, total, end, rng)     # to the first crossing, or a crash before it
    active = ~end
    while active.any():
        s, ts, w = state[active], t[active], which[active]
        rew, end = env.step(s, ts, w, policy[s], rng)
        state[active], t[active] = s, ts
        total[active] += rew
        active[np.flatnonzero(active)[end]] = False
    crash = -total.mean()
    assert crash == pytest.approx(mdp.evaluate(policy)["crash"], abs=4 * np.sqrt(crash * (1 - crash) / n))


@needs_prism
def test_explicit_ppo_freezes_a_policy_for_every_decision_state():
    from core.explicit import ExplicitMDP
    from baselines.ppo import PPO, ExplicitPPO
    cfg, domain, inst = _pacman()
    ppo = PPO(domain, {"model": "explicit", "observation": "features", "features": ["moves_left", "wall"],
                       "steps": 8192, "envs": 16, "seeds": [0]})
    assert isinstance(ppo, ExplicitPPO) and ppo.explicit
    ppo.config = cfg
    ppo.fit([inst[0]])
    mdp = ExplicitMDP(domain, inst[1], cfg)
    policy = ppo.policies(mdp)["ppo_features_seed0"]
    assert ((policy >= 0) == mdp.decision).all() and policy.max() < 4
    assert ppo.training("ppo_features_seed0") == [0]


def test_hashed_policies_are_fixed_by_their_seed():
    from baselines.simple import hashed_actions
    obs = np.random.default_rng(0).integers(0, 12, (5000, 10))
    a = hashed_actions(obs, 3, 4)
    assert (a == hashed_actions(obs, 3, 4)).all() and set(a) == {0, 1, 2, 3}
    assert (a != hashed_actions(obs, 4, 4)).mean() > 0.6
    assert (hashed_actions(obs[:10], 3, 4) == a[:10]).all()


@needs_prism
def test_one_instance_keeps_its_optimum_and_leaves_unseen_states_open():
    from baselines import load_method
    from core.explicit import ExplicitMDP
    cfg, domain, inst = _pacman()
    seen = ["xP", "yP", "dP", "xG0", "yG0", "dG0", "xG1", "yG1", "dG1"]
    one = load_method("one_instance")(domain, {"instance": 0, "keys": {"by_steps": seen + ["steps"],
                                                                        "by_end": seen + ["moves_left"]}})
    one.config = cfg
    one.fit([inst[0]])
    own = ExplicitMDP(domain, inst[0], cfg)
    for policy in one.policies(own).values():
        assert ((policy >= 0) == own.decision).all()
        assert own.evaluate(policy)["crash"] == pytest.approx(0.5511, abs=1e-9)
    longer = ExplicitMDP(domain, inst[1], cfg)
    by_steps = one.policies(longer)["by_steps"]
    steps = longer.obs[:, [v.name for v in longer.spec.variables].index("steps")]
    assert (by_steps[longer.decision & (steps >= 8)] == -1).all()
