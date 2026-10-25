"""src/baselines/explicit_mdp.py: a domain's model as PRISM exports it, its optimum, and the PRISM certificate of a
per-state policy."""
import shutil

import numpy as np
import pytest

from baselines.explicit_mdp import ExplicitMDP, per_state_rules
from baselines.gridworld_mdp import ACTIONS, GridMDP, optimal_actions
from config import load_config
from core.domain import Instance, load_domain
from core.prism import PrismRunner
from core.verifier import PolicyVerifier
from transfer import certify

pytestmark = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")
CFG = load_config("pacman_baseline")


def pacman(max_steps):
    return Instance(id=str(max_steps), data={"name": f"steps_{max_steps}", "max_steps": max_steps,
                                             "crash_threshold": 0.56})


@pytest.fixture(scope="module")
def pac8():
    return ExplicitMDP(load_domain("pacman"), pacman(8), CFG)


@pytest.mark.parametrize("max_steps", [8, 25])
def test_pacman_optimum_is_prisms(max_steps, pac8):
    """The optimum over all schedulers (the idle step included) is QVBS's 0.5511 at every horizon."""
    mdp = pac8 if max_steps == 8 else ExplicitMDP(load_domain("pacman"), pacman(max_steps), CFG)
    crash = mdp.spec.requirements[0]
    value, optimal = mdp.optimal_actions(crash)
    assert value[mdp.initial] == pytest.approx(0.5511, abs=1e-9)
    assert optimal[mdp.decision].any(axis=1).all()


def test_decision_states_are_the_verifiers(pac8):
    verifier = PolicyVerifier(pac8.domain, pac8.instance, CFG)
    forced = verifier.forced_states()
    result = verifier.runner.run(verifier.model, [], export_transitions=True)
    assert result.variables == pac8.variables
    theirs = {s for s in result.states if s not in forced}
    assert {tuple(int(v) for v in row) for row in pac8.values[pac8.decision]} == theirs


def test_features_are_the_rule_evaluators(pac8):
    states = np.flatnonzero(pac8.decision)
    table, columns = pac8.features(states, ["moves_left", "wall"])
    assert [c.label for c in columns] == ["moves_left"] + [f"wall[{a}]" for a in pac8.actions]
    assert (table[:, 0] == 8 - pac8.obs[states, [v.name for v in pac8.spec.variables].index("steps")]).all()
    walls = {(6, 4): "up", (8, 4): "up", (10, 4): "right"}
    pos = [v.name for v in pac8.spec.variables]
    for row, s in zip(table, states):
        x, y = pac8.obs[s, pos.index("xP")], pac8.obs[s, pos.index("yP")]
        if (x, y) in walls:
            assert row[1 + pac8.actions.index(walls[(x, y)])] == 1


def test_explicit_and_rule_certificates_agree(pac8):
    """A per-state policy certified through the explicit model and as one rule per state gives the same verdict,
    worst case and uncovered count, also when some decision states are left undecided."""
    rng = np.random.default_rng(1)
    policy = np.where(pac8.decision, rng.integers(0, 4, pac8.S), -1)
    partial = policy.copy()
    partial[np.flatnonzero(pac8.decision)[::5]] = -1
    domain, runner = pac8.domain, PrismRunner(CFG.prism)
    for p in (policy, partial):
        explicit = pac8.certify({"p": p})[0]
        rules = certify(domain, CFG, runner, pac8.instance, {"p": per_state_rules(pac8, p)})[0]
        assert explicit["worst_crash"] == pytest.approx(rules["worst_crash"], abs=1e-9)
        assert explicit["certified"] == rules["certified"] and explicit["uncovered"] == rules["uncovered"]
        assert pac8.evaluate(p)["crash"] == pytest.approx(explicit["worst_crash"], abs=1e-9)
    assert pac8.certify({"partial": partial})[0]["uncovered"] > 0


def test_a_randomized_policy_is_the_mixture_of_its_actions(pac8):
    """Uniform over the four actions: certified by PRISM on the mixed model, as numpy computes it."""
    uniform = np.full((pac8.S, 4), 0.25)
    row = pac8.certify({"uniform": uniform})[0]
    assert row["worst_crash"] == pytest.approx(pac8.evaluate(uniform)["crash"], abs=1e-9)
    bare = pac8.under(np.full(pac8.S, -1)).value(pac8.spec.requirements[0], worst=True)    # every choice open
    assert 0.5511 < row["worst_crash"] < bare


def test_optimal_strategy_is_the_trees_optimal_actions(pac8):
    """explicit_mdp.optimal_strategy (`feedback.optimal`, `prompt.demonstration` on Pac-Man and MER): per state the
    trees' optimal actions and the chance of not crashing."""
    from baselines.explicit_mdp import optimal_strategy
    strategy = optimal_strategy(pac8.domain, pac8.instance, CFG)
    crash, actions = pac8.spec.requirements[0], pac8.actions
    value, optimal = pac8.optimal_actions(crash)
    for s in np.flatnonzero(pac8.decision)[::50]:
        info = strategy(dict(zip(pac8.variables, pac8.values[s].tolist())))
        assert info.value == pytest.approx(1 - value[s])
        assert info.actions == tuple(a for a, ok in zip(actions, optimal[s]) if ok)
    assert strategy(dict(zip(pac8.variables, pac8.values[pac8.initial].tolist()))).value == pytest.approx(1 - 0.5511,
                                                                                                          abs=1e-9)


def test_only_reachability_and_until_over_labels():
    from baselines.explicit_mdp import _path_sets
    labels = {"a": np.array([True, False, False]), "b": np.array([False, True, False])}
    stay, goal = _path_sets('!"a" U ("b" | "a")', labels, 3)
    assert stay.tolist() == [False, True, True] and goal.tolist() == [True, True, False]
    for formula in ['G "a"', 'F<=3 "a"', 'F x = 1', '"a" U "b" U "a"']:
        with pytest.raises(NotImplementedError):
            _path_sets(formula, labels, 3)


def test_gridworld_optimum_and_policy_values_are_gridmdps():
    """On one small grid: Pmax of reaching goal 1 equals value iteration on GridMDP, and GridMDP's optimal policy
    has the same goal and sequence values on both models."""
    domain = load_domain("gridworld", ["obs_idx"])
    inst = domain.load_instances("grid_20_balanced.csv")[0]
    grid, mdp = GridMDP(inst.data), ExplicitMDP(domain, inst, load_config())
    V = np.array([(r, c) == grid.goals[0] for (r, c, s, o) in grid.states], dtype=float)
    goal = V == 1
    for _ in range(10000):
        V = np.where(goal, 1.0, (grid.prob[None] * V[grid.succ]).sum(axis=2).max(axis=1))
    reqs = {r.name: r for r in mdp.spec.requirements}
    assert mdp.value(reqs["goal1"], worst=False) == pytest.approx(V[grid.initial], abs=1e-9)

    _, optimal = optimal_actions(grid)
    acts = optimal.argmax(axis=1)
    col = {v: i for i, v in enumerate(mdp.variables)}
    flags = sorted(v for v in mdp.variables if v.startswith("g") and v[1:].isdigit())
    stage = sum(mdp.values[:, col[f]] for f in flags)
    phase = mdp.values[:, col["obs_idx"]] if "obs_idx" in col else np.zeros(mdp.S, dtype=int)
    index = np.array([grid.index[(x, y, s, o)] for x, y, s, o in zip(mdp.values[:, col["x"]], mdp.values[:, col["y"]],
                                                                      stage, phase)])
    policy = np.array([mdp.actions.index(ACTIONS[a]) for a in acts[index]])
    names = [n for n in reqs if n.startswith("goal") or n == "seq_1_before_2"]
    ours, theirs = mdp.evaluate(policy, names=names), grid.evaluate(acts)
    for name in names:
        assert ours[name] == pytest.approx(theirs[name], abs=1e-9)
