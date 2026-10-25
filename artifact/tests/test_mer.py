"""The MER domain reproduces 1-2-3-Go!'s published optima, and its thresholds separate x <= 0.05 from x >= 0.08."""
import math
import shutil

import pytest

from config import load_config
from core.domain import load_domain
from core.prism import PrismRunner
from core.rules import RuleError
from core.verifier import PolicyVerifier

CFG = load_config()

pytestmark = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")


@pytest.fixture(scope="module")
def domain():
    return load_domain("mer")


@pytest.fixture(scope="module")
def instances(domain):
    return {i.data["name"]: i for i in domain.load_instances("mer_transfer.csv")}


@pytest.mark.parametrize("name, published", [("n10_x0.01", 0.2016128), ("n10_x0.08", 0.213675199),
                                             ("n10_x0.55", 0.3570457237), ("n10_x0.89", 0.6711936689)])
def test_bare_mdp_matches_published_optimum(domain, instances, name, published):
    """1-2-3-Go! (arXiv:2410.18293, Table 2) reports Storm's Pmax for mer+p1 at n = 10."""
    result = PrismRunner(CFG.prism).run(domain.model(instances[name]), ['Pmax=? [ F "err" ]'], export_transitions=True)
    assert len(result.states) == 60634
    assert result.initial_values[0] == pytest.approx(published, abs=1e-9)


def test_thresholds_are_the_optimum_minus_five_percent(instances):
    for inst in instances.values():
        assert inst.data["err_threshold"] == math.floor(0.95 * inst.data["optimum"] * 1e4) / 1e4


def test_published_123go_value_passes_only_small_x(instances):
    """1-2-3-Go!'s trees reach 0.19999 on every instance: near-optimal while x <= 0.05, failing from x = 0.08."""
    passing = {name for name, inst in instances.items() if 0.19999 >= inst.data["err_threshold"]}
    assert passing == {"n10_x0.01", "n10_x0.03", "n10_x0.05", "n1_x0.01", "n20_x0.01", "n100_x0.01"}


def test_one_shot_schedule(domain, instances):
    """One try at the arbiter's shortcut reaches the error with probability 0.2 whatever x is."""
    rules = domain.reference_policies()["one_shot"]
    for name, certified in (("n10_x0.01", True), ("n10_x0.08", False)):
        verifier = PolicyVerifier(domain, instances[name], CFG)
        v = verifier.verify(verifier.spec.policy(rules), analysis=False)
        assert v.worst["err"] == pytest.approx(0.2, abs=1e-9)
        assert (v.worst["err"] >= instances[name].data["err_threshold"]) == certified


def test_extended_vocabulary(domain, instances):
    """`enabled` holds exactly where an action changes the state, so adding it to every rule changes nothing."""
    cfg = load_config(None, ["rules.extended=true"])
    verifier = PolicyVerifier(domain, instances["n10_x0.01"], cfg)
    spec = verifier.spec
    constants = {c.name: c.value for c in spec.constants}
    assert (constants["N"], constants["X_PCT"], constants["HOLDING"], constants["BOTH"]) == (10, 1, 3, 3)
    rules = [(f"({c}) & enabled", a) for c, a in domain.reference_policies()["one_shot"]]
    v = verifier.verify(spec.policy(rules, spec.vocabulary(cfg.rules)), analysis=False)
    assert v.worst["err"] == pytest.approx(0.2, abs=1e-9)


def test_general_rules_name_the_codes(domain, instances):
    """In the general language (no number above 1) the one-shot schedule is written with the named codes."""
    cfg = load_config("haiku_mi_mer")
    verifier = PolicyVerifier(domain, instances["n10_x0.01"], cfg)
    spec = verifier.spec
    vocabulary = spec.vocabulary(cfg.rules)
    with pytest.raises(RuleError):
        spec.policy([("s = 2 & gd", "u2_grant_drive")], vocabulary)
    rules = [("s = -1", "u1_grant_comm"), ("s = USER2 & gd", "u2_grant_drive"),
             ("s2 = IDLE & r2 = 0", "u2_want_drive"), ("s2 = IDLE & r2 = DRIVE", "u2_ready"),
             ("s2 = READY & s = 0", "u2_request_drive"), ("s1 = IDLE & r1 = 0", "u1_want_comm"),
             ("s1 = IDLE & r1 = COMM", "u1_ready"), ("s1 = READY & s = 0 & driveUser = USER2", "u1_request_comm")]
    v = verifier.verify(spec.policy(rules, vocabulary), analysis=False)
    assert v.worst["err"] == pytest.approx(0.2, abs=1e-9)


def test_sealed_instances_are_refused():
    from transfer import guard_sealed
    guard_sealed("mer_transfer.csv", list(range(9)))
    for ids in ([9], [10], [11], [0, 11]):
        with pytest.raises(SystemExit, match="sealed"):
            guard_sealed("mer_transfer.csv", ids)


def test_baseline_optimal_actions_make_progress(domain, instances):
    """A disabled action is a self-loop, so its value equals the state's; the baselines' optimal actions (maximizing)
    must still lead to the error, so the strategy of first optimal actions reaches the optimum."""
    import numpy as np
    from core.explicit import ExplicitMDP
    mdp = ExplicitMDP(domain, instances["n1_x0.01"], load_config("mer_baseline"))
    _, optimal = mdp.optimal_actions(mdp.spec.requirements[0])
    policy = np.where(optimal.any(axis=1), optimal.argmax(axis=1), -1)
    assert mdp.evaluate(policy)["err"] == pytest.approx(0.2, abs=1e-9)


def test_a_demonstration_takes_the_shortcut(domain, instances):
    """`prompt.demonstration` on n = 1: the optimal strategy's run, along the outcomes most likely to succeed, asks for
    comm while user 2 holds drive and takes the arbiter's shortcut (k becomes 1), which ends in the error."""
    from core.explicit import optimal_strategy
    from core.optimal import demonstration
    config = load_config("mer_baseline")
    steps = demonstration(PolicyVerifier(domain, instances["n1_x0.01"], config),
                          optimal_strategy(domain, instances["n1_x0.01"], config))
    assert [s.actions for s in steps[-2:]] == [("u1_request_comm",), ("u1_grant_comm",)]
    assert steps[-2].valuation["driveUser"] == 2 and steps[-1].valuation["k"] == 1


def test_n_family(domain):
    """Across n at x = 0.89 the optimum grows with the shortcut attempts: 0.2 at n = 1, 0.4438 at 3, 0.6944 at 100."""
    rows = {i.data["name"]: i.data for i in domain.load_instances("mer_n_transfer.csv")}
    assert [r["n"] for r in rows.values()] == [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 50, 100]
    assert rows["n1_x0.89"]["optimum"] == 0.2 and rows["n3_x0.89"]["optimum"] == pytest.approx(0.4437888)
    assert rows["n10_x0.89"]["optimum"] == pytest.approx(0.6711936689)        # 1-2-3-Go!'s published x = 0.89 point
    for r in rows.values():
        assert r["err_threshold"] == math.floor(0.95 * r["optimum"] * 1e4) / 1e4
    from transfer import guard_sealed
    guard_sealed("mer_n_transfer.csv", list(range(9)))
    with pytest.raises(SystemExit, match="sealed"):
        guard_sealed("mer_n_transfer.csv", [9])
