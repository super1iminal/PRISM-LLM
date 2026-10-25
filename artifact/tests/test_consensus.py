"""The consensus domain reproduces the QVBS benchmark's published values, and one generic rule list certifies the
whole family across K."""
import math
import shutil
from dataclasses import replace

import pytest

from config import load_config
from core.domain import Instance, load_domain
from core.prism import PrismRunner
from core.rules import RuleError
from core.verifier import PolicyVerifier

CFG = load_config()

pytestmark = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")


@pytest.fixture(scope="module")
def domain():
    return load_domain("consensus")


@pytest.fixture(scope="module")
def instances(domain):
    return {i.data["name"]: i for i in domain.load_instances("consensus_transfer.csv")}


def bare(n: int, k: int) -> Instance:
    return Instance("0", {"name": f"n{n}_k{k}", "n": n, "k": k, "optimum": 0.0, "disagree_threshold": 0.0})


@pytest.mark.parametrize("n, k, states, disagree, c2", [(2, 2, 272, 13 / 120, 49 / 128),
                                                         (2, 4, 528, 251 / 4080, 1793 / 4096),
                                                         (4, 2, 22656, 170112531 / 577765376, 325 / 1024)])
def test_bare_mdp_matches_published_values(domain, n, k, states, disagree, c2):
    """QVBS (benchmarks/mdp/consensus, Storm/exact): the state count and the properties disagree (Pmax) and c2
    (Pmin). Every label moves some process until all have finished, so the policy keeps the original choices. The
    random walk leaks probability slowly, so the default solver stops early: Gauss-Seidel at 1e-12 here."""
    props = ['Pmax=? [ F "finished" & !"agree" ]', 'Pmin=? [ F "finished" & "all_coins_equal_1" ]']
    exact = replace(CFG.prism, method="", max_iters=CFG.prism.exact_max_iters)
    runner = PrismRunner(exact, extra_args=["-gaussseidel", "-epsilon", "1e-12"])
    result = runner.run(domain.model(bare(n, k)), props, export_transitions=True)
    assert len(result.states) == states
    assert result.initial_values[0] == pytest.approx(disagree, abs=1e-6)
    assert result.initial_values[1] == pytest.approx(c2, abs=1e-6)


def test_dataset(instances):
    """N = 3 throughout; thresholds are the optimum minus 5% of it; from K = 8 on the optimum is 1/(2K)."""
    rows = [i.data for i in instances.values()]
    assert [r["k"] for r in rows] == [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48]
    assert {r["n"] for r in rows} == {3}
    for r in rows:
        assert r["disagree_threshold"] == math.floor(0.95 * r["optimum"] * 1e4) / 1e4
        if r["k"] >= 8:
            assert r["optimum"] == pytest.approx(1 / (2 * r["k"]), abs=1e-6)


def test_sealed_instances_are_refused():
    from transfer import guard_sealed
    guard_sealed("consensus_transfer.csv", list(range(9)))
    for ids in ([9], [10], [11], [0, 11]):
        with pytest.raises(SystemExit, match="sealed"):
            guard_sealed("consensus_transfer.csv", ids)


# A generic schedule, written with the optimal values in hand (a check that the family can be solved in the rule
# language, as for the gridworld families; not a baseline): aim at the value nobody has decided yet, or at the nearer
# barrier while nobody has decided; prefer writes towards it, then checks that decide it, re-flips, flips.
AIM = {"TAILS": "(heads_decided | (!tails_decided & to_heads > to_tails))",
       "HEADS": "(tails_decided | (!heads_decided & to_heads <= to_tails))"}
RACE = [("heads_decided & tails_decided", "any")] + [
    rule for v, decides in (("TAILS", "to_tails <= 0"), ("HEADS", "to_heads <= 0")) for rule in (
        (f"{AIM[v]} & stage = WRITE & coin = {v}", "any"), (f"{AIM[v]} & stage = CHECK & {decides}", "any"),
        (f"{AIM[v]} & stage = CHECK & to_tails > 0 & to_heads > 0", "any"), (f"{AIM[v]} & stage = FLIP", "any"),
        (f"{AIM[v]} & stage = CHECK", "any"), (f"{AIM[v]} & stage = WRITE", "any"))]


@pytest.mark.parametrize("name, value", [("n3_k2", 0.219425), ("n3_k8", 0.0625)])
def test_one_generic_list_certifies(domain, instances, name, value):
    """The list above certifies small and large K, in the worst case over which process its `any` rules move."""
    cfg = load_config("consensus_baseline", ["rules.general=true"])
    verifier = PolicyVerifier(domain, instances[name], cfg)
    spec = verifier.spec
    v, _ = verifier.verify_exact(spec.policy(RACE, spec.vocabulary(cfg.rules)))
    assert v.worst["disagree"] == pytest.approx(value, abs=1e-5)
    assert v.worst["disagree"] >= instances[name].data["disagree_threshold"]


def test_general_rules_name_the_barriers(domain, instances):
    """In the general language the barriers are named constants; their values move with K."""
    cfg = load_config("consensus_baseline", ["rules.general=true"])
    for name, (left, right, start) in (("n3_k2", (3, 15, 9)), ("n3_k16", (3, 99, 51))):
        spec = PolicyVerifier(domain, instances[name], cfg).spec
        constants = {c.name: c.value for c in spec.constants}
        assert (constants["LEFT"], constants["RIGHT"], constants["START"], constants["CHECK"]) == (left, right, start, 2)
        with pytest.raises(RuleError):
            spec.policy([(f"counter >= {right}", "p1")], spec.vocabulary(cfg.rules))
