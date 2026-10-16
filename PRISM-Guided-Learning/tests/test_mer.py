"""The MER domain reproduces 1-2-3-Go!'s published optima, and its thresholds separate x <= 0.05 from x >= 0.08."""
import math
import shutil

import pytest

from config import load_config
from core.domain import load_domain
from core.prism import PrismRunner
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
    assert [c.name for c in spec.constants] == ["N", "X_PCT"] and [c.value for c in spec.constants] == [10, 1]
    rules = [(f"({c}) & enabled", a) for c, a in domain.reference_policies()["one_shot"]]
    v = verifier.verify(spec.policy(rules, spec.vocabulary(cfg.rules)), analysis=False)
    assert v.worst["err"] == pytest.approx(0.2, abs=1e-9)


def test_sealed_instances_are_refused():
    from transfer import guard_sealed
    guard_sealed("mer_transfer.csv", list(range(9)))
    for ids in ([9], [10], [11], [0, 11]):
        with pytest.raises(SystemExit, match="sealed"):
            guard_sealed("mer_transfer.csv", ids)
