"""The gridworld domain MDP + a full atomic policy must reproduce the legacy DTMC exactly."""
import logging
import random
import shutil

import pytest

from core.domain import load_domain
from core.prism import PrismRunner
from core.rules import SymbolicPolicy
from core.verifier import PolicyVerifier
from domains.gridworld.legacy_translate import legacy_policy_to_rules
from legacy.gridworld import GridWorld as LegacyGridWorld
from legacy.prism_model import PrismModelGenerator
from legacy.requirements import SimplifiedVerifier

pytestmark = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")

LOG = logging.getLogger("test")


@pytest.mark.parametrize("sample", [0, 7, 16])
def test_random_policy_matches_legacy(sample):
    domain = load_domain("gridworld")
    instance = domain.load_instances("grid_20_balanced.csv")[sample]
    d = instance.data
    rng = random.Random(sample)
    states = [(x, y, g1, g2, g3) for x in range(d["n"]) for y in range(d["n"])
              for g1 in (False, True) for g2 in (False, True) for g3 in (False, True)]
    legacy_policy = [(s, rng.randrange(4)) for s in states]

    runner = PrismRunner(extra_args=["-epsilon", "1e-10"])
    gw = LegacyGridWorld(d["n"], d["goals"], d["static"], d["moving"])
    model = PrismModelGenerator(gw, LOG).generate_prism_model(dict(legacy_policy))
    legacy = runner.run(model, [r.property for r in SimplifiedVerifier(None, gw, LOG).requirements]).initial_values

    verifier = PolicyVerifier(domain, instance, runner)
    policy = SymbolicPolicy.from_raw(verifier.spec.variables, list(verifier.spec.actions),
                                     legacy_policy_to_rules(legacy_policy, 3))
    v = verifier.verify(policy, analysis=False)
    assert v.uncovered_situations == 0
    for req, expected in zip(verifier.spec.requirements, legacy):
        assert v.best[req.name] == pytest.approx(expected, abs=1e-7)
        assert v.worst[req.name] == pytest.approx(expected, abs=1e-7)


def test_partial_policy_bounds():
    """Dropping rules can only widen [worst, best] around the full policy's value."""
    domain = load_domain("gridworld")
    instance = domain.load_instances("grid_20_balanced.csv")[0]
    verifier = PolicyVerifier(domain, instance)
    spec = verifier.spec
    full = SymbolicPolicy.from_raw(spec.variables, list(spec.actions),
                                   [("!g1 & y < 1", "right"), ("!g1", "up"), ("g1 & !g2", "down"),
                                    ("g2 & !g3 & x > 1", "up"), ("true", "right")])
    partial = SymbolicPolicy.from_raw(spec.variables, list(spec.actions), [(r.condition, r.action) for r in full.rules[:3]])
    vf, vp = verifier.verify(full, analysis=False), verifier.verify(partial, analysis=False)
    for req in spec.requirements:
        assert vp.worst[req.name] <= vf.worst[req.name] + 1e-6
        assert vp.best[req.name] >= vf.best[req.name] - 1e-6
