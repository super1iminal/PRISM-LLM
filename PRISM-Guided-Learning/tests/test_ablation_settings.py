"""Settings behind the deferred ablations: blame signals, table feedback, legacy retry/examples, horizon."""
import shutil

import numpy as np
import pytest

from config import load_config
from core.analysis import MassAnalyzer
from core.domain import load_domain
from core.rules import SymbolicPolicy
from core.verifier import PolicyVerifier
from legacy.prompting import get_prompt

needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")


def test_horizon_is_steps_or_the_domains_own():
    uuv, grid = load_domain("uuv"), load_domain("gridworld")
    north_sea, caribbean = uuv.load_instances("uuv_paper.csv")
    grid_instance = grid.load_instances("grid_20_balanced.csv")[0]
    default, u1 = load_config().feedback, load_config("U1").feedback
    assert default.horizon_for(grid, grid_instance) == 100 == default.horizon_for(uuv, north_sea)
    assert u1.horizon_for(uuv, north_sea) == 30 and u1.horizon_for(uuv, caribbean) == 70   # the mission deadline
    assert load_config("U1", ["feedback.horizon=12"]).feedback.horizon_for(uuv, north_sea) == 12
    with pytest.raises(ValueError, match="no horizon of its own"):
        u1.horizon_for(grid, grid_instance)


def test_legacy_examples_switch():
    args = (4, [(1, 1)], [], [], (3, 3))
    assert "=== EXAMPLE 1 ===" in get_prompt(*args) and "=== EXAMPLE 1 ===" not in get_prompt(*args, examples=False)


@needs_prism
@pytest.mark.parametrize("method", ["mass", "regret", "random"])
def test_blame_methods(method):
    domain = load_domain("gridworld", ["obs_idx"])
    verifier = PolicyVerifier(domain, domain.load_instances("grid_20_balanced.csv")[0])
    spec = verifier.spec
    policy = SymbolicPolicy.from_raw(spec.variables, list(spec.actions), [("!g1", "left"), ("g1 & !g2", "left")])
    v = verifier.verify(policy)
    failing = [r for r in spec.requirements if not r.satisfied(v.best[r.name])]
    assert failing
    blame = MassAnalyzer(verifier, method=method, seed=1).rule_blame(v, failing)
    assert blame and np.isclose(sum(b.mass for b in blame), 1.0)
    empty = verifier.verify(verifier.empty_policy())   # everything uncovered: best and worst differ
    hotspots = MassAnalyzer(verifier, method=method, seed=1).uncovered_hotspots(empty, spec.requirements)
    assert hotspots and 0 < sum(h.mass for h in hotspots) <= 1 + 1e-9   # top k of the shares
    if method == "regret":   # "left" before goal 1 is bad, so rule 1 carries regret
        assert 0 in [b.rule for b in blame]
