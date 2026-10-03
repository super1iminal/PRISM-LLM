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


def test_uuv_horizon_is_the_deadline():
    cfg = load_config()
    uuv, grid = load_domain("uuv"), load_domain("gridworld")
    north_sea, caribbean = uuv.load_instances("uuv_paper.csv")
    assert cfg.feedback.horizon_for(uuv, north_sea) == 30 and cfg.feedback.horizon_for(uuv, caribbean) == 70
    assert cfg.feedback.horizon_for(grid, grid.load_instances("grid_20_balanced.csv")[0]) == 100
    assert load_config(overrides=["feedback.horizon_by_domain={uuv: 12}"]).feedback.horizon_for(uuv, north_sea) == 12


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
