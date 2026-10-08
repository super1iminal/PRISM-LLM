"""The Pac-Man domain reproduces the QVBS benchmark and its thresholds separate the reference policies."""
import json
import shutil
from pathlib import Path

import pytest

from config import load_config
from core.domain import Instance, load_domain
from core.prism import PrismRunner
from core.rules import SymbolicPolicy
from core.verifier import PolicyVerifier

ROOT = Path(__file__).resolve().parent.parent
CFG = load_config()
REFERENCE = json.loads((ROOT / "domains/pacman/data/reference_policies.json").read_text(encoding="utf-8"))

pytestmark = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")


@pytest.fixture(scope="module")
def domain():
    return load_domain("pacman")


def instance(max_steps, threshold=0.56):
    return Instance(id="0", data={"name": f"steps_{max_steps}", "max_steps": max_steps, "crash_threshold": threshold})


def test_bare_mdp_matches_qvbs(domain):
    """QVBS reports 498 states (Storm) and Pmin(F Crash) = 5511/10000 for MAXSTEPS = 5."""
    result = PrismRunner(CFG.prism).run(domain.model(instance(5)), ['Pmin=? [ F "Crash" ]'], export_transitions=True)
    assert len(result.states) == 498
    assert result.initial_values[0] == pytest.approx(0.5511, abs=1e-9)


def test_crash_range_over_controllers(domain):
    """Computed from the unchanged QVBS file with PRISM 4.10.1: the minimum stays 0.5511, the maximum grows."""
    verifier = PolicyVerifier(domain, instance(10), CFG)
    best, worst = verifier.runner.run(verifier.model, ['Pmin=? [ F "Crash" ]', 'Pmax=? [ F "Crash" ]']).initial_values
    assert (best, worst) == (pytest.approx(0.5511, abs=1e-9), pytest.approx(0.8844, abs=1e-9))


def test_decisions_only_at_crossings(domain):
    module = type(domain).context.__globals__
    verifier = PolicyVerifier(domain, instance(10), CFG)
    v = verifier.verify(verifier.empty_policy())
    pos = v.result.positions
    cells = {(s[pos["xP"]], s[pos["yP"]]) for s in v.result.states}
    deciding = {(s[pos["xP"]], s[pos["yP"]]) for s, d in zip(v.result.states, v.decisions) if d}
    assert cells <= set(domain.walkable())
    assert deciding <= set(module["CROSSINGS"]) and len(deciding) > 1


def test_thresholds_separate_reference_policies(domain):
    def crash(max_steps, rules):
        verifier = PolicyVerifier(domain, instance(max_steps), CFG)
        spec = verifier.spec
        return verifier.verify(SymbolicPolicy.from_raw(spec.variables, list(spec.actions), rules),
                               analysis=False).worst["crash"]

    assert crash(8, REFERENCE["keep_heading"]) <= 0.56                        # short games are easy
    assert all(crash(12, rules) > 0.56 for rules in REFERENCE.values())       # no simple policy survives 12 moves


def test_horizon(domain):
    assert domain.horizon(instance(10)) == 33
