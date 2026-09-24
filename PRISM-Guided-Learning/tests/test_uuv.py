"""The UUV domain reproduces the paper's results and its thresholds separate the reference policies."""
import json
import shutil
from pathlib import Path

import pytest

from core.domain import load_domain
from core.prism import PrismRunner
from core.rules import SymbolicPolicy
from core.verifier import PolicyVerifier

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")

# Paessler et al. (iFM 2023), Sec. 4: Pmin(F done), Table 2 (energy/time min/max), Pmin(G safe)
PAPER = {
    "north_sea": [1.0, 24.78, 44.39, 23.66, 32.40, 0.65],
    "caribbean_sea": [1.0, 59.08, 4723.29, 55.54, 1315.58, 0.32],
}
PAPER_PROPS = ['Pmin=? [ F "done" ]', 'R{"energy"}min=? [ F "done" ]', 'R{"energy"}max=? [ F "done" ]',
               'R{"time"}min=? [ F "done" ]', 'R{"time"}max=? [ F "done" ]', 'Pmin=? [ G !"thruster_failure" ]']

# Keep the current altitude while searching; pick the highest allowed one when a search starts
STAY = json.loads((ROOT / "domains/uuv/data/reference_policies.json").read_text(encoding="utf-8"))["stay"]


@pytest.fixture(scope="module")
def domain():
    return load_domain("uuv")


@pytest.mark.parametrize("sample", [0, 1])
def test_bare_mdp_matches_paper(domain, sample):
    instance = domain.load_instances("uuv_paper.csv")[sample]
    values = PrismRunner().run(domain.model(instance), PAPER_PROPS).initial_values
    for value, expected in zip(values, PAPER[instance.data["name"]]):
        assert round(value, 2) == pytest.approx(expected)


@pytest.mark.parametrize("sample", [0, 1])
def test_thresholds_separate_reference_policies(domain, sample):
    instance = domain.load_instances("uuv_paper.csv")[sample]
    verifier = PolicyVerifier(domain, instance)
    spec = verifier.spec

    def passes(rules):
        v = verifier.verify(SymbolicPolicy.from_raw(spec.variables, list(spec.actions), rules), analysis=False)
        assert v.uncovered_situations == 0
        return {r.name: r.satisfied(v.worst[r.name]) for r in spec.requirements}

    assert passes(STAY) == {"no_thruster_failure": True, "done_in_time": True}
    assert passes([("true", "low")])["no_thruster_failure"] is False
    assert passes([("true", "high")])["done_in_time"] is False


def test_forced_states_are_not_situations(domain):
    """Rules for the search phase alone cover the policy: following, found and done are forced."""
    instance = domain.load_instances("uuv_paper.csv")[0]
    verifier = PolicyVerifier(domain, instance)
    spec = verifier.spec
    search_only = SymbolicPolicy.from_raw(spec.variables, list(spec.actions),
                                          [("s = 11 | s <= 3 | (s >= 6 & s <= 8)", "high")])
    v = verifier.verify(search_only, analysis=False)
    assert v.uncovered_situations == 0
    assert v.best == v.worst
    assert not all(v.decisions)
