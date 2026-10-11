"""PolicyVerifier helpers that read PRISM's state tuples, and `failing` on verification results."""
import shutil

import pytest

from config import load_config
from core.domain import Requirement, failing, load_domain
from core.prism import PrismResult
from core.verifier import PolicyVerifier

needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")


def test_positions_map_variable_names_to_tuple_slots():
    result = PrismResult(["obs_idx", "g2", "x"], [(1, False, 2)], [], [])
    assert result.positions == {"obs_idx": 0, "g2": 1, "x": 2}
    assert result.positions is result.positions   # computed once per result


@needs_prism
def test_policy_valuation_follows_spec_order_whatever_prism_order(tiny_grid):
    domain = load_domain("gridworld", ["obs_idx"])
    verifier = PolicyVerifier(domain, domain.load_instances(str(tiny_grid))[0], load_config())
    names = [v.name for v in verifier.spec.variables]
    assert names == ["x", "y", "g1", "g2", "obs_idx"]
    prism_order = ["obs_idx", "g2", "y", "g1", "x", "hidden"]
    result = PrismResult(prism_order, [(1, True, 3, False, 2, 7), (0, False, 0, False, 0, 0)], [], [])
    assert verifier.policy_valuation(result, 0) == {"x": 2, "y": 3, "g1": False, "g2": True, "obs_idx": 1}
    assert list(verifier.policy_valuation(result, 1)) == names


def test_failing_respects_each_bound():
    reach = Requirement("reach", 'F "goal"', 0.8)
    energy = Requirement("energy", 'F "done"', 50.0, bound="<=", reward="energy")
    assert failing([reach, energy], {"reach": 0.9, "energy": 40.0}) == []
    assert failing([reach, energy], {"reach": 0.8, "energy": 50.0}) == []   # thresholds are inclusive
    assert failing([reach, energy], {"reach": 0.79, "energy": 50.5}) == [reach, energy]
    assert failing([energy, reach], {"reach": 0.5, "energy": 10.0}) == [reach]


@needs_prism
def test_a_prism_timeout_moves_the_exact_check_to_its_fallback_solver(tiny_grid):
    """A timeout reaches the verifier as subprocess.TimeoutExpired thrown into its generator, at the point
    where subprocess.run used to raise it, so the exact check still falls back to Gauss-Seidel."""
    import subprocess

    from core.prism import run_call
    domain = load_domain("gridworld", ["obs_idx"])
    verifier = PolicyVerifier(domain, domain.load_instances(str(tiny_grid))[0], load_config())
    policy = verifier.empty_policy()
    steps, timed_out = verifier.verify_exact_steps(policy), []
    try:
        call = next(steps)
        while True:
            if "-intervaliter" in call.cmd:
                timed_out.append(call)
                call = steps.throw(subprocess.TimeoutExpired(list(call.cmd), call.timeout))
            else:
                call = steps.send(run_call(call))
    except StopIteration as done:
        exact, solver = done.value
    assert len(timed_out) == 1 and solver == "Gauss-Seidel 1e-12"
    assert exact.worst == pytest.approx(verifier.verify_exact(policy)[0].worst, abs=1e-6)
