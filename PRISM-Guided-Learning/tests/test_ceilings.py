"""Per-instance ceilings: optimum of each requirement and joint feasibility on the bare MDP."""
import shutil

import pytest

from ceilings import ceilings
from config import load_config
from core.domain import load_domain
from core.verifier import PolicyVerifier

needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")


@needs_prism
def test_gridworld_ceilings(tiny_grid):
    cfg = load_config(overrides=[f"domain.dataset={tiny_grid.as_posix()}"])
    df = ceilings(cfg)
    row = df.loc[0]
    domain = load_domain("gridworld")
    optimum = PolicyVerifier(domain, domain.load_instances(str(tiny_grid))[0], cfg).optimum()[0]
    for name, value in optimum.items():
        assert row[f"optimum_{name}"] == pytest.approx(value, abs=1e-9)
        assert row[f"achievable_{name}"] == (value >= row[f"threshold_{name}"])
    assert row["individually_feasible"] and df.jointly_feasible.tolist() == [True]


@needs_prism
def test_undecided_joint_query_is_reported_not_raised():
    df = ceilings(load_config("U1"))   # UUV's step-bounded deadline: PRISM's LP method cannot decide
    assert len(df) == 2 and df.jointly_feasible.isna().all()
    assert {"optimum_no_thruster_failure", "achievable_done_in_time"} <= set(df.columns)
