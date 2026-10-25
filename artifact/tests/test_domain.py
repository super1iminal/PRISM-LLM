"""The Domain base class: reading a dataset, the reference policies, and parsing rules against a spec."""
import pytest

from core.domain import load_domain
from core.rules import RuleError, SymbolicPolicy


def test_a_dataset_is_read_by_name_or_by_absolute_path(tiny_grid):
    domain = load_domain("gridworld")
    by_path = domain.read_dataset(str(tiny_grid))
    assert list(by_path.columns) == ["n", "goals", "static", "moving", "BFS_steps"] and len(by_path) == 1
    assert len(domain.read_dataset("grid_20_balanced.csv")) == 20
    assert [i.data["n"] for i in domain.load_instances(str(tiny_grid))] == [4]


def test_reference_policies_come_from_the_domains_data_folder():
    uuv = load_domain("uuv").reference_policies()
    assert {"stay", "always_high"} <= set(uuv) and all(len(rule) == 2 for rule in uuv["stay"])
    assert load_domain("gridworld").reference_policies() == {}   # gridworld has none


def test_a_spec_parses_rules_over_its_variables_and_actions(tiny_grid):
    domain = load_domain("gridworld", ["obs_idx"])
    spec = domain.spec(domain.load_instances(str(tiny_grid))[0])
    rules = [("x = 0 & !g1", "right"), ("true", "down")]
    policy = spec.policy(rules)
    assert policy.to_dicts() == SymbolicPolicy.from_raw(spec.variables, list(spec.actions), rules).to_dicts()
    assert [v.name for v in policy.variables] == [v.name for v in spec.variables]
    with pytest.raises(RuleError, match="unknown variable 'z'"):
        spec.policy([("z > 1", "up")])
