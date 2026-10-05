"""compare.py: per-sample metrics against each requirement's own bound, the report's summaries, and an
end-to-end run on the committed legacy and symbolic gridworld runs."""
import pandas as pd
import pytest

from compare import _metric_columns, _Stats, per_sample_table, render_report
from conftest import ROOT
from core.domain import Requirement

REACH = Requirement("reach", 'F "goal"', 0.8)
ENERGY = Requirement("energy", 'F "done"', 50.0, bound="<=", reward="energy")


def test_metric_columns_against_thresholds_and_what_is_achievable():
    row = {"legacy_reach": 0.7, "legacy_energy": 60.0, "symbolic_worst_reach": 0.9}
    optimum = pd.Series({"optimum_reach": 0.75, "optimum_energy": 55.0})   # neither threshold reachable
    out = _metric_columns(row, ["energy", "reach"], [REACH, ENERGY], optimum)
    assert out["legacy_met"] == 0 and out["legacy_shortfall"] == pytest.approx(0.1 + 10 / 50)
    assert out["legacy_achievable_shortfall"] == pytest.approx(0.05 + 5 / 55)   # vs 0.75 and <= 55
    assert (out["symbolic_worst_met"], out["symbolic_worst_shortfall"]) == (1, 0.0)
    assert "symbolic_best_met" not in out                                  # no best-case values in the row
    assert "legacy_achievable_shortfall" not in _metric_columns(row, ["energy", "reach"], [REACH, ENERGY], None)


def test_stats_format_and_report_missing_columns():
    s = _Stats(pd.DataFrame({"ok": [True, False, None], "x": [1.0, 2.0, 3.0], "solvable": [True, True, False]}))
    assert s.rate("ok") == "1/3" and s.mean("x") == "2.0" and s.median("x", "{:.0f}") == "2 [2–2]"
    assert s.solved_of_solvable("ok") == "1/2" and s.total("x") == 6
    assert s.rate("missing") == s.mean("missing") == s.median("missing") == s.total("missing") == "n/a"


def test_report_on_the_committed_gridworld_runs():
    legacy, symbolic = ROOT / "out/results/legacy_grid20", ROOT / "out/results/symbolic_grid20_capped"
    ceilings = pd.read_csv(ROOT / "out/results/ceilings/gridworld_grid_20_balanced.csv", index_col="sample_id")
    table, requirements = per_sample_table(legacy, symbolic, ceilings)
    assert list(table.index) == list(range(20)) and len(requirements) == 9
    assert table.legacy_met.between(0, 9).all() and table.symbolic_worst_met.le(table.symbolic_best_met).all()
    assert (table.symbolic_worst_achievable_shortfall <= table.symbolic_worst_shortfall + 1e-12).all()
    report = render_report(table, requirements, legacy, symbolic)
    for heading in ("- Samples: 20", "## Medians", "## Mean final probability per requirement", "## By grid size",
                    "| success, of jointly solvable instances | "):
        assert heading in report
