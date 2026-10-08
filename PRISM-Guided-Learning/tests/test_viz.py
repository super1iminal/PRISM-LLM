"""Figure scripts: the shared axes style, and that every script still imports."""
import importlib
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viz"))
from theme import GRID, INK_2, SURFACE, style  # noqa: E402


def test_style():
    fig, ax = plt.subplots()
    style(ax, "Title", "unit")
    assert ax.get_title(loc="left") == "Title" and ax.get_ylabel() == "unit"
    assert [ax.spines[s].get_visible() for s in ("top", "right", "left", "bottom")] == [False, False, False, True]
    assert matplotlib.colors.to_hex(ax.get_facecolor()) == SURFACE
    assert matplotlib.colors.to_hex(ax.spines["bottom"].get_edgecolor()) == GRID
    assert matplotlib.colors.to_hex(ax.yaxis.label.get_color()) == INK_2
    plt.close(fig)


def test_style_without_ylabel_or_pad_keeps_matplotlib_defaults():
    fig, (plain, default_pad, padded) = plt.subplots(1, 3)
    plain.set_title("Title", loc="left")
    default_pad.set_ylabel("kept")
    style(default_pad, "Title", pad=None)
    style(padded, "Title")
    assert default_pad.get_ylabel() == "kept"
    assert np.allclose(default_pad.titleOffsetTrans.get_matrix(), plain.titleOffsetTrans.get_matrix())
    assert not np.allclose(padded.titleOffsetTrans.get_matrix(), plain.titleOffsetTrans.get_matrix())
    plt.close(fig)


@pytest.mark.parametrize("module", ["ablation_summary", "plot_ablation_grid", "plot_budget", "plot_comparison",
                                    "plot_domain", "plot_runs", "plot_uuv_summary"])
def test_figure_scripts_import(module):
    importlib.import_module(module)


def test_holm_matches_the_step_down_rule():
    from ablation_summary import holm
    p = np.array([0.04, 0.01, 0.03, 0.20])
    # sorted 0.01, 0.03, 0.04, 0.20 -> x4, x3, x2, x1, each at least the previous: 0.04, 0.09, 0.09, 0.20
    assert np.allclose(holm(p), [0.09, 0.04, 0.09, 0.20])
    assert holm(np.array([0.6, 0.5]))[1] == 1.0   # capped at 1


def test_at_budget_takes_the_last_round_that_fits_and_at_least_the_first():
    import pandas as pd
    from plot_budget import at_budget, with_cost
    curve = pd.DataFrame({"seed": "s", "sample_id": [0, 0, 0, 1, 1], "k": [1, 2, 3, 1, 2], "met": [3, 5, 7, 4, 6],
                          "input_tokens": [100, 200, 300, 500, 900], "output_tokens": [0, 0, 0, 0, 0]})
    curve = with_cost(curve, {"input": 1, "output": 5})
    rows = at_budget(curve, 250, ("seed", "sample_id")).set_index("sample_id")
    assert rows.met.to_dict() == {0: 5, 1: 4}            # sample 1's first round costs 500 > 250, kept anyway
    per_grid = at_budget(curve, pd.Series({0: 1000, 1: 600}), ("seed", "sample_id")).set_index("sample_id")
    assert per_grid.met.to_dict() == {0: 7, 1: 4}        # all rounds of sample 0 fit; sample 1's second doesn't
