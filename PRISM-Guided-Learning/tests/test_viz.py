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
