"""Draw the planned ablation grids (conditions x factors) for docs/ablations.md.

Usage (from PRISM-Guided-Learning/): python viz/plot_ablation_grid.py configs/plot/ablation_grid.yaml
Settings: the plot config (GPU-hour estimates, output files). Seeds come from configs/ablation/, grids,
rounds and model from configs/run/ (the defaults, and the D7 and R4 conditions), the badge families from
configs/plot/ablation_summary.yaml. Edit the row tables below when the plan changes.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from loaders import plot_config, short_model  # noqa: E402
from config import load_config, load_file  # noqa: E402
from core.domain import load_domain  # noqa: E402
from core.retry import RetryPolicy  # noqa: E402
from run_ablation import DEFAULT_CONFIG as ABLATION_CONFIG, AblationConfig  # noqa: E402
from theme import FAMILY_COLORS, GRID as RULE, INK, INK_2, SURFACE  # noqa: E402

INK_3 = "#8a8984"
CHANGED_FILL, CHANGED_INK = "#fde7dc", "#9a3412"   # differs from the row's reference
REF_FILL = "#f1f0ec"


@dataclass
class PlotConfig:
    """configs/plot/ablation_grid.yaml documents each key."""
    script: str
    gpu_hours: Dict[str, float]
    families_from: str
    out_run: str
    out_not_run: str

FACTORS = ["Policy form", "Feedback", "Retry trigger", "Prompt examples", "Blame signal"]
LEGACY_REF = ["per-state", "probabilities +\nprevious policy", "never", "2 worked examples", "—"]
SYMBOLIC_REF = ["rules", "blame +\nREFINE / EXTEND", "after 2 stalls", "rule examples", "mass"]

COLS = ["", "Condition"] + FACTORS + ["Question it answers", "Seeds", "GPU h"]
WIDTHS = [0.5, 1.95, 1.05, 1.55, 1.3, 1.6, 1.35, 3.45, 0.65, 0.75]


def sym(**changes):
    values = list(SYMBOLIC_REF)
    for k, v in changes.items():
        values[{"feedback": 1, "retry": 2, "examples": 3, "blame": 4}[k]] = v
    return values


def leg(**changes):
    values = list(LEGACY_REF)
    for k, v in changes.items():
        values[{"feedback": 1, "retry": 2, "examples": 3}[k]] = v
    return values


def run_rows(seeds: int, hours: Dict[str, float], rounds: int, d7_rounds: int) -> list:
    """Rows of the grid of conditions that ran:
    (id, name, family, values, question, seeds, GPU hours or None, is_reference)."""
    return [
        ("group", "Reference"),
        ("B2", "Symbolic defaults", "symbolic", SYMBOLIC_REF, "Main method; reference for every row below", seeds,
         seeds * hours["symbolic"], True),
        ("group", "Retry sweep  (vs. symbolic defaults; Marsha's suggestion)"),
        ("R1", "Never restart", "symbolic", sym(retry="never"), "Is feedback alone enough?", seeds, seeds * hours["symbolic"],
         False),
        ("R2", "Restart after 1 stall", "symbolic", sym(retry="after 1 stall"), "Restart sooner when stuck?", seeds,
         seeds * hours["symbolic"], False),
        ("R3", "Restart every 3rd round", "symbolic", sym(retry="every 3rd round"),
         "Scheduled restarts instead of stalls?", seeds, seeds * hours["symbolic"], False),
        ("R4", "Restart on small gain", "symbolic", sym(retry="gain < ε"), "Restart when progress is slow, not zero?",
         seeds, seeds * hours["symbolic"], False),
        ("R5", "Always restart", "symbolic", sym(feedback="none", retry="every round", blame="—"),
         "Is feedback better than resampling at all?", seeds, seeds * hours["symbolic"], False),
        ("group", "Feedback content  (vs. symbolic defaults)"),
        ("S1", "Results table only", "symbolic", sym(feedback="probabilities +\nprevious rules", blame="—"),
         "Does blame feedback help, beyond the table?", seeds, seeds * hours["symbolic"], False),
        ("S4", "No examples", "symbolic", sym(examples="none"), "Prompt confound, symbolic side", seeds,
         seeds * hours["symbolic"], False),
        ("group", "Blame signal  (vs. symbolic defaults)"),
        ("S5", "Regret blame", "symbolic", sym(blame="one-step regret"), "Does local blame beat mass?", seeds,
         seeds * hours["symbolic"], False),
        ("V1", "Random blame", "symbolic", sym(blame="random rules"), "Does blame need to point at the right rules?",
         seeds, seeds * hours["symbolic"], False),
        ("V2", "No blame section", "symbolic", sym(feedback="table +\nREFINE / EXTEND", blame="—"),
         "Does a blame hint help at all?", seeds, seeds * hours["symbolic"], False),
        ("group", "Rounds budget  (vs. restart on slow progress, the new default)"),
        ("D7", "New default, 7 rounds", "symbolic", sym(retry="gain < ε"), "Do more rounds keep paying off?", seeds,
         seeds * hours["symbolic"] * d7_rounds / rounds, False),
        ("group", "Free  (from the runs above)"),
        ("F1", f"Rounds budget 1–{rounds}", "both", ["—"] * 5, "Success vs budget (pass@k-style curves)", None, 0.0, False),
    ]


def not_run_rows(seeds: int, hours: Dict[str, float]) -> list:
    """Rows of the grid of conditions that did not run (same shape as run_rows)."""
    return [
        ("group", "Legacy  (stopped Oct 3: lower priority than the symbolic results)"),
        ("B1", "Legacy baseline", "legacy", LEGACY_REF, "Main comparison vs. symbolic", seeds,
         seeds * hours["legacy_visible"], True),
        ("L1", "Legacy + blind restart", "legacy", leg(retry="after 2 stalls"), "Does restarting alone close the gap?",
         seeds, seeds * hours["legacy_visible"], False),
        ("L2", "Legacy, no examples", "legacy", leg(examples="none"), "Prompt confound, legacy side", seeds,
         seeds * hours["legacy_visible"], False),
    ]


def draw(rows, title, subtitle, notes, out, families):
    row_h, head_h, group_h = 0.62, 0.55, 0.42
    n_rows = sum(1 for r in rows if r[0] != "group")
    n_groups = sum(1 for r in rows if r[0] == "group")
    fig_w = sum(WIDTHS) + 0.4
    fig_h = n_rows * row_h + n_groups * group_h + head_h + 1.6 + 0.3 * len(notes)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    left = 0.2
    xs = [left]
    for w in WIDTHS:
        xs.append(xs[-1] + w)
    top = fig_h - 1.3
    ax.text(left, fig_h - 0.45, title, fontsize=17, fontweight="bold", color=INK, va="center")
    ax.text(left, fig_h - 0.85, subtitle, fontsize=10, color=INK_2, va="center")

    for c, (name, x0, x1) in enumerate(zip(COLS, xs[:-1], xs[1:])):
        ax.text(x0 + 0.08, top - head_h / 2, name, fontsize=10, fontweight="bold", color=INK_2, va="center")
    ax.plot([left, xs[-1]], [top, top], color=RULE, lw=0.8)
    ax.plot([left, xs[-1]], [top - head_h, top - head_h], color=INK_3, lw=1)

    y = top - head_h
    total = 0.0
    spans = [(top - head_h, top)]
    for row in rows:
        if row[0] == "group":
            ax.text(left + 0.08, y - group_h / 2 - 0.02, row[1].upper(), fontsize=8.5, fontweight="bold",
                    color=INK_3, va="center")
            y -= group_h
            continue
        rid, name, family, values, question, seeds, hours, is_ref = row
        ref = LEGACY_REF if family == "legacy" else SYMBOLIC_REF
        y0, y1, mid = y - row_h, y, y - row_h / 2
        if is_ref:
            ax.add_patch(Rectangle((left, y0), xs[-1] - left, row_h, color=REF_FILL, lw=0, zorder=0))
        color = FAMILY_COLORS.get(families.get(rid), INK_3)
        ax.add_patch(FancyBboxPatch((xs[0] + 0.08, y0 + 0.16), xs[1] - xs[0] - 0.16, row_h - 0.32,
                                    boxstyle="round,pad=0,rounding_size=0.06", color=color, lw=0))
        ax.text((xs[0] + xs[1]) / 2, mid, rid, color="white", fontsize=9, fontweight="bold", ha="center", va="center")
        ax.text(xs[1] + 0.08, mid, name, fontsize=10.5, fontweight="bold" if is_ref else "normal", color=INK,
                va="center")
        for k, v in enumerate(values):
            cx0, cx1 = xs[2 + k], xs[3 + k]
            changed = not is_ref and family != "both" and v != ref[k]
            if changed:
                ax.add_patch(FancyBboxPatch((cx0 + 0.04, y0 + 0.07), cx1 - cx0 - 0.08, row_h - 0.14,
                                            boxstyle="round,pad=0,rounding_size=0.05", color=CHANGED_FILL, lw=0))
            ax.text(cx0 + 0.1, mid, v, fontsize=9, va="center", linespacing=1.15,
                    color=CHANGED_INK if changed else (INK_3 if v == "—" else INK_2),
                    fontweight="bold" if changed else "normal")
        ax.text(xs[7] + 0.08, mid, question, fontsize=9.5, color=INK, va="center")
        ax.text(xs[8] + 0.1, mid, str(seeds) if seeds else "—", fontsize=10, color=INK if seeds else INK_3, va="center")
        if hours is None:
            ax.text(xs[9] + 0.1, mid, "—", fontsize=10, color=INK_3, va="center")
        else:
            total += hours
            ax.text(xs[9] + 0.1, mid, f"{hours:.1f}", fontsize=10, color=INK, va="center")
        ax.plot([left, xs[-1]], [y0, y0], color=RULE, lw=0.8)
        ax.plot([left, xs[-1]], [y1, y1], color=RULE, lw=0.8)
        spans.append((y0, y1))
        y = y0

    for x in [xs[0]] + xs[2:]:
        for y0, y1 in spans:
            ax.plot([x, x], [y0, y1], color=RULE, lw=0.8, zorder=1)

    ax.plot([xs[8], xs[-1]], [y - 0.08, y - 0.08], color=INK_3, lw=1)
    ax.text(xs[9] + 0.1, y - 0.38, f"{total:.1f}", fontsize=11, fontweight="bold", color=INK, va="center")
    ax.text(xs[9] - 0.08, y - 0.38, "Total", fontsize=10, fontweight="bold", color=INK_2, va="center", ha="right")
    for i, note in enumerate(notes):
        ax.text(left, y - 0.38 - 0.3 * i, note, fontsize=8.5, color=INK_3, va="center")

    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(out)


def main():
    cfg = plot_config(PlotConfig, "plot_ablation_grid")
    hours = cfg.gpu_hours
    run = load_config()
    seeds = len(load_file(ABLATION_CONFIG, AblationConfig).seeds)
    grids = len(load_domain(run.domain.name).load_instances(run.domain.dataset))
    rounds, d7_rounds = run.planner.max_rounds, load_config("D7").planner.max_rounds
    gain = RetryPolicy.parse(load_config("R4").planner.retry).value
    conditions = yaml.safe_load(Path(cfg.families_from).read_text(encoding="utf-8"))["conditions"]
    families = {c: meta["family"] for c, meta in conditions.items()}
    out_run, out_not_run = Path(cfg.out_run), Path(cfg.out_not_run)
    out_run.parent.mkdir(exist_ok=True)
    out_not_run.parent.mkdir(exist_ok=True)

    common = (f"Each condition: {grids} gridworlds × {seeds} seeds, {rounds} rounds, {short_model(run.llm.model)}, "
              "obstacle phase visible to rules. Shaded cells differ from the row's reference.")
    draw(run_rows(seeds, hours, rounds, d7_rounds), "Ablations run (Oct 3)", common, [
        f"GPU hours: about {hours['symbolic']} h per {grids}-grid symbolic run (measured). Symbolic rows include the "
        f"joint best-case branch. R4's ε is the minimum drop in total worst-case shortfall that counts as progress "
        f"({gain:g}).",
        "S4 drops the example block, including the catch-all example rule (the instruction stays). V1 keeps the prompt's "
        "shape but blames random rules. Also run: U1, symbolic defaults on UUV with the energy budget.",
    ], out_run, families)
    draw(not_run_rows(seeds, hours), "Ablations not run", common, [
        f"GPU hours per legacy run: {hours['legacy_visible']} h (estimate: {hours['legacy']} h measured with the "
        "obstacle hidden, × mean cycle length 3.9). Also not run: a stronger model on a subset.",
    ], out_not_run, families)

if __name__ == "__main__":
    main()
