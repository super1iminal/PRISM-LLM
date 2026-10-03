"""Draw the planned ablation grids (conditions x factors) for docs/ablations.md.

Usage (from PRISM-Guided-Learning/): python viz/plot_ablation_grid.py
Writes docs/ablation_now.png and docs/ablation_deferred.png at the repo root.
Edit the ROWS tables below when the plan changes.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

SURFACE, INK, INK_2, INK_3, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
LEGACY, SYMBOLIC = "#2a78d6", "#eb6834"
CHANGED_FILL, CHANGED_INK = "#fde7dc", "#9a3412"   # differs from the row's reference
REF_FILL = "#f1f0ec"
DOCS = Path(__file__).resolve().parents[2] / "docs"

SEEDS = 2
H_LEGACY, H_SYMBOLIC = 1.15, 0.5   # measured GPU hours per 20-grid run (2 workers), obstacle hidden
H_LEGACY_VISIBLE = 4.5             # estimate: legacy writes one action per (cell, obstacle phase); mean cycle 3.9

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


# Row: (id, name, family, values, question, seeds, gpu hours or None, is_reference)
NOW = [
    ("group", "1. Symbolic  (main method; reference for the sweep)"),
    ("B2", "Symbolic (full)", "symbolic", SYMBOLIC_REF, "Main method; reference for the sweep", SEEDS,
     SEEDS * H_SYMBOLIC, True),
    ("group", "2. Retry-policy sweep  (vs. B2; Marsha's suggestion)"),
    ("R1", "Never retry", "symbolic", sym(retry="never"), "Is feedback alone enough?", SEEDS, SEEDS * H_SYMBOLIC, False),
    ("R2", "Retry after 1 stall", "symbolic", sym(retry="after 1 stall"), "Retry sooner when stuck?", SEEDS,
     SEEDS * H_SYMBOLIC, False),
    ("R3", "Retry every 3rd round", "symbolic", sym(retry="every 3rd round"), "Scheduled restarts instead of stalls?",
     SEEDS, SEEDS * H_SYMBOLIC, False),
    ("R4", "Retry on small gain", "symbolic", sym(retry="gain < ε"), "Restart when progress is slow, not zero?",
     SEEDS, SEEDS * H_SYMBOLIC, False),
    ("R5", "Retry every round", "symbolic", sym(feedback="none", retry="every round", blame="—"),
     "Is feedback better than resampling at all?", SEEDS, SEEDS * H_SYMBOLIC, False),
    ("group", "3. Legacy  (after all symbolic runs)"),
    ("B1", "Legacy", "legacy", LEGACY_REF, "Baseline for the paper's comparison", SEEDS,
     SEEDS * H_LEGACY_VISIBLE, True),
    ("group", "Free  (from the runs above)"),
    ("F1", "Rounds budget 1–5", "both", ["—"] * 5, "Success vs budget (pass@k-style curves)", None, 0.0, False),
]

DEFERRED = [
    ("group", "References  (from the main comparison)"),
    ("B1", "Legacy", "legacy", LEGACY_REF, "Reference for legacy ablations", None, None, True),
    ("B2", "Symbolic (full)", "symbolic", SYMBOLIC_REF, "Reference for symbolic ablations", None, None, True),
    ("group", "Symbolic ablations  (vs. B2)"),
    ("S1", "Legacy-style feedback", "symbolic", sym(feedback="probabilities +\nprevious rules", blame="—"),
     "Does blame feedback help, beyond the table?", SEEDS, SEEDS * H_SYMBOLIC, False),
    ("S4", "No examples", "symbolic", sym(examples="none"), "Prompt confound, symbolic side", SEEDS,
     SEEDS * H_SYMBOLIC, False),
    ("S5", "Regret blame", "symbolic", sym(blame="one-step regret"), "Does local blame beat mass?", SEEDS,
     SEEDS * H_SYMBOLIC, False),
    ("group", "Legacy ablations  (vs. B1)"),
    ("L1", "Legacy + blind retry", "legacy", leg(retry="after 2 stalls"), "Does retry alone close the gap?", SEEDS,
     SEEDS * H_LEGACY_VISIBLE, False),
    ("L2", "Legacy, no examples", "legacy", leg(examples="none"), "Prompt confound, legacy side", SEEDS,
     SEEDS * H_LEGACY_VISIBLE, False),
]


def draw(rows, title, subtitle, notes, out):
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
        color = {"legacy": LEGACY, "symbolic": SYMBOLIC}.get(family, INK_3)
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
    DOCS.mkdir(exist_ok=True)
    common = (f"Each condition: 20 gridworlds × {SEEDS} seeds, 5 rounds, qwen3:14b, obstacle phase visible to rules. "
              "Shaded cells differ from the row's reference.")
    draw(NOW, "Ablations: now", common, [
        f"GPU hours per 20-grid run: symbolic {H_SYMBOLIC} h (measured); legacy {H_LEGACY_VISIBLE} h "
        f"(estimate: {H_LEGACY} h measured with the obstacle hidden, × mean cycle length 3.9).",
        "Run in group order (symbolic first). Symbolic rows include the joint best-case branch. R4's ε is the minimum drop in total worst-case "
        "shortfall that counts as progress (proposed 0.05).",
    ], DOCS / "ablation_now.png")
    draw(DEFERRED, "Ablations: deferred", common, [
        "Deferred at the Sept 24 meeting until the story is settled. S4 drops the example block, including the "
        "catch-all example rule (the instruction stays).",
        "Also deferred: blame validation (top-k blamed vs random vs no hint, a separate one-round protocol) and "
        "a stronger model on a subset.",
    ], DOCS / "ablation_deferred.png")


if __name__ == "__main__":
    main()
