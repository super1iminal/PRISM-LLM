"""UUV: our symbolic policy vs the paper's controller, on probability, policy size and the paper's costs.

Usage (from PRISM-Guided-Learning/):
    python viz/plot_uuv_summary.py --symbolic out/results/symbolic_uuv --out viz/figures/uuv_summary.png

Left: verified probability of our final policy against the paper's controller at its best case
(PRISM's maximum per requirement; each is maximized separately, so no single controller reaches
both). Right: policy size, rules vs the number of states where an optimal PRISM strategy must choose.
Also writes a markdown table with the paper's Table 2 measures (expected energy and time to
finish) for each policy, and our North Sea rules transferred unchanged to the Caribbean.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from core.domain import load_domain  # noqa: E402
from core.rules import SymbolicPolicy  # noqa: E402
from core.verifier import PolicyVerifier  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COSTS = ['R{"energy"}min=? [ F "done" ]', 'R{"energy"}max=? [ F "done" ]',
         'R{"time"}min=? [ F "done" ]', 'R{"time"}max=? [ F "done" ]']


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def evaluate(verifier, rules):
    """Worst-case probabilities and expected (energy, time) to finish of a full rule policy."""
    spec = verifier.spec
    policy = SymbolicPolicy.from_raw(spec.variables, list(spec.actions), rules)
    v = verifier.verify(policy, analysis=False)
    costs = verifier.runner.run(verifier.model + "\n" + policy.to_prism_module() + "\n", COSTS).initial_values
    return v, costs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbolic", type=Path, required=True)
    parser.add_argument("--data", default="uuv_paper.csv")
    parser.add_argument("--out", type=Path, default=Path("viz/figures/uuv_summary.png"))
    args = parser.parse_args()

    domain = load_domain("uuv")
    instances = domain.load_instances(args.data)
    ours = [[(r["condition"], r["action"]) for r in json.loads(p.read_text(encoding="utf-8"))["final_rules"]]
            for p in sorted((args.symbolic / "outputs").glob("sample_*.json"))]

    fig = plt.figure(figsize=(13, 4.8), facecolor=SURFACE)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1])
    rows, sizes = [], []
    for i, inst in enumerate(instances):
        verifier = PolicyVerifier(domain, inst)
        reqs = verifier.spec.requirements
        bare = verifier.runner.run(verifier.model, [f"{op}=? [ {r.formula} ]" for r in reqs for op in ("Pmin", "Pmax")]
                                   + COSTS).initial_values
        v_ours, c_ours = evaluate(verifier, ours[i])
        name = inst.data["name"].replace("_", " ").title()

        ax = fig.add_subplot(grid[0, i])
        for j, r in enumerate(reqs):
            for dx, val, color in ((-0.19, bare[2 * j + 1], ORANGE), (0.19, v_ours.worst[r.name], BLUE)):
                ax.bar(j + dx, val, width=0.34, color=color, zorder=2)
                ax.text(j + dx, val - 0.03, f"{val:.3f}", ha="center", va="top", fontsize=9, fontweight="bold",
                        color=SURFACE, zorder=5)
            ax.plot([j - 0.42, j + 0.42], [r.threshold] * 2, color=INK, linestyle=(0, (4, 3)), linewidth=1.2, zorder=4)
        ax.set_xticks(range(len(reqs)))
        ax.set_xticklabels(["no thruster failure", f"done within {inst.data['deadline']} steps"])
        ax.set_ylim(0, 1)
        style(ax, name)
        if i == 0:
            ax.set_ylabel("probability", color=INK_2, fontsize=9)

        decision_states = len(verifier.optimum()[2].states) - len(verifier.forced_states())
        sizes.append((name, len(ours[i]), decision_states))
        rows.append((name, "paper controller (range)", f"{bare[0]:.3f}..{bare[1]:.3f}", f"{bare[2]:.3f}..{bare[3]:.3f}",
                     f"{bare[4]:.2f}..{bare[5]:.2f}", f"{bare[6]:.2f}..{bare[7]:.2f}", f">= {decision_states} states"))
        rows.append((name, "ours (qwen3:14b)", f"{v_ours.worst[reqs[0].name]:.3f}", f"{v_ours.worst[reqs[1].name]:.3f}",
                     f"{c_ours[0]:.2f}", f"{c_ours[2]:.2f}", f"{len(ours[i])} rules"))
        if i == 1:
            v_tr, c_tr = evaluate(verifier, ours[0])
            rows.append((name, "ours, North Sea rules reused", f"{v_tr.worst[reqs[0].name]:.3f}",
                         f"{v_tr.worst[reqs[1].name]:.3f}", f"{c_tr[0]:.2f}", f"{c_tr[2]:.2f}", f"{len(ours[0])} rules"))

    ax = fig.add_subplot(grid[0, 2])
    for k, (name, n_ours, n_opt) in enumerate(sizes):
        for dx, val, color in ((-0.19, n_opt, ORANGE), (0.19, n_ours, BLUE)):
            ax.bar(k + dx, val, width=0.34, color=color, zorder=2)
            ax.text(k + dx, val * 1.15, f"{val:,}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(1, 10 ** 5)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([s[0] for s in sizes])
    style(ax, "Policy size")
    ax.set_ylabel("rules / strategy states (log)", color=INK_2, fontsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ORANGE, BLUE)]
    handles.append(plt.Line2D([], [], color=INK, linestyle=(0, (4, 3)), linewidth=1.2))
    fig.legend(handles, ["paper (best case)", "ours", "required"],
               loc="upper left", bbox_to_anchor=(0.01, 0.94), frameon=False, fontsize=9, ncol=3)
    fig.suptitle("UUV pipeline inspection: ours vs paper", x=0.01, ha="left", fontsize=12, color=INK,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)

    header = ("scenario", "policy", "P(no thruster failure)", "P(done in time)", "E[energy] to done",
              "E[time] to done", "size")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    note = ("\nPaper controller rows give its min..max over all resolutions (energy/time = the paper's Table 2). "
            "Policy rows are exact (fully covering policies: best = worst).\n")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n" + note, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
