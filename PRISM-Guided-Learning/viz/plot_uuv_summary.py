"""UUV: our symbolic policy vs the paper's controller, on probability, policy size and the paper's costs.

Usage (from PRISM-Guided-Learning/):
    python viz/plot_uuv_summary.py --symbolic out/results/symbolic_uuv --out viz/figures/uuv_summary.png

Left: worst-case probability of our final policy and of the hand-written `stay` reference, against
the paper's nondeterministic controller (bar = its worst case, whisker = its best case; each
requirement is maximized separately, so no single controller reaches both best cases).
Right: policy size, rules vs the number of states where an optimal PRISM strategy must choose.
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

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
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
    stay = json.loads((domain.root / "data" / "reference_policies.json").read_text(encoding="utf-8"))["stay"]
    ours = [[(r["condition"], r["action"]) for r in json.loads(p.read_text(encoding="utf-8"))["final_rules"]]
            for p in sorted((args.symbolic / "outputs").glob("sample_*.json"))]

    fig = plt.figure(figsize=(14, 5.2), facecolor=SURFACE)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.35, 1])
    rows, sizes = [], []
    for i, inst in enumerate(instances):
        verifier = PolicyVerifier(domain, inst)
        reqs = verifier.spec.requirements
        bare = verifier.runner.run(verifier.model, [f"{op}=? [ {r.formula} ]" for r in reqs for op in ("Pmin", "Pmax")]
                                   + COSTS).initial_values
        v_ours, c_ours = evaluate(verifier, ours[i])
        v_stay, c_stay = evaluate(verifier, stay)
        name = inst.data["name"].replace("_", " ").title()

        ax = fig.add_subplot(grid[0, i])
        for j, r in enumerate(reqs):
            lo, hi = bare[2 * j], bare[2 * j + 1]
            bars = [(-0.27, lo, ORANGE), (0, v_ours.worst[r.name], BLUE), (0.27, v_stay.worst[r.name], GREEN)]
            for dx, val, color in bars:
                ax.bar(j + dx, val, width=0.24, color=color, zorder=2)
                inside = val > 0.12
                ax.text(j + dx, val - 0.03 if inside else val + 0.015, f"{val:.3f}", ha="center",
                        va="top" if inside else "bottom", fontsize=8, fontweight="bold", zorder=5,
                        color=SURFACE if inside else INK, rotation=90 if inside else 0)
            ax.plot([j - 0.27] * 2, [lo, hi], color=INK_2, linewidth=1.2, zorder=3)
            ax.plot([j - 0.33, j - 0.21], [hi] * 2, color=INK_2, linewidth=1.2, zorder=3)
            ax.text(j - 0.27, hi + 0.012, f"best {hi:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK_2)
            ax.plot([j - 0.42, j + 0.42], [r.threshold] * 2, color=INK, linestyle=(0, (4, 3)), linewidth=1.2, zorder=4)
        ax.set_xticks(range(len(reqs)))
        ax.set_xticklabels(["no thruster failure", f"done within {inst.data['deadline']} steps"])
        ax.set_ylim(0, 1.08)
        ok = all(r.satisfied(v_ours.worst[r.name]) for r in reqs)
        style(ax, f"{name}: ours {'meets both' if ok else 'misses one'} (dashed = needed)")
        if i == 0:
            ax.set_ylabel("probability (policies: guaranteed worst case)", color=INK_2, fontsize=9)

        decision_states = len(verifier.optimum()[2].states) - len(verifier.forced_states())
        sizes.append((name, len(ours[i]), len(stay), decision_states))
        rows.append((name, "paper controller (range)", f"{bare[0]:.3f}..{bare[1]:.3f}", f"{bare[2]:.3f}..{bare[3]:.3f}",
                     f"{bare[4]:.2f}..{bare[5]:.2f}", f"{bare[6]:.2f}..{bare[7]:.2f}", f">= {decision_states} states"))
        for label, v, c, n in (("ours (qwen3:14b)", v_ours, c_ours, len(ours[i])), ("stay (hand-written)", v_stay, c_stay, len(stay))):
            rows.append((name, label, f"{v.worst[reqs[0].name]:.3f}", f"{v.worst[reqs[1].name]:.3f}",
                         f"{c[0]:.2f}", f"{c[2]:.2f}", f"{n} rules"))
        if i == 1:
            v_tr, c_tr = evaluate(verifier, ours[0])
            rows.append((name, "ours, North Sea rules reused", f"{v_tr.worst[reqs[0].name]:.3f}",
                         f"{v_tr.worst[reqs[1].name]:.3f}", f"{c_tr[0]:.2f}", f"{c_tr[2]:.2f}", f"{len(ours[0])} rules"))

    ax = fig.add_subplot(grid[0, 2])
    for k, (name, n_ours, n_stay, n_opt) in enumerate(sizes):
        for dx, val, color in ((-0.27, n_opt, ORANGE), (0, n_ours, BLUE), (0.27, n_stay, GREEN)):
            ax.bar(k + dx, val, width=0.24, color=color, zorder=2)
            ax.text(k + dx, val * 1.15, f"{val:,}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(1, 10 ** 5)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([s[0] for s in sizes])
    style(ax, "Policy size (log scale)")
    ax.set_ylabel("rules, or states an optimal strategy must decide", color=INK_2, fontsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ORANGE, BLUE, GREEN)]
    fig.legend(handles, ["paper's controller (Päßler et al. 2023): worst case, whisker to best case; size = PRISM strategy table",
                         "ours: qwen3:14b symbolic rules", "stay: hand-written 4-rule reference"],
               loc="upper left", bbox_to_anchor=(0.01, 0.93), frameon=False, fontsize=9, ncol=3)
    fig.suptitle("UUV pipeline inspection: synthesized rules vs the paper's adaptation logic "
                 "(best cases maximize each requirement separately; no single controller reaches both)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
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
