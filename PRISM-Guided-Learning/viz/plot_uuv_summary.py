"""UUV: our symbolic policy vs the paper's controller, on probability, policy size and the paper's costs.

Usage (from PRISM-Guided-Learning/):
    python viz/plot_uuv_summary.py configs/plot/uuv_summary.yaml
Settings: the plot config (PlotConfig below). PRISM settings: the run config.

Left: verified probability of our final policy against the paper's controller at its best case
(PRISM's maximum per requirement; each is maximized separately, so no single controller reaches
all of them), for each probability requirement. Right: policy size, rules vs the number of states
where an optimal PRISM strategy must choose. Also writes a markdown table with the paper's Table 2
measures (expected energy and time to finish) for each policy, whether it meets every requirement
(reward ones such as the energy budget included), and our North Sea rules transferred unchanged to
the Caribbean.
"""
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import load_config, plot_config  # noqa: E402
from core.domain import load_domain  # noqa: E402
from core.prism import PrismRunner  # noqa: E402
from core.rules import SymbolicPolicy  # noqa: E402
from core.verifier import PolicyVerifier  # noqa: E402
from loaders import short_model  # noqa: E402
from results_io import run_facts  # noqa: E402
from theme import GRID, INK, INK_2, SURFACE, style  # noqa: E402,F401

BLUE, ORANGE = "#2a78d6", "#eb6834"
COSTS = ['R{"energy"}min=? [ F "done" ]', 'R{"energy"}max=? [ F "done" ]',
         'R{"time"}min=? [ F "done" ]', 'R{"time"}max=? [ F "done" ]']
TICK_LABELS = {"no_thruster_failure": "no thruster failure", "done_in_time": "done within {deadline} steps"}


@dataclass
class PlotConfig:
    """A configs/plot/*.yaml file for this script (configs/plot/uuv_summary.yaml documents each key)."""
    script: str
    symbolic: str
    dataset: str
    prism_method: str
    out: str


def evaluate(verifier, rules):
    """Worst-case requirement values and expected (energy, time) to finish of a full rule policy."""
    spec = verifier.spec
    policy = SymbolicPolicy.from_raw(spec.variables, list(spec.actions), rules)
    v = verifier.verify(policy, analysis=False)
    costs = verifier.runner.run(verifier.compose(policy), COSTS).initial_values
    return v, costs


def policy_row(name, label, v, costs, probabilities, requirements, size):
    meets = "yes" if all(r.satisfied(v.worst[r.name]) for r in requirements) else "no"
    return (name, label, *[f"{v.worst[r.name]:.3f}" for r in probabilities], f"{costs[0]:.2f}", f"{costs[2]:.2f}",
            meets, size)


def main():
    cfg = plot_config(PlotConfig, "plot_uuv_summary")
    symbolic, out = Path(cfg.symbolic), Path(cfg.out)
    model = short_model(run_facts(symbolic).model)

    domain = load_domain("uuv")
    instances = domain.load_instances(cfg.dataset)
    ours = [[(r["condition"], r["action"]) for r in json.loads(p.read_text(encoding="utf-8"))["final_rules"]]
            for p in sorted((symbolic / "outputs").glob("sample_*.json"))]
    run = load_config()
    runner = PrismRunner(replace(run.prism, method=cfg.prism_method))   # the paper's solver, for its Table 2

    fig = plt.figure(figsize=(13, 4.8), facecolor=SURFACE)
    grid = fig.add_gridspec(1, len(instances) + 1, width_ratios=[1.2] * len(instances) + [1])
    rows, sizes, budgets = [], [], []
    for i, inst in enumerate(instances):
        verifier = PolicyVerifier(domain, inst, run, runner)
        reqs = verifier.spec.requirements
        probabilities = [r for r in reqs if r.reward is None]   # rewards (energy) are in the table, not the axis
        n = 2 * len(probabilities)
        bare = verifier.runner.run(verifier.model, [f"{op}=? [ {r.formula} ]" for r in probabilities
                                                    for op in ("Pmin", "Pmax")] + COSTS).initial_values
        v_ours, c_ours = evaluate(verifier, ours[i])
        name = inst.data["name"].replace("_", " ").title()
        budgets += [f"{name} {r.name} {r.bound} {r.threshold}" for r in reqs if r.reward is not None]

        ax = fig.add_subplot(grid[0, i])
        for j, r in enumerate(probabilities):
            for dx, val, color in ((-0.19, bare[2 * j + 1], ORANGE), (0.19, v_ours.worst[r.name], BLUE)):
                ax.bar(j + dx, val, width=0.34, color=color, zorder=2)
                ax.text(j + dx, val - 0.03, f"{val:.3f}", ha="center", va="top", fontsize=9, fontweight="bold",
                        color=SURFACE, zorder=5)
            ax.plot([j - 0.42, j + 0.42], [r.threshold] * 2, color=INK, linestyle=(0, (4, 3)), linewidth=1.2, zorder=4)
        ax.set_xticks(range(len(probabilities)))
        ax.set_xticklabels([TICK_LABELS.get(r.name, r.name.replace("_", " ")).format(**inst.data)
                            for r in probabilities])
        ax.set_ylim(0, 1)
        style(ax, name, pad=None)
        if i == 0:
            ax.set_ylabel("probability", color=INK_2, fontsize=9)

        decision_states = len(verifier.optimum()[2].states) - len(verifier.forced_states())
        sizes.append((name, len(ours[i]), decision_states))
        rows.append((name, "paper controller (range)", *[f"{bare[2 * j]:.3f}..{bare[2 * j + 1]:.3f}"
                                                         for j in range(len(probabilities))],
                     f"{bare[n]:.2f}..{bare[n + 1]:.2f}", f"{bare[n + 2]:.2f}..{bare[n + 3]:.2f}", "—",
                     f">= {decision_states} states"))
        rows.append(policy_row(name, f"ours ({model})", v_ours, c_ours, probabilities, reqs, f"{len(ours[i])} rules"))
        if i == 1:
            v_tr, c_tr = evaluate(verifier, ours[0])
            rows.append(policy_row(name, "ours, North Sea rules reused", v_tr, c_tr, probabilities, reqs,
                                   f"{len(ours[0])} rules"))

    ax = fig.add_subplot(grid[0, len(instances)])
    for k, (name, n_ours, n_opt) in enumerate(sizes):
        for dx, val, color in ((-0.19, n_opt, ORANGE), (0.19, n_ours, BLUE)):
            ax.bar(k + dx, val, width=0.34, color=color, zorder=2)
            ax.text(k + dx, val * 1.15, f"{val:,}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(1, 10 ** 5)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([s[0] for s in sizes])
    style(ax, "Policy size", pad=None)
    ax.set_ylabel("rules / strategy states (log)", color=INK_2, fontsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ORANGE, BLUE)]
    handles.append(plt.Line2D([], [], color=INK, linestyle=(0, (4, 3)), linewidth=1.2))
    fig.legend(handles, ["paper (best case)", "ours", "required"],
               loc="upper left", bbox_to_anchor=(0.01, 0.94), frameon=False, fontsize=9, ncol=3)
    fig.suptitle("UUV pipeline inspection: ours vs paper", x=0.01, ha="left", fontsize=12, color=INK,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)

    header = ("scenario", "policy", *[f"P({r.name.replace('_', ' ')})" for r in probabilities], "E[energy] to done",
              "E[time] to done", "meets every requirement", "size")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    note = ("\nPaper controller rows give its min..max over all resolutions (energy/time = the paper's Table 2). "
            "Policy rows are exact (fully covering policies: best = worst)."
            + (f" Reward requirements: {'; '.join(budgets)}." if budgets else "") + "\n")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n" + note, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
