"""Symbolic runs on any domain vs non-LLM references, per instance and requirement.

Usage (from PRISM-Guided-Learning/):
    python viz/plot_domain.py --domain uuv --data uuv_paper.csv \
        --symbolic "qwen=out/results/symbolic_uuv" --out viz/figures/uuv.png

Each panel is one instance x requirement. The grey band is what any controller can achieve on
the bare MDP (min..max), the dashed line is the threshold. Dots are worst-case values (black
ticks: best case) of each symbolic run's final policy and of the domain's reference policies
(`domains/<domain>/data/reference_policies.json`, if present). Also writes a markdown table.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from core.domain import load_domain  # noqa: E402
from core.rules import SymbolicPolicy  # noqa: E402
from core.verifier import PolicyVerifier  # noqa: E402

# Reference categorical palette, slots 1-3 (validated all-pairs in light and dark)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#eeede9"


def final_results(run_dir: Path) -> dict:
    """sample_id -> (final worst, final best, success) from a symbolic run's parquet."""
    df = pd.read_parquet(run_dir / "SYMBOLIC_results.parquet").reset_index()
    out = {}
    for sid, f in df[df.is_final].set_index("sample_id").iterrows():
        reqs = [c[len("final_worst_"):] for c in f.index if c.startswith("final_worst_")]
        out[int(sid)] = ({r: f[f"final_worst_{r}"] for r in reqs}, {r: f[f"final_best_{r}"] for r in reqs},
                         bool(f["success"]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--symbolic", action="append", default=[], help="label=path, repeatable (max 3)")
    parser.add_argument("--title", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    domain = load_domain(args.domain)
    instances = domain.load_instances(args.data)
    runs = [(label, final_results(Path(path))) for label, path in (s.split("=", 1) for s in args.symbolic)]
    ref_path = domain.root / "data" / "reference_policies.json"
    references = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else {}

    reqs = domain.spec(instances[0]).requirements
    height = (0.45 * (len(runs) + len(references)) + 1.6) * len(instances)
    fig, axes = plt.subplots(len(instances), len(reqs), figsize=(5.2 * len(reqs), height), squeeze=False,
                             facecolor=SURFACE)
    table = []
    for i, inst in enumerate(instances):
        verifier = PolicyVerifier(domain, inst)
        spec = verifier.spec
        props = [f"{op}=? [ {r.formula} ]" for r in spec.requirements for op in ("Pmin", "Pmax")]
        bounds = verifier.runner.run(verifier.model, props).initial_values
        rows = []   # (label, worst, best, color, success)
        for k, (label, res) in enumerate(runs):
            worst, best, ok = res[i]
            rows.append((label, worst, best, COLORS[k], ok))
        for name, rules in references.items():
            v = verifier.verify(SymbolicPolicy.from_raw(spec.variables, list(spec.actions), rules), analysis=False)
            ok = all(r.satisfied(v.worst[r.name]) for r in spec.requirements)
            rows.append((name, v.worst, v.best, INK_2, ok))
        for label, worst, best, _, ok in rows:
            table.append({"instance": inst.data.get("name", inst.id), "policy": label, "success": ok,
                          **{r.name: round(worst[r.name], 4) for r in spec.requirements}})

        for j, req in enumerate(spec.requirements):
            ax = axes[i][j]
            lo, hi = bounds[2 * j], bounds[2 * j + 1]
            ax.axvspan(lo, hi, color=BAND, zorder=0)
            ax.axvline(req.threshold, color=INK, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
            for y, (label, worst, best, color, ok) in enumerate(rows):
                ax.plot([worst[req.name]], [y], "o", markersize=8, color=color, markeredgecolor=SURFACE,
                        markeredgewidth=2, zorder=3)
                if abs(best[req.name] - worst[req.name]) > 1e-9:
                    ax.plot([best[req.name]] * 2, [y - 0.25, y + 0.25], color=INK, linewidth=1.5, zorder=2)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([f"{label} {'✓' if ok else '✗'}" if j == 0 else label for label, *_, ok in rows])
            ax.invert_yaxis()
            # Zoom on the policies and the threshold; the band continues past the panel edges
            shown = [req.threshold, hi] + [w[req.name] for _, w, *_ in rows] + [b[req.name] for _, _, b, *_ in rows]
            left, right = max(lo, min(shown)), max(shown)
            pad = (right - left) * 0.15 or 0.01
            ax.set_xlim(left - pad, right + pad)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.set_title(f"{inst.data.get('name', inst.id)}: {req.name} ({req.bound} {req.threshold})",
                         loc="left", fontsize=10, color=INK, fontweight="bold")
            ax.set_facecolor(SURFACE)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(colors=INK_2, labelsize=9, length=0)
    fig.suptitle((args.title or f"{args.domain}: worst-case probability of each final policy") +
                 "\ngrey band = what any controller achieves, dashed = threshold, black tick = best case, "
                 "✓ = all requirements met", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    cols = list(table[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(row[c]) for c in cols) + " |" for row in table]
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
