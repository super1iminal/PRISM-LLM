"""Symbolic runs on any domain vs non-LLM references, per instance and requirement.

Usage (from PRISM-Guided-Learning/):
    python viz/plot_domain.py configs/plot/uuv.yaml
Settings: the plot config (PlotConfig below). PRISM settings for the reference policies: the run config.

Each panel is one instance x requirement. The grey band is what any controller can achieve on
the bare MDP (min..max), the dashed line is the threshold. Dots are worst-case values (black
ticks: best case) of each symbolic run's final policy and of the domain's reference policies
(`domains/<domain>/data/reference_policies.json`, if present). Also writes a markdown table.
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import load_config, plot_config  # noqa: E402
from core.domain import load_domain  # noqa: E402
from core.rules import SymbolicPolicy  # noqa: E402
from core.verifier import PolicyVerifier  # noqa: E402
from results_io import SYMBOLIC_RESULTS, run_facts  # noqa: E402
from theme import GRID, INK, INK_2, SURFACE  # noqa: E402

# Reference categorical palette, slots 1-3 (validated all-pairs in light and dark)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
BAND = "#eeede9"


@dataclass
class PlotConfig:
    """A configs/plot/*.yaml file for this script (configs/plot/uuv.yaml documents each key)."""
    script: str
    domain: str
    dataset: str
    symbolic: Dict[str, str]
    title: Optional[str]
    out: str


def final_results(run_dir: Path) -> dict:
    """sample_id -> (final worst, final best, success) from a symbolic run's parquet."""
    df = pd.read_parquet(run_dir / SYMBOLIC_RESULTS).reset_index()
    out = {}
    for sid, f in df[df.is_final].set_index("sample_id").iterrows():
        reqs = [c[len("final_worst_"):] for c in f.index if c.startswith("final_worst_")]
        out[int(sid)] = ({r: f[f"final_worst_{r}"] for r in reqs}, {r: f[f"final_best_{r}"] for r in reqs},
                         bool(f["success"]))
    return out


def main():
    cfg = plot_config(PlotConfig, "plot_domain")
    out = Path(cfg.out)
    for label, path in cfg.symbolic.items():
        facts = run_facts(Path(path))
        if facts.recorded and (facts.domain, facts.dataset) != (cfg.domain, cfg.dataset):
            raise SystemExit(f"{label} ({path}) ran {facts.domain} / {facts.dataset}, not {cfg.domain} / {cfg.dataset}")

    domain = load_domain(cfg.domain)
    instances = domain.load_instances(cfg.dataset)
    runs = [(label, final_results(Path(path))) for label, path in cfg.symbolic.items()]
    ref_path = domain.root / "data" / "reference_policies.json"
    references = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else {}

    reqs = domain.spec(instances[0]).requirements
    height = (0.45 * (len(runs) + len(references)) + 1.6) * len(instances)
    fig, axes = plt.subplots(len(instances), len(reqs), figsize=(5.2 * len(reqs), height), squeeze=False,
                             facecolor=SURFACE)
    table = []
    for i, inst in enumerate(instances):
        verifier = PolicyVerifier(domain, inst, load_config())
        spec = verifier.spec
        # Band = min..max any controller achieves: Pmin/Pmax, or R{..}min/max for a reward requirement
        props = [f"{op}=? [ {r.formula} ]" for r in spec.requirements
                 for op in ((r.worst_op(), r.best_op()) if r.maximize else (r.best_op(), r.worst_op()))]
        bounds = verifier.runner.run(verifier.model, props).initial_values
        rows = []   # (label, worst, best, color, success)
        for k, (label, res) in enumerate(runs):
            worst, best, ok = res[i]
            missing = [r.name for r in spec.requirements if r.name not in worst]   # runs older than a requirement
            rows.append((f"{label} (no {', '.join(missing)})" if missing else label, worst, best, COLORS[k], ok))
        for name, rules in references.items():
            v = verifier.verify(SymbolicPolicy.from_raw(spec.variables, list(spec.actions), rules), analysis=False)
            ok = all(r.satisfied(v.worst[r.name]) for r in spec.requirements)
            rows.append((name, v.worst, v.best, INK_2, ok))
        for label, worst, best, _, ok in rows:
            table.append({"instance": inst.data.get("name", inst.id), "policy": label, "success": ok,
                          **{r.name: round(worst[r.name], 4) if r.name in worst else "—" for r in spec.requirements}})

        for j, req in enumerate(spec.requirements):
            ax = axes[i][j]
            lo, hi = bounds[2 * j], bounds[2 * j + 1]
            ax.axvspan(lo, hi, color=BAND, zorder=0)
            ax.axvline(req.threshold, color=INK, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
            present = [(y, worst, best) for y, (_, worst, best, *_) in enumerate(rows) if req.name in worst]
            for y, worst, best in present:
                ax.plot([worst[req.name]], [y], "o", markersize=8, color=rows[y][3], markeredgecolor=SURFACE,
                        markeredgewidth=2, zorder=3)
                if abs(best[req.name] - worst[req.name]) > 1e-9:
                    ax.plot([best[req.name]] * 2, [y - 0.25, y + 0.25], color=INK, linewidth=1.5, zorder=2)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([f"{label} {'✓' if ok else '✗'}" for label, *_, ok in rows] if j == 0 else [])
            ax.set_ylim(len(rows) - 0.5, -0.5)   # first row on top
            # Zoom on the policies, the threshold and the band's good end; the band continues past the edges
            shown = ([req.threshold, hi if req.maximize else lo] + [w[req.name] for _, w, _ in present]
                     + [b[req.name] for _, _, b in present])
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
    fig.suptitle((cfg.title or f"{cfg.domain}: worst-case value of each final policy") +
                 "\ngrey band = what any controller achieves, dashed = threshold, black tick = best case, "
                 "✓ = all requirements met", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    cols = list(table[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(row[c]) for c in cols) + " |" for row in table]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
