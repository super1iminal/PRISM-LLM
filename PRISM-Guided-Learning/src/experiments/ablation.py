"""RQ2: what each mechanism contributes. One factor at a time against a reference cell, per benchmark: held-out
targets certified per rule list (mean over runs, range), the difference to the reference with a bootstrap interval
over runs, the same with ties left open, the library, training sets solved and tokens. Figure: one panel per
benchmark, each run a dot, the reference's mean a dashed line.
"""
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from experiments.base import Experiment, held_out, span
from experiments.store import ABLATIONS_PATH
from settings import PROJECT_ROOT
from transfer import parse_ids


def bootstrap_diff(values, reference, reps: int = 10000, seed: int = 0) -> Tuple[float, float, float]:
    """Mean of `values` minus mean of `reference`, and a 90% percentile bootstrap interval (resampling the runs of
    each side independently)."""
    a, b = np.asarray(values, dtype=float), np.asarray(reference, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if not len(a) or not len(b):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = rng.choice(a, (reps, len(a))).mean(axis=1) - rng.choice(b, (reps, len(b))).mean(axis=1)
    return a.mean() - b.mean(), *np.percentile(diffs, [5, 95])


class Ablation(Experiment):
    root = ABLATIONS_PATH

    def reference(self) -> str:
        return self.spec.get("reference") or self.spec["cells"][0].name

    def _of(self, bench) -> str:
        """What `certified` counts out of on a benchmark."""
        if bench.groups:
            sizes = {len(parse_ids(t)) for t in bench.groups.values()}
            return f"of {sizes.pop()} in its own group" if len(sizes) == 1 else "in its own group"
        return f"of {len(held_out(bench))}"

    def report(self, runs: pd.DataFrame) -> str:
        ref = self.reference()
        lines = [f"# {self.title}", "", self.spec.get("question", "").strip(), ""]
        if self.spec.get("budget"):        # planned comparisons at equal cost and the budget figure (plan 03)
            from budget_transfer import report_section
            lines += report_section(self)
        lines += [f"Reference: **{ref}**. Certified: held-out targets certified per rule list (mean over runs, range "
                 "over runs). Δ: difference of means to the reference, with a 90% bootstrap interval over runs (few "
                 "runs: read it as a rough guide). Ties open: certified with the action orders left open, i.e. "
                 "however ties are broken. Library: targets certified by at least one of a run's lists.", ""]
        for bench in self.spec["benchmarks"].values():
            b = runs[runs.bench == bench.name] if not runs.empty else runs
            if b.empty:
                continue
            ref_values = b[b.cell == ref].certified
            lines += [f"## {bench.label or bench.name}", "",
                      f"Targets {bench.targets} (each list's own training instances left out); certified {self._of(bench)}.", "",
                      "| condition | code | runs | certified | Δ vs reference | ties open | library | sets solved | "
                      "rounds / set | output tokens / set |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            cells = [c for c in self.spec["cells"] if bench.name in c.only]
            for cell in cells:
                g = b[b.cell == cell.name]
                if g.empty:
                    lines.append(f"| {cell.name} | {cell.code} | 0 | – | – | – | – | – | – | – |")
                    continue
                unchecked = int((~g.checked.fillna(False).astype(bool)).sum())
                if cell.name == ref:
                    delta = "reference"
                else:
                    d, lo, hi = bootstrap_diff(g.certified, ref_values)
                    delta = "–" if np.isnan(d) else f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                lines.append(f"| {cell.name}{f' ({unchecked} unchecked)' if unchecked else ''} | {cell.code} | "
                             f"{len(g)} | {span(g.certified, '.2f')} | {delta} | {span(g.certified_open, '.2f')} | "
                             f"{span(g.library, '.1f')} | {span(g.solved / g.sets, '.0%')} | "
                             f"{span(g.rounds, '.1f')} | {span(g.output_tokens, ',.0f')} |")
            if bench.groups:
                lines += ["", "Per group (certified per rule list on its own group, mean over runs):", "",
                          "| condition | " + " | ".join(bench.groups) + " |", "|---|" + "---|" * len(bench.groups)]
                for cell in cells:
                    g = b[b.cell == cell.name]
                    cells = [pd.Series([d.get(k) for d in g.by_group.dropna()], dtype=float).mean()
                             for k in bench.groups] if not g.empty else [float("nan")] * len(bench.groups)
                    lines.append(f"| {cell.name} | " + " | ".join("–" if pd.isna(c) else f"{c:.1f}" for c in cells)
                                 + " |")
            lines.append("")
        figure = self.figure(runs)
        if figure:
            lines += [f"![{self.title}]({figure.name})", ""]
        return "\n".join(lines)

    def figure(self, runs: pd.DataFrame):
        if runs.empty:
            return None
        sys.path.insert(0, str(PROJECT_ROOT / "viz"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from theme import GRID, INK, INK_2, SURFACE, style

        benches = [b for b in self.spec["benchmarks"].values() if (runs.bench == b.name).any()]
        cells = self.spec["cells"]
        ref = self.reference()
        fig, axes = plt.subplots(1, len(benches), figsize=(4.2 * len(benches) + 1.8, 0.42 * len(cells) + 1.4),
                                 sharey=True, squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        for ax, bench in zip(axes[0], benches):
            b = runs[runs.bench == bench.name]
            style(ax, bench.label or bench.name)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            ref_mean = b[b.cell == ref].certified.mean()
            if not np.isnan(ref_mean):
                ax.axvline(ref_mean, color=INK_2, linestyle="--", linewidth=1)
            for y, cell in enumerate(cells):
                values = b[b.cell == cell.name].certified.dropna()
                if values.empty:
                    continue
                color = "#2a78d6" if cell.name == ref else INK
                ax.plot([values.min(), values.max()], [y, y], color=GRID, linewidth=5, solid_capstyle="round")
                ax.scatter(values, [y] * len(values), s=16, facecolors="none", edgecolors=INK_2, linewidths=1,
                           zorder=3)
                ax.scatter([values.mean()], [y], s=46, color=color, zorder=4)
            ax.set_xlabel(f"held-out targets certified per rule list ({self._of(bench)})", color=INK_2, fontsize=8)
            ax.set_ylim(len(cells) - 0.5, -0.5)
        axes[0][0].set_yticks(range(len(cells)))
        axes[0][0].set_yticklabels([c.name for c in cells], fontsize=8.5, color=INK)
        fig.suptitle(self.title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
        fig.tight_layout()
        path = self.out / f"{self.name}.png"
        self.out.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, facecolor=SURFACE)
        plt.close(fig)
        return Path(path)
