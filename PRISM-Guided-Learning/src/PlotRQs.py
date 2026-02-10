"""
PlotRQs.py - Generate plots for Research Questions 1-3.

Usage:
    python src/PlotRQs.py [results_dir]

    If results_dir not provided, uses the most recent run in out/results/.

RQ1: Effect of Feedback
    Compares four feedback approaches (Vanilla, Vanilla+, Feedback-, Feedback)
    across available LLMs. Plots: TTS bar chart, success boxplot, summary table.

RQ2: Comparative Analysis
    Compares LLM+feedback methods against RL.
    Plots: TTS bar chart, success boxplot, summary table, scaling line plot.

RQ3: LLM Scalability
    Isolates LLM methods on a line plot of avg TTS vs grid size.
"""

import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

FEEDBACK_ORDER = ["Vanilla", "Vanilla+", "Feedback-", "Feedback"]

_FEEDBACK_PARSE_MAP = {
    "VANILLA_PLUS": "Vanilla+",
    "FEEDBACK_MINUS": "Feedback-",
    "VANILLA": "Vanilla",
    "FEEDBACK": "Feedback",
}

_LLM_DISPLAY = {
    "GPT5_NANO": "GPT-5 Nano",
    "GPT5_MINI": "GPT-5 Mini",
    "GEMINI_PRO": "Gemini 2.5 Pro",
}

FEEDBACK_COLORS = {
    "Vanilla": "#5DADE2",
    "Vanilla+": "#2E86C1",
    "Feedback-": "#F39C12",
    "Feedback": "#E74C3C",
}

RL_COLOR = "#27AE60"

LLM_HATCHES = ["", "///", "..."]


# ──────────────────────────────────────────────
# Data loading and aggregation
# ──────────────────────────────────────────────

def parse_model_name(name):
    """Parse a model name into (feedback_type, llm_display_name).

    Returns ('RL', None) for RL, (type_str, llm_str) for LLM models.
    """
    if name == "RL":
        return "RL", None
    if name in ("RANDOM", "UNIFORM"):
        return name, None
    if not name.startswith("LLM_"):
        return name, None

    rest = name[4:]
    # Longest prefix first to avoid partial matches
    for key in ("VANILLA_PLUS", "FEEDBACK_MINUS", "VANILLA", "FEEDBACK"):
        if rest.startswith(key + "_"):
            llm_key = rest[len(key) + 1:]
            return _FEEDBACK_PARSE_MAP[key], _LLM_DISPLAY.get(llm_key, llm_key)
    return name, None


def load_results(run_dir):
    """Load all *_results.parquet files from *run_dir*.

    Returns ``{model_name: DataFrame}`` (MultiIndex DataFrames).
    """
    results = {}
    for f in sorted(Path(run_dir).glob("*_results.parquet")):
        model_name = f.stem.replace("_results", "")
        results[model_name] = pd.read_parquet(f)
    return results


def summarize_samples(df):
    """Collapse a multi-index (sample_id, iteration) DataFrame into one row
    per sample with aggregated metrics.

    Returned columns:
        total_time, success, total_mistakes, final_cost,
        num_iterations, size, complexity, final_ltl_score
    """
    records = []
    for sid in df.index.get_level_values("sample_id").unique():
        sample = df.loc[sid]
        if isinstance(sample, pd.Series):
            records.append({
                "sample_id": sid,
                "total_time": sample.get("iteration_time", 0),
                "success": bool(sample.get("success", False)),
                "total_mistakes": int(sample.get("mistakes", 0)),
                "final_cost": float(sample.get("cost", 0)),
                "num_iterations": 1,
                "size": int(sample.get("size", 0)),
                "complexity": int(sample.get("complexity", 0)),
                "final_ltl_score": float(sample.get("final_ltl_score", 0)),
            })
        else:
            final = sample[sample["is_final"]].iloc[0] if "is_final" in sample.columns and sample["is_final"].any() else sample.iloc[-1]
            records.append({
                "sample_id": sid,
                "total_time": float(sample["iteration_time"].sum()),
                "success": bool(final["success"]),
                "total_mistakes": int(sample["mistakes"].sum()),
                "final_cost": float(final["cost"]),
                "num_iterations": len(sample),
                "size": int(sample["size"].iloc[0]),
                "complexity": int(sample["complexity"].iloc[0]),
                "final_ltl_score": float(final["final_ltl_score"]),
            })
    return pd.DataFrame(records).set_index("sample_id")


def _latest_run_dir(base):
    base = Path(base)
    dirs = [d for d in base.iterdir() if d.is_dir() and d.name != "prev"]
    if not dirs:
        raise FileNotFoundError(f"No result directories in {base}")
    return str(max(dirs, key=lambda d: d.stat().st_mtime))


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# Helpers for method grouping
# ──────────────────────────────────────────────

def _split_llm_methods(summaries):
    """Return {(feedback_type, llm): summary_df} for LLM models only."""
    out = {}
    for model_name, df in summaries.items():
        fb, llm = parse_model_name(model_name)
        if llm is not None:
            out[(fb, llm)] = df
    return out


def _present_feedback_types(llm_methods):
    return [ft for ft in FEEDBACK_ORDER if any(k[0] == ft for k in llm_methods)]


def _present_llms(llm_methods):
    return sorted(set(k[1] for k in llm_methods))


def _method_label(fb, llm, multi_llm):
    return f"{fb} ({llm})" if multi_llm else fb


# ──────────────────────────────────────────────
# RQ1: Effect of Feedback
# ──────────────────────────────────────────────

def rq1(summaries, out_dir, raw=None):
    print("\n=== RQ1: Effect of Feedback ===")

    llm_methods = _split_llm_methods(summaries)
    if not llm_methods:
        print("  No LLM methods found – skipping RQ1.")
        return

    fb_types = _present_feedback_types(llm_methods)
    llms = _present_llms(llm_methods)
    multi_llm = len(llms) > 1

    _rq1_tts_bar(llm_methods, fb_types, llms, multi_llm, out_dir)
    _rq1_success_grouped_bar(llm_methods, fb_types, llms, multi_llm, out_dir)
    _rq1_success_heatmap(llm_methods, fb_types, llms, multi_llm, out_dir)
    _rq1_table(llm_methods, fb_types, llms, multi_llm, out_dir)
    if raw is not None:
        _rq1_mistakes_per_iteration(raw, out_dir)


def _rq1_tts_bar(M, fb_types, llms, multi_llm, out_dir):
    """Grouped bar chart – median TTS by feedback type, sub-grouped by LLM."""
    n_fb = len(fb_types)
    n_llm = len(llms)
    bar_w = 0.7 / max(n_llm, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_fb * 2.5), 5))
    x = np.arange(n_fb)

    for li, llm in enumerate(llms):
        vals = []
        for fb in fb_types:
            key = (fb, llm)
            vals.append(M[key]["total_time"].median() if key in M else 0)
        offset = (li - (n_llm - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset, vals, bar_w * 0.9,
            label=llm if multi_llm else None,
            color=[FEEDBACK_COLORS.get(fb, "#999") for fb in fb_types],
            hatch=LLM_HATCHES[li % len(LLM_HATCHES)],
            edgecolor="black", linewidth=0.5,
        )
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.02,
                        f"{v:.0f}s", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Feedback Approach")
    ax.set_ylabel("Median Time to Solution (s)")
    ax.set_title("RQ1: Median TTS by Feedback Approach")
    ax.set_xticks(x)
    ax.set_xticklabels(fb_types)
    if multi_llm:
        ax.legend(title="LLM")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, os.path.join(out_dir, "rq1_tts_bar.png"))


def _rq1_success_grouped_bar(M, fb_types, llms, multi_llm, out_dir):
    """Grouped bar chart: success rate by grid size, bars grouped by method."""
    # Collect all grid sizes present
    all_sizes = sorted(set(
        s for key in M for s in M[key]["size"].unique()
    ))

    # Build method list
    method_keys = []
    for fb in fb_types:
        for llm in llms:
            if (fb, llm) in M:
                method_keys.append((fb, llm))

    n_methods = len(method_keys)
    n_sizes = len(all_sizes)
    bar_w = 0.7 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_sizes * n_methods * 0.5), 5))
    x = np.arange(n_sizes)

    for mi, (fb, llm) in enumerate(method_keys):
        df = M[(fb, llm)]
        rates_by_size = df.groupby("size")["success"].mean()
        vals = [rates_by_size.get(s, 0) for s in all_sizes]
        offset = (mi - (n_methods - 1) / 2) * bar_w
        label = _method_label(fb, llm, multi_llm)
        ax.bar(x + offset, vals, bar_w * 0.9,
               label=label, color=FEEDBACK_COLORS.get(fb, "#999"),
               hatch=LLM_HATCHES[mi % len(LLM_HATCHES)],
               edgecolor="black", linewidth=0.5, alpha=0.8)

    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Success Rate")
    ax.set_title("RQ1: Success Rate by Grid Size")
    ax.set_xticks(x)
    ax.set_xticklabels(all_sizes)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, os.path.join(out_dir, "rq1_success_grouped_bar.png"))


def _rq1_success_heatmap(M, fb_types, llms, multi_llm, out_dir):
    """Heatmap: methods (rows) x grid sizes (columns), cell = success rate."""
    all_sizes = sorted(set(
        s for key in M for s in M[key]["size"].unique()
    ))

    method_labels = []
    heat_data = []
    for fb in fb_types:
        for llm in llms:
            key = (fb, llm)
            if key not in M:
                continue
            rates_by_size = M[key].groupby("size")["success"].mean()
            row = [rates_by_size.get(s, float("nan")) for s in all_sizes]
            heat_data.append(row)
            method_labels.append(_method_label(fb, llm, multi_llm))

    if not heat_data:
        return

    data = np.array(heat_data)
    fig, ax = plt.subplots(figsize=(max(6, len(all_sizes) * 1.2), max(3, len(method_labels) * 0.6 + 1)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    # Annotate cells
    for i in range(len(method_labels)):
        for j in range(len(all_sizes)):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.4 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)

    ax.set_xticks(range(len(all_sizes)))
    ax.set_xticklabels(all_sizes)
    ax.set_yticks(range(len(method_labels)))
    ax.set_yticklabels(method_labels)
    ax.set_xlabel("Grid Size")
    ax.set_title("RQ1: Success Rate Heatmap")
    fig.colorbar(im, ax=ax, label="Success Rate", shrink=0.8)
    _save(fig, os.path.join(out_dir, "rq1_success_heatmap.png"))


def _rq1_table(M, fb_types, llms, multi_llm, out_dir):
    """Summary table: successes, mistakes, cost, iterations, TTS."""
    rows = []
    for fb in fb_types:
        for llm in llms:
            key = (fb, llm)
            if key not in M:
                continue
            df = M[key]
            n = len(df)
            rows.append({
                "Method": _method_label(fb, llm, multi_llm),
                "Successes": f"{int(df['success'].sum())}/{n}",
                "Success %": f"{df['success'].mean():.0%}",
                "Avg Mistakes/Iter": f"{df['total_mistakes'].sum() / df['num_iterations'].sum():.1f}",
                "Avg Cost": f"{df['final_cost'].mean():.3f}",
                "Med. Iters": f"{df['num_iterations'].median():.1f}",
                "Med. TTS (s)": f"{df['total_time'].median():.1f}",
            })

    if not rows:
        return
    tdf = pd.DataFrame(rows)

    # Console output
    print("\n  RQ1 Summary Table:")
    print(tdf.to_string(index=False))

    # Figure
    fig, ax = plt.subplots(figsize=(12, max(1.8, len(rows) * 0.45 + 1.2)))
    ax.axis("off")
    ax.set_title("RQ1: Summary Metrics", fontsize=13, pad=20)
    tbl = ax.table(cellText=tdf.values, colLabels=tdf.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(tdf.columns))))
    for j in range(len(tdf.columns)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    _save(fig, os.path.join(out_dir, "rq1_table.png"))


def _rq1_mistakes_per_iteration(raw, out_dir):
    """Line plot: average mistakes per iteration for LLM feedback methods.

    Uses the raw multi-index DataFrames so we can group by iteration number.
    Each line shows how the average number of mistakes evolves across
    feedback iterations, with sample counts annotated.
    """
    multi_llm = len(set(
        parse_model_name(m)[1] for m in raw if parse_model_name(m)[1] is not None
    )) > 1

    items = []  # (label, fb_type, df)
    for model_name, df in raw.items():
        fb, llm = parse_model_name(model_name)
        if llm is None:
            continue
        label = _method_label(fb, llm, multi_llm)
        items.append((label, fb, df))

    if not items:
        return

    # Sort by feedback order
    fb_rank = {fb: i for i, fb in enumerate(FEEDBACK_ORDER)}
    items.sort(key=lambda t: (fb_rank.get(t[1], 99), t[0]))

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for i, (label, fb, df) in enumerate(items):
        # Only include samples that ran all 3 iterations (max iteration = 3)
        # so the cohort is consistent across all points
        max_iter = df.index.get_level_values("iteration").max()
        full_run_samples = df.groupby("sample_id").filter(
            lambda g: g.index.get_level_values("iteration").max() == max_iter
        )
        if full_run_samples.empty:
            continue
        by_iter = full_run_samples.groupby("iteration")["mistakes"].agg(["mean", "count"])
        ax.plot(
            by_iter.index, by_iter["mean"],
            marker=markers[i % len(markers)],
            color=FEEDBACK_COLORS.get(fb, "#999"),
            label=label, linewidth=2, markersize=7,
        )
        # Annotate sample count at each point
        for it, row in by_iter.iterrows():
            ax.annotate(
                f"n={int(row['count'])}",
                (it, row["mean"]),
                textcoords="offset points", xytext=(0, 10),
                fontsize=7, ha="center", color="#555",
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Mistakes")
    ax.set_title("RQ1: Average Mistakes per Iteration")
    ax.set_xticks(sorted(df.index.get_level_values("iteration").unique()))
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(out_dir, "rq1_mistakes_per_iter.png"))


# ──────────────────────────────────────────────
# RQ2: Comparative Analysis
# ──────────────────────────────────────────────

def _build_rq2_methods(summaries):
    """Return ordered list of (display_name, summary_df, feedback_type)."""
    multi_llm = len(set(
        parse_model_name(m)[1] for m in summaries if parse_model_name(m)[1] is not None
    )) > 1

    methods = {}  # display_name -> (df, fb_type)
    for model_name, df in summaries.items():
        fb, llm = parse_model_name(model_name)
        if fb == "RL":
            methods["RL"] = (df, "RL")
        elif llm is not None:
            label = _method_label(fb, llm, multi_llm)
            methods[label] = (df, fb)

    # Ordering: RL first, then feedback order
    ordered = []
    if "RL" in methods:
        ordered.append("RL")
    for fb in FEEDBACK_ORDER:
        for name in sorted(methods):
            if name == "RL":
                continue
            if methods[name][1] == fb and name not in ordered:
                ordered.append(name)
    for name in methods:
        if name not in ordered:
            ordered.append(name)
    return ordered, methods


def rq2(summaries, out_dir):
    print("\n=== RQ2: Comparative Analysis ===")

    ordered, methods = _build_rq2_methods(summaries)
    if not methods:
        print("  No methods found – skipping RQ2.")
        return

    _rq2_tts_bar(ordered, methods, out_dir)
    _rq2_success_grouped_bar(ordered, methods, out_dir)
    _rq2_success_heatmap(ordered, methods, out_dir)
    _rq2_table(ordered, methods, out_dir)
    _rq2_scaling(ordered, methods, out_dir)


def _method_color(fb_type):
    if fb_type == "RL":
        return RL_COLOR
    return FEEDBACK_COLORS.get(fb_type, "#999")


def _rq2_tts_bar(ordered, M, out_dir):
    fig, ax = plt.subplots(figsize=(max(8, len(ordered) * 1.5), 5))
    x = np.arange(len(ordered))
    vals = [M[n][0]["total_time"].median() for n in ordered]
    colors = [_method_color(M[n][1]) for n in ordered]

    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.02,
                f"{v:.0f}s", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Median Time to Solution (s)")
    ax.set_title("RQ2: Median TTS – All Methods")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, os.path.join(out_dir, "rq2_tts_bar.png"))


def _rq2_success_grouped_bar(ordered, M, out_dir):
    """Grouped bar chart: success rate by grid size, bars grouped by method."""
    all_sizes = sorted(set(
        s for n in ordered for s in M[n][0]["size"].unique()
    ))

    n_methods = len(ordered)
    n_sizes = len(all_sizes)
    bar_w = 0.7 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_sizes * n_methods * 0.5), 5))
    x = np.arange(n_sizes)

    for mi, n in enumerate(ordered):
        df, fb = M[n]
        rates_by_size = df.groupby("size")["success"].mean()
        vals = [rates_by_size.get(s, 0) for s in all_sizes]
        offset = (mi - (n_methods - 1) / 2) * bar_w
        ax.bar(x + offset, vals, bar_w * 0.9,
               label=n, color=_method_color(fb),
               edgecolor="black", linewidth=0.5, alpha=0.8)

    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Success Rate")
    ax.set_title("RQ2: Success Rate by Grid Size")
    ax.set_xticks(x)
    ax.set_xticklabels(all_sizes)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, os.path.join(out_dir, "rq2_success_grouped_bar.png"))


def _rq2_success_heatmap(ordered, M, out_dir):
    """Heatmap: methods (rows) x grid sizes (columns), cell = success rate."""
    all_sizes = sorted(set(
        s for n in ordered for s in M[n][0]["size"].unique()
    ))

    heat_data = []
    for n in ordered:
        df = M[n][0]
        rates_by_size = df.groupby("size")["success"].mean()
        heat_data.append([rates_by_size.get(s, float("nan")) for s in all_sizes])

    data = np.array(heat_data)
    fig, ax = plt.subplots(figsize=(max(6, len(all_sizes) * 1.2), max(3, len(ordered) * 0.6 + 1)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    for i in range(len(ordered)):
        for j in range(len(all_sizes)):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.4 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_color)

    ax.set_xticks(range(len(all_sizes)))
    ax.set_xticklabels(all_sizes)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered)
    ax.set_xlabel("Grid Size")
    ax.set_title("RQ2: Success Rate Heatmap")
    fig.colorbar(im, ax=ax, label="Success Rate", shrink=0.8)
    _save(fig, os.path.join(out_dir, "rq2_success_heatmap.png"))


def _rq2_table(ordered, M, out_dir):
    rows = []
    for n in ordered:
        df = M[n][0]
        total = len(df)
        rows.append({
            "Method": n,
            "Successes": f"{int(df['success'].sum())}/{total}",
            "Success %": f"{df['success'].mean():.0%}",
            "Avg Mistakes/Iter": f"{df['total_mistakes'].sum() / df['num_iterations'].sum():.1f}",
            "Avg Cost": f"{df['final_cost'].mean():.3f}",
            "Med. TTS (s)": f"{df['total_time'].median():.1f}",
        })
    tdf = pd.DataFrame(rows)

    print("\n  RQ2 Summary Table:")
    print(tdf.to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, max(1.8, len(rows) * 0.45 + 1.2)))
    ax.axis("off")
    ax.set_title("RQ2: Comparative Summary", fontsize=13, pad=20)
    tbl = ax.table(cellText=tdf.values, colLabels=tdf.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(tdf.columns))))
    for j in range(len(tdf.columns)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    _save(fig, os.path.join(out_dir, "rq2_table.png"))


def _rq2_scaling(ordered, M, out_dir):
    """Overlaid line plot: average TTS vs grid size for every method."""
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]

    for i, n in enumerate(ordered):
        df = M[n][0]
        by_size = df.groupby("size")["total_time"].mean().sort_index()
        ax.plot(by_size.index, by_size.values,
                marker=markers[i % len(markers)],
                color=_method_color(M[n][1]),
                label=n, linewidth=2, markersize=6)

    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Average Time to Solution (s)")
    ax.set_title("RQ2: Scaling – Avg TTS vs Grid Size")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(out_dir, "rq2_scaling.png"))


# ──────────────────────────────────────────────
# RQ3: LLM Scalability
# ──────────────────────────────────────────────

def rq3(summaries, out_dir):
    """Line plot of average TTS vs grid size for LLM methods only."""
    print("\n=== RQ3: LLM Scalability ===")

    multi_llm = len(set(
        parse_model_name(m)[1] for m in summaries if parse_model_name(m)[1] is not None
    )) > 1

    llm_items = []  # (display, df, fb_type)
    for model_name, df in summaries.items():
        fb, llm = parse_model_name(model_name)
        if llm is not None:
            label = _method_label(fb, llm, multi_llm)
            llm_items.append((label, df, fb))

    if not llm_items:
        print("  No LLM methods found – skipping RQ3.")
        return

    # Sort by feedback order
    fb_rank = {fb: i for i, fb in enumerate(FEEDBACK_ORDER)}
    llm_items.sort(key=lambda t: (fb_rank.get(t[2], 99), t[0]))

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]

    for i, (label, df, fb) in enumerate(llm_items):
        by_size = df.groupby("size")["total_time"].mean().sort_index()
        ax.plot(by_size.index, by_size.values,
                marker=markers[i % len(markers)],
                color=FEEDBACK_COLORS.get(fb, "#999"),
                label=label, linewidth=2, markersize=6)

    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Average Time to Solution (s)")
    ax.set_title("RQ3: LLM Scalability – Avg TTS vs Grid Size")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(out_dir, "rq3_scaling.png"))


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

## ── Change this to plot a specific run, or leave None for latest ──
RUN_FOLDER = "20_20260210_11-48-01"


def main():
    base_path = os.path.join("PRISM-Guided-Learning", "out", "results")

    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    elif RUN_FOLDER is not None:
        run_dir = os.path.join(base_path, RUN_FOLDER)
    else:
        run_dir = _latest_run_dir(base_path)

    print(f"Loading results from: {run_dir}")

    raw = load_results(run_dir)
    if not raw:
        print("No parquet files found!")
        return

    print(f"Found {len(raw)} model(s): {', '.join(raw.keys())}")

    summaries = {name: summarize_samples(df) for name, df in raw.items()}

    out_dir = os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    rq1(summaries, out_dir, raw=raw)
    rq2(summaries, out_dir)
    rq3(summaries, out_dir)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
