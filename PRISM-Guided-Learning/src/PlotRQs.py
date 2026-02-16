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

FEEDBACK_ORDER = [r"None", r"IT", r"ITF", r"CITF", r"ITFN"]

# Feedback types to show on the GSR-by-LLM scaling plot (alongside RL)
GSR_LLM_SCALING_FEEDBACK = [r"ITF", r"CITF", r"ITFN"]

# Feedback type to use for the GSR-by-feedback scaling plot
GSR_FEEDBACK_SCALING_METHOD = r"ITF"

# Metric naming: internal keys and display labels (LaTeX / matplotlib)
RATE_KEYS   = ["SR_all", "SR_C1", "SR_C2", "SR_C3"]
RATE_LABELS = [r"\srAll", r"\srCon{C1}", r"\srCon{C2}", r"\srCon{C3}"]
PROX_KEYS   = ["d_all", "d_C1", "d_C2", "d_C3"]
PROX_LABELS = [r"\prox{all}", r"\prox{C1}", r"\prox{C2}", r"\prox{C3}"]

_FEEDBACK_PARSE_MAP = {
    "VANILLA_PLUS": r"IT",
    "FEEDBACK_SIMPLIFIED": r"CITF",
    "FEEDBACK_MINUS": r"ITF",
    "VANILLA": r"None",
    "FEEDBACK": r"ITFN",
}

_LLM_DISPLAY = {
    "GPT5_NANO": "GPT-5 Nano",
    "GPT5_MINI": "GPT-5 Mini",
    "GEMINI_PRO": "Gemini 2.5 Pro",
}

LLM_ORDER = ["GPT-5 Mini", "Gemini 2.5 Pro", "GPT-5 Nano"]

FEEDBACK_COLORS = {
    "Vanilla": "#5DADE2",
    "Vanilla+": "#2E86C1",
    "Feedback-": "#F39C12",
    "FeedbackS": "#8E44AD",
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
    for key in ("VANILLA_PLUS", "FEEDBACK_SIMPLIFIED", "FEEDBACK_MINUS", "VANILLA", "FEEDBACK"):
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
        df = pd.read_parquet(f)
        print(f"  LOAD {model_name}: shape={df.shape}, index={type(df.index).__name__}, "
              f"index.names={df.index.names}, cols={df.columns.tolist()[:5]}...")
        # Ensure MultiIndex (sample_id, iteration) even if saved as columns
        if not isinstance(df.index, pd.MultiIndex):
            if "sample_id" in df.columns and "iteration" in df.columns:
                df = df.set_index(["sample_id", "iteration"])
        results[model_name] = df
    return results


def summarize_samples(df):
    """Collapse a multi-index (sample_id, iteration) DataFrame into one row
    per sample with aggregated metrics.

    Returned columns:
        total_time, success, total_mistakes, final_mistakes, final_cost,
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
                "final_mistakes": int(sample.get("mistakes", 0)),
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
                "final_mistakes": int(final["mistakes"]),
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


def _save_latex_table(df, path):
    """Save a DataFrame as a booktabs-compatible LaTeX table to a .txt file."""
    def _esc(s):
        return str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")

    col_fmt = "l" + "r" * (len(df.columns) - 1)
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(_esc(c) for c in df.columns) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(_esc(v) for v in row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {path}")


def _compute_group_rates(raw_df):
    """Compute SR_all and per-group success rates from raw multi-index DataFrame.

    Returns dict with keys: SR_all, SR_C1, SR_C2, SR_C3 (all as percentages 0-100).
    """
    from config.Settings import (
        GOAL_REACHABILITY_THRESHOLD,
        SEQUENCE_ORDERING_THRESHOLD,
        OBSTACLE_AVOIDANCE_THRESHOLD,
    )

    # Get final iteration rows
    if "is_final" in raw_df.columns:
        finals = raw_df[raw_df["is_final"]]
    else:
        finals = raw_df.groupby("sample_id").tail(1)

    # SR_all
    sr_all = finals["success"].astype(bool).mean() * 100

    # Identify prob columns by category
    prob_cols = [c for c in raw_df.columns if c.startswith("prob_")]
    goal_cols = sorted([c for c in prob_cols if c.startswith("prob_goal")])
    seq_cols = sorted([c for c in prob_cols
                       if c.startswith("prob_seq_") or c == "prob_complete_sequence"])
    obs_cols = sorted([c for c in prob_cols if c.startswith("prob_avoid")])

    def group_rate(cols, threshold):
        if not cols:
            return float("nan")
        vals = finals[cols]
        total = vals.size  # total individual requirements across all samples
        successes = int((vals >= threshold).sum().sum())
        return (successes / total) * 100 if total > 0 else float("nan")

    return {
        "SR_all": sr_all,
        "SR_C1": group_rate(goal_cols, GOAL_REACHABILITY_THRESHOLD),
        "SR_C2": group_rate(seq_cols, SEQUENCE_ORDERING_THRESHOLD),
        "SR_C3": group_rate(obs_cols, OBSTACLE_AVOIDANCE_THRESHOLD),
    }


def _build_raw_lookup(raw):
    """Build {(feedback_type, llm): raw_df} and {'RL': raw_df} from raw dict."""
    lookup = {}
    for model_name, df in raw.items():
        fb, llm = parse_model_name(model_name)
        if fb == "RL":
            lookup[("RL", None)] = df
        elif llm is not None:
            lookup[(fb, llm)] = df
    return lookup


def _compute_proximity(raw_df):
    """Compute proximity-to-success metrics for FAILED samples only.

    Proximity for a failed requirement = threshold - achieved_probability.
    Only requirements that failed (prob < threshold) are included.

    Returns dict with keys:
        d_all, d_C1, d_C2, d_C3  – median (across failed samples) of avg proximity
    All values are floats or NaN if no failed samples / no failed requirements.
    """
    from config.Settings import (
        GOAL_REACHABILITY_THRESHOLD,
        SEQUENCE_ORDERING_THRESHOLD,
        OBSTACLE_AVOIDANCE_THRESHOLD,
    )

    # Get final iteration rows
    if "is_final" in raw_df.columns:
        finals = raw_df[raw_df["is_final"]]
    else:
        finals = raw_df.groupby("sample_id").tail(1)

    # Filter to failed samples
    failed = finals[~finals["success"].astype(bool)]
    if failed.empty:
        return {k: float("nan") for k in PROX_KEYS}

    # Identify prob columns by category
    prob_cols = [c for c in raw_df.columns if c.startswith("prob_")]
    goal_cols = sorted([c for c in prob_cols if c.startswith("prob_goal")])
    seq_cols = sorted([c for c in prob_cols
                       if c.startswith("prob_seq_") or c == "prob_complete_sequence"])
    obs_cols = sorted([c for c in prob_cols if c.startswith("prob_avoid")])

    # Build (col_name -> threshold) mapping
    col_thresh = {}
    for c in goal_cols:
        col_thresh[c] = GOAL_REACHABILITY_THRESHOLD
    for c in seq_cols:
        col_thresh[c] = SEQUENCE_ORDERING_THRESHOLD
    for c in obs_cols:
        col_thresh[c] = OBSTACLE_AVOIDANCE_THRESHOLD

    group_map = {
        "d_C1": goal_cols,
        "d_C2": seq_cols,
        "d_C3": obs_cols,
    }

    def _sample_proximity(row, cols):
        """Return (avg, max) proximity for failed requirements in *cols*."""
        gaps = []
        for c in cols:
            prob = row[c]
            thresh = col_thresh[c]
            if prob < thresh:
                gaps.append(thresh - prob)
        if not gaps:
            return float("nan"), float("nan")
        return np.mean(gaps), max(gaps)

    # Per-sample metrics
    all_cols = goal_cols + seq_cols + obs_cols
    per_sample_all_avg = []
    per_sample_all_max = []
    per_sample_group = {g: [] for g in group_map}  # list of avg per sample

    for _, row in failed.iterrows():
        avg_all, max_all = _sample_proximity(row, all_cols)
        per_sample_all_avg.append(avg_all)
        per_sample_all_max.append(max_all)

        for g, cols in group_map.items():
            avg_g, _ = _sample_proximity(row, cols)
            per_sample_group[g].append(avg_g)

    def _median_dropna(vals):
        clean = [v for v in vals if not np.isnan(v)]
        return float(np.median(clean)) if clean else float("nan")

    result = {
        "d_all": _median_dropna(per_sample_all_avg),
    }
    for g in group_map:
        result[g] = _median_dropna(per_sample_group[g])

    return result


def _save_multicolumn_latex_table(rows, row_labels, col_groups, sub_cols,
                                  path, caption="", label="",
                                  lower_is_better=False):
    """Save a LaTeX table with multicolumn headers like the reference image.

    Bold = best in each column, underline = 2nd best.

    Parameters
    ----------
    rows : list of list of str
        Each inner list has one value per (col_group × sub_col).
    row_labels : list of str
        Row labels (one per row).
    col_groups : list of str
        Top-level column group names (e.g. LLM names).
    sub_cols : list of str
        Sub-column names repeated under each group (e.g. RATE_LABELS or PROX_LABELS).
    lower_is_better : bool
        If True, the lowest value is bolded (best) instead of highest.
    """
    n_sub = len(sub_cols)
    n_groups = len(col_groups)
    total_data_cols = n_groups * n_sub
    n_rows = len(rows)

    # ── Determine best / 2nd-best per column ──
    # Parse numeric values (None for non-numeric like "--")
    numeric = []
    for row_data in rows:
        parsed = []
        for v in row_data:
            try:
                parsed.append(float(v))
            except (ValueError, TypeError):
                parsed.append(None)
        numeric.append(parsed)

    # For each column, find the best and 2nd-best VALUES, then mark all
    # rows that match.  This handles ties (e.g. RL repeating the same value).
    best_val = [None] * total_data_cols
    second_val = [None] * total_data_cols
    for ci in range(total_data_cols):
        col_vals = [numeric[ri][ci] for ri in range(n_rows)
                    if numeric[ri][ci] is not None]
        if not col_vals:
            continue
        unique_sorted = sorted(set(col_vals), reverse=(not lower_is_better))
        best_val[ci] = unique_sorted[0]
        if len(unique_sorted) >= 2:
            second_val[ci] = unique_sorted[1]

    # Format cells with \textbf / \underline
    formatted = []
    for ri, row_data in enumerate(rows):
        fmt_row = []
        for ci, v in enumerate(row_data):
            nv = numeric[ri][ci]
            if nv is not None and nv == best_val[ci]:
                fmt_row.append(f"\\textbf{{{v}}}")
            else:
                fmt_row.append(v)
        formatted.append(fmt_row)

    # ── Build LaTeX ──
    col_parts = ["l"]
    for g in range(n_groups):
        col_parts.append("|")
        col_parts.extend(["r"] * n_sub)
    col_fmt = "".join(col_parts)

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_fmt}}}")
    lines.append("\\toprule")

    # Top header row: empty cell, then multicolumn for each group
    header_parts = [""]
    for gi, g in enumerate(col_groups):
        sep = "|c|" if gi < n_groups - 1 else "|c"
        header_parts.append(f"\\multicolumn{{{n_sub}}}{{{sep}}}{{{g}}}")
    lines.append(" & ".join(header_parts) + " \\\\")

    # Cline under group headers
    start = 2  # data cols start at column 2
    for gi in range(n_groups):
        end = start + n_sub - 1
        lines.append(f"\\cline{{{start}-{end}}}")
        start = end + 1

    # Sub-header row
    sub_header_parts = ["Methods"]
    for _ in col_groups:
        sub_header_parts.extend(sub_cols)
    lines.append(" & ".join(sub_header_parts) + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for rl, fmt_row in zip(row_labels, formatted):
        parts = [rl] + fmt_row
        lines.append(" & ".join(parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
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
    present = set(k[1] for k in llm_methods)
    ordered = [llm for llm in LLM_ORDER if llm in present]
    # Append any LLMs not in LLM_ORDER (sorted, as fallback)
    ordered += sorted(present - set(LLM_ORDER))
    return ordered


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
    _rq1_table(llm_methods, fb_types, llms, multi_llm, out_dir, raw=raw)
    if raw is not None:
        _rq1_mistakes_per_iteration(raw, out_dir)
        _rq1_failure_reasons(raw, out_dir)
        _rq1_proximity_table(fb_types, llms, out_dir, raw)


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


def _rq1_table(M, fb_types, llms, multi_llm, out_dir, raw=None):
    """Summary table with multicolumn layout: SR metrics per LLM."""

    # Build raw lookup for per-group rates
    raw_lookup = _build_raw_lookup(raw) if raw is not None else {}

    # Build data rows
    row_labels = []
    rows = []
    for fb in fb_types:
        row_data = []
        for llm in llms:
            key = (fb, llm)
            if key not in M:
                row_data.extend(["--"] * len(RATE_KEYS))
                continue
            df = M[key]
            sr_all = df["success"].mean() * 100

            # Per-group rates from raw data
            if key in raw_lookup:
                rates = _compute_group_rates(raw_lookup[key])
            else:
                rates = {"SR_all": sr_all, "SR_C1": float("nan"), "SR_C2": float("nan"), "SR_C3": float("nan")}

            for sc in RATE_KEYS:
                v = rates[sc]
                row_data.append(f"{v:.2f}" if not np.isnan(v) else "--")
        row_labels.append(fb)
        rows.append(row_data)

    if not rows:
        return

    # Console output
    print("\n  RQ1 Summary Table:")
    header = f"{'Method':<14}" + "".join(f"  {llm:>36}" for llm in llms)
    print(header)
    for rl, rd in zip(row_labels, rows):
        print(f"{rl:<14}" + "  ".join(f"{v:>8}" for v in rd))

    # LaTeX table
    _save_multicolumn_latex_table(
        rows, row_labels, llms, RATE_LABELS,
        os.path.join(out_dir, "rq1_table.txt"),
        caption="RQ1: Effect of Feedback",
        label="tab:rq1",
    )

    # Also save the old-style simple table as a figure
    flat_cols = ["Method"]
    for llm in llms:
        for sc in RATE_LABELS:
            flat_cols.append(f"{sc}" if not multi_llm else f"{llm} {sc}")
    flat_rows = []
    for rl, rd in zip(row_labels, rows):
        flat_rows.append([rl] + rd)
    tdf = pd.DataFrame(flat_rows, columns=flat_cols)

    fig, ax = plt.subplots(figsize=(max(12, len(flat_cols) * 1.2), max(1.8, len(rows) * 0.45 + 1.2)))
    ax.axis("off")
    ax.set_title("RQ1: Summary Metrics", fontsize=13, pad=20)
    tbl = ax.table(cellText=tdf.values, colLabels=tdf.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
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
        if llm is None or fb == "Vanilla":
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


def _rq1_failure_reasons(raw, out_dir):
    """Table counting failure reasons (goal reachability / sequential ordering /
    obstacle avoidance) for failed final solutions, broken down by approach.

    Uses the raw multi-index DataFrames to inspect per-probability columns.
    """
    from config.Settings import (
        GOAL_REACHABILITY_THRESHOLD,
        SEQUENCE_ORDERING_THRESHOLD,
        OBSTACLE_AVOIDANCE_THRESHOLD,
    )

    multi_llm = len(set(
        parse_model_name(m)[1] for m in raw if parse_model_name(m)[1] is not None
    )) > 1

    rows = []
    for model_name, df in raw.items():
        fb, llm = parse_model_name(model_name)
        if llm is None:
            continue
        label = _method_label(fb, llm, multi_llm)

        # Get final iteration rows only
        if "is_final" in df.columns:
            finals = df[df["is_final"]]
        else:
            finals = df.groupby("sample_id").tail(1)

        # Filter to failed samples
        failed = finals[~finals["success"].astype(bool)]
        n_failed = len(failed)
        if n_failed == 0:
            rows.append({
                "Method": label,
                "Failed": 0,
                "Goal Reach.": "0",
                "Seq. Order.": "0",
                "Obst. Avoid.": "0",
            })
            continue

        # Identify prob columns by category
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        goal_cols = [c for c in prob_cols if c.startswith("prob_goal")]
        seq_cols = [c for c in prob_cols
                    if c.startswith("prob_seq_") or c == "prob_complete_sequence"]
        obs_cols = [c for c in prob_cols if c.startswith("prob_avoid")]

        # Count samples where at least one prob in category is below threshold
        goal_fail = (failed[goal_cols] < GOAL_REACHABILITY_THRESHOLD).any(axis=1).sum() if goal_cols else 0
        seq_fail = (failed[seq_cols] < SEQUENCE_ORDERING_THRESHOLD).any(axis=1).sum() if seq_cols else 0
        obs_fail = (failed[obs_cols] < OBSTACLE_AVOIDANCE_THRESHOLD).any(axis=1).sum() if obs_cols else 0

        rows.append({
            "Method": label,
            "Failed": n_failed,
            "Goal Reach.": f"{int(goal_fail)} ({goal_fail/n_failed:.0%})",
            "Seq. Order.": f"{int(seq_fail)} ({seq_fail/n_failed:.0%})",
            "Obst. Avoid.": f"{int(obs_fail)} ({obs_fail/n_failed:.0%})",
        })

    if not rows:
        return

    # Sort by feedback order
    fb_rank = {fb: i for i, fb in enumerate(FEEDBACK_ORDER)}
    rows.sort(key=lambda r: fb_rank.get(r["Method"].split(" (")[0], 99))

    tdf = pd.DataFrame(rows)

    print("\n  RQ1 Failure Reasons:")
    print(tdf.to_string(index=False))

    _save_latex_table(tdf, os.path.join(out_dir, "rq1_failure_reasons.txt"))

    fig, ax = plt.subplots(figsize=(12, max(1.8, len(rows) * 0.45 + 1.2)))
    ax.axis("off")
    ax.set_title("RQ1: Failure Reasons (Final Solutions)", fontsize=13, pad=20)
    tbl = ax.table(cellText=tdf.values, colLabels=tdf.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(tdf.columns))))
    for j in range(len(tdf.columns)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    _save(fig, os.path.join(out_dir, "rq1_failure_reasons.png"))


def _rq1_proximity_table(fb_types, llms, out_dir, raw):
    """Multicolumn proximity-to-success table for failed samples (RQ1)."""
    raw_lookup = _build_raw_lookup(raw)

    row_labels = []
    rows = []
    for fb in fb_types:
        row_data = []
        for llm in llms:
            key = (fb, llm)
            if key in raw_lookup:
                prox = _compute_proximity(raw_lookup[key])
            else:
                prox = {k: float("nan") for k in PROX_KEYS}
            for sc in PROX_KEYS:
                v = prox[sc]
                row_data.append(f"{v:.3f}" if not np.isnan(v) else "0.0")
        row_labels.append(fb)
        rows.append(row_data)

    if not rows:
        return

    print("\n  RQ1 Proximity Table:")
    for rl, rd in zip(row_labels, rows):
        print(f"  {rl:<14}" + "  ".join(f"{v:>8}" for v in rd))

    _save_multicolumn_latex_table(
        rows, row_labels, llms, PROX_LABELS,
        os.path.join(out_dir, "rq1_proximity_table.txt"),
        caption="RQ1: Proximity to Success (Failed Samples)",
        label="tab:rq1_proximity",
        lower_is_better=True,
    )

    # Figure
    flat_cols = ["Method"]
    for llm in llms:
        for sc in PROX_LABELS:
            flat_cols.append(f"{llm} {sc}")
    flat_rows = [[rl] + rd for rl, rd in zip(row_labels, rows)]
    tdf = pd.DataFrame(flat_rows, columns=flat_cols)

    fig, ax = plt.subplots(figsize=(max(12, len(flat_cols) * 1.2),
                                    max(1.8, len(rows) * 0.45 + 1.2)))
    ax.axis("off")
    ax.set_title("RQ1: Proximity to Success (Failed Samples)", fontsize=13, pad=20)
    tbl = ax.table(cellText=tdf.values, colLabels=tdf.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(list(range(len(tdf.columns))))
    for j in range(len(tdf.columns)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    _save(fig, os.path.join(out_dir, "rq1_proximity_table.png"))


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


def rq2(summaries, out_dir, raw=None):
    print("\n=== RQ2: Comparative Analysis ===")

    ordered, methods = _build_rq2_methods(summaries)
    if not methods:
        print("  No methods found – skipping RQ2.")
        return

    _rq2_tts_bar(ordered, methods, out_dir)
    _rq2_success_grouped_bar(ordered, methods, out_dir)
    _rq2_success_heatmap(ordered, methods, out_dir)
    _rq2_table(ordered, methods, out_dir, raw=raw)
    _rq2_scaling(ordered, methods, out_dir)
    if raw is not None:
        _rq2_proximity_table(ordered, methods, out_dir, raw)
        _rq2_gsr_by_llm_scaling(summaries, out_dir, raw)
        _rq2_gsr_by_feedback_scaling(summaries, out_dir, raw)


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


def _rq2_table(ordered, M, out_dir, raw=None):
    """Multicolumn table: columns = LLMs + RL, rows = feedback types,
    sub-columns = SR_all / SR_C1 / SR_C2 / SR_C3."""

    raw_lookup = _build_raw_lookup(raw) if raw is not None else {}

    # Determine which LLMs and feedback types are present
    llm_set = set()
    fb_set = set()
    has_rl = False
    for name in ordered:
        df, fb = M[name]
        if fb == "RL":
            has_rl = True
        else:
            _, llm = None, None
            for model_name in (raw or {}):
                mfb, mllm = parse_model_name(model_name)
                lbl = _method_label(mfb, mllm, True) if mllm else mfb
                if lbl == name:
                    llm_set.add(mllm)
                    fb_set.add(mfb)
                    break

    # Fallback: extract from display names
    if not llm_set:
        for name in ordered:
            if name == "RL":
                continue
            if " (" in name:
                fb_part, llm_part = name.split(" (", 1)
                llm_set.add(llm_part.rstrip(")"))
                fb_set.add(fb_part)
            else:
                fb_set.add(name)

    llms = [l for l in LLM_ORDER if l in llm_set] + sorted(llm_set - set(LLM_ORDER))
    fb_types = [fb for fb in FEEDBACK_ORDER if fb in fb_set]

    # Column groups: LLMs first, then RL if present
    col_groups = list(llms)
    if has_rl:
        col_groups.append("RL")

    # Compute RL rates once
    rl_rates = None
    if has_rl and ("RL", None) in raw_lookup:
        rl_rates = _compute_group_rates(raw_lookup[("RL", None)])
    elif has_rl:
        # Fallback from summary
        rl_df = M.get("RL", (None, None))[0]
        if rl_df is not None:
            rl_rates = {"SR_all": rl_df["success"].mean() * 100,
                        "SR_C1": float("nan"), "SR_C2": float("nan"), "SR_C3": float("nan")}

    # Build rows
    row_labels = []
    rows = []
    for fb in fb_types:
        row_data = []
        for llm in llms:
            # Find the display name for this (fb, llm) combo
            key = (fb, llm)
            if key in raw_lookup:
                rates = _compute_group_rates(raw_lookup[key])
            else:
                # Try to find from summaries
                disp = f"{fb} ({llm})"
                if disp in M:
                    rates = {"SR_all": M[disp][0]["success"].mean() * 100,
                             "SR_C1": float("nan"), "SR_C2": float("nan"), "SR_C3": float("nan")}
                elif fb in M:
                    rates = {"SR_all": M[fb][0]["success"].mean() * 100,
                             "SR_C1": float("nan"), "SR_C2": float("nan"), "SR_C3": float("nan")}
                else:
                    rates = {"SR_all": float("nan"), "SR_C1": float("nan"),
                             "SR_C2": float("nan"), "SR_C3": float("nan")}
            for sc in RATE_KEYS:
                v = rates[sc]
                row_data.append(f"{v:.2f}" if not np.isnan(v) else "--")

        # RL column (same values for every row)
        if has_rl and rl_rates:
            for sc in RATE_KEYS:
                v = rl_rates[sc]
                row_data.append(f"{v:.2f}" if not np.isnan(v) else "--")
        elif has_rl:
            row_data.extend(["--"] * len(RATE_KEYS))

        row_labels.append(fb)
        rows.append(row_data)

    if not rows:
        # Fallback to old-style table
        _rq2_table_fallback(ordered, M, out_dir)
        return

    # Console output
    print("\n  RQ2 Summary Table:")
    for rl, rd in zip(row_labels, rows):
        print(f"  {rl:<14}" + "  ".join(f"{v:>8}" for v in rd))

    # LaTeX table
    _save_multicolumn_latex_table(
        rows, row_labels, col_groups, RATE_LABELS,
        os.path.join(out_dir, "rq2_table.txt"),
        caption="RQ2: Comparative Analysis",
        label="tab:rq2",
    )

    # Figure
    flat_cols = ["Method"]
    for cg in col_groups:
        for sc in RATE_LABELS:
            flat_cols.append(f"{cg} {sc}")
    flat_rows = [[rl] + rd for rl, rd in zip(row_labels, rows)]
    tdf = pd.DataFrame(flat_rows, columns=flat_cols)

    fig, ax = plt.subplots(figsize=(max(12, len(flat_cols) * 1.2), max(1.8, len(rows) * 0.45 + 1.2)))
    ax.axis("off")
    ax.set_title("RQ2: Comparative Summary", fontsize=13, pad=20)
    tbl = ax.table(cellText=tdf.values, colLabels=tdf.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(list(range(len(tdf.columns))))
    for j in range(len(tdf.columns)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    _save(fig, os.path.join(out_dir, "rq2_table.png"))


def _rq2_table_fallback(ordered, M, out_dir):
    """Old-style flat table as fallback when raw data isn't available."""
    rows = []
    for n in ordered:
        df = M[n][0]
        total = len(df)
        rows.append({
            "Method": n,
            "Successes": f"{int(df['success'].sum())}/{total}",
            "Success %": f"{df['success'].mean():.0%}",
            "Avg LTL Score": f"{df['final_ltl_score'].mean():.3f}",
            "Med. Cost": f"{df['final_cost'].median():.3f}",
            "Med. TTS (s)": f"{df['total_time'].median():.1f}",
        })
    tdf = pd.DataFrame(rows)
    print("\n  RQ2 Summary Table (fallback):")
    print(tdf.to_string(index=False))
    _save_latex_table(tdf, os.path.join(out_dir, "rq2_table.txt"))


def _rq2_proximity_table(ordered, M, out_dir, raw):
    """Multicolumn proximity-to-success table for RQ2 (LLMs + RL)."""
    raw_lookup = _build_raw_lookup(raw)

    # Determine LLMs, feedback types, RL presence (same logic as _rq2_table)
    llm_set = set()
    fb_set = set()
    has_rl = False
    for model_name in raw:
        fb, llm = parse_model_name(model_name)
        if fb == "RL":
            has_rl = True
        elif llm is not None:
            llm_set.add(llm)
            fb_set.add(fb)

    llms = [l for l in LLM_ORDER if l in llm_set] + sorted(llm_set - set(LLM_ORDER))
    fb_types = [fb for fb in FEEDBACK_ORDER if fb in fb_set]

    col_groups = list(llms)
    if has_rl:
        col_groups.append("RL")

    # Compute RL proximity once
    rl_prox = None
    if has_rl and ("RL", None) in raw_lookup:
        rl_prox = _compute_proximity(raw_lookup[("RL", None)])

    # Build rows
    row_labels = []
    rows = []
    for fb in fb_types:
        row_data = []
        for llm in llms:
            key = (fb, llm)
            if key in raw_lookup:
                prox = _compute_proximity(raw_lookup[key])
            else:
                prox = {k: float("nan") for k in PROX_KEYS}
            for sc in PROX_KEYS:
                v = prox[sc]
                row_data.append(f"{v:.2f}" if not np.isnan(v) else "N/A")

        if has_rl and rl_prox:
            for sc in PROX_KEYS:
                v = rl_prox[sc]
                row_data.append(f"{v:.2f}" if not np.isnan(v) else "N/A")
        elif has_rl:
            row_data.extend(["N/A"] * len(PROX_KEYS))

        row_labels.append(fb)
        rows.append(row_data)

    if not rows:
        return

    print("\n  RQ2 Proximity Table:")
    for rl, rd in zip(row_labels, rows):
        print(f"  {rl:<14}" + "  ".join(f"{v:>8}" for v in rd))

    _save_multicolumn_latex_table(
        rows, row_labels, col_groups, PROX_LABELS,
        os.path.join(out_dir, "rq2_proximity_table.txt"),
        caption="RQ2: Proximity to Success (Failed Samples)",
        label="tab:rq2_proximity",
        lower_is_better=True,
    )

    # Figure
    flat_cols = ["Method"]
    for cg in col_groups:
        for sc in PROX_LABELS:
            flat_cols.append(f"{cg} {sc}")
    flat_rows = [[rl] + rd for rl, rd in zip(row_labels, rows)]
    tdf = pd.DataFrame(flat_rows, columns=flat_cols)

    fig, ax = plt.subplots(figsize=(max(12, len(flat_cols) * 1.2),
                                    max(1.8, len(rows) * 0.45 + 1.2)))
    ax.axis("off")
    ax.set_title("RQ2: Proximity to Success (Failed Samples)", fontsize=13, pad=20)
    tbl = ax.table(cellText=tdf.values, colLabels=tdf.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(list(range(len(tdf.columns))))
    for j in range(len(tdf.columns)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    _save(fig, os.path.join(out_dir, "rq2_proximity_table.png"))


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


def _rq2_gsr_by_llm_scaling(summaries, out_dir, raw):
    """Line plot: GSR vs grid size for best-performing LLM with selected feedback types + RL.

    Feedback types shown are controlled by GSR_LLM_SCALING_FEEDBACK.
    Best-performing LLM = the LLM with highest overall GSR across those feedback types.
    """
    raw_lookup = _build_raw_lookup(raw)

    # Gather all LLMs and feedback types
    llm_set = set()
    fb_set = set()
    for model_name in raw:
        fb, llm = parse_model_name(model_name)
        if llm is not None:
            llm_set.add(llm)
            fb_set.add(fb)

    llms = [l for l in LLM_ORDER if l in llm_set] + sorted(llm_set - set(LLM_ORDER))
    # Only include feedback types that are both configured and present in data
    fb_types = [fb for fb in GSR_LLM_SCALING_FEEDBACK if fb in fb_set]

    if not llms or not fb_types:
        print("  Skipping RQ2 GSR-by-LLM scaling: not enough data.")
        return

    # Compute overall GSR for each LLM (averaging across selected feedback types)
    llm_gsr = {}
    for llm in llms:
        gsr_vals = []
        for fb in fb_types:
            key = (fb, llm)
            if key in raw_lookup:
                rates = _compute_group_rates(raw_lookup[key])
                if not np.isnan(rates["SR_all"]):
                    gsr_vals.append(rates["SR_all"])
        llm_gsr[llm] = np.mean(gsr_vals) if gsr_vals else 0

    best_llm = max(llm_gsr, key=llm_gsr.get)
    print(f"  Best-performing LLM for GSR scaling: {best_llm} (avg GSR={llm_gsr[best_llm]:.2f}%)")

    # Collect all grid sizes
    all_sizes = set()
    for key, df in raw_lookup.items():
        if "size" in df.columns:
            all_sizes.update(df["size"].unique())
        elif "size" in df.index.names:
            all_sizes.update(df.index.get_level_values("size").unique())
    all_sizes = sorted(all_sizes)

    if not all_sizes:
        print("  Skipping RQ2 GSR-by-LLM scaling: no grid sizes found.")
        return

    # Build data series: selected feedback types for the best LLM + RL
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D"]
    line_styles = ["-", "--", "-.", ":"]

    series_idx = 0

    # Add RL first
    if ("RL", None) in raw_lookup:
        df = raw_lookup[("RL", None)]
        if "is_final" in df.columns:
            finals = df[df["is_final"]]
        else:
            finals = df.groupby("sample_id").tail(1)

        gsr_by_size = finals.groupby("size")["success"].mean() * 100
        sizes = [s for s in all_sizes if s in gsr_by_size.index]
        vals = [gsr_by_size.get(s, np.nan) for s in sizes]

        ax.plot(sizes, vals,
                marker=markers[series_idx % len(markers)],
                linestyle=line_styles[series_idx % len(line_styles)],
                color=RL_COLOR,
                label="RL", linewidth=2, markersize=6)
        series_idx += 1

    for fb in fb_types:
        key = (fb, best_llm)
        if key not in raw_lookup:
            continue

        df = raw_lookup[key]
        # Get final rows
        if "is_final" in df.columns:
            finals = df[df["is_final"]]
        else:
            finals = df.groupby("sample_id").tail(1)

        gsr_by_size = finals.groupby("size")["success"].mean() * 100
        sizes = [s for s in all_sizes if s in gsr_by_size.index]
        vals = [gsr_by_size.get(s, np.nan) for s in sizes]

        label = f"{fb} ({best_llm})"
        ax.plot(sizes, vals,
                marker=markers[series_idx % len(markers)],
                linestyle=line_styles[series_idx % len(line_styles)],
                color=FEEDBACK_COLORS.get(fb, "#999"),
                label=label, linewidth=2, markersize=6)
        series_idx += 1

    fb_label = ", ".join(fb_types)
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("$SR_{all}$ (%)")
    ax.set_title(f"RQ2: $SR_{{all}}$ vs Grid Size – RL + {fb_label} ({best_llm})")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)
    _save(fig, os.path.join(out_dir, "rq2_gsr_by_llm_scaling.png"))


def _rq2_gsr_by_feedback_scaling(summaries, out_dir, raw):
    """Line plot: GSR vs grid size for a specified feedback type (all its LLMs) + RL.

    Feedback type is controlled by GSR_FEEDBACK_SCALING_METHOD.
    """
    raw_lookup = _build_raw_lookup(raw)

    # Gather all LLMs and feedback types
    llm_set = set()
    fb_set = set()
    for model_name in raw:
        fb, llm = parse_model_name(model_name)
        if llm is not None:
            llm_set.add(llm)
            fb_set.add(fb)

    llms = [l for l in LLM_ORDER if l in llm_set] + sorted(llm_set - set(LLM_ORDER))

    if not llms:
        print("  Skipping RQ2 GSR-by-Feedback scaling: not enough data.")
        return

    best_fb = GSR_FEEDBACK_SCALING_METHOD
    if best_fb not in fb_set:
        print(f"  Skipping RQ2 GSR-by-Feedback scaling: {best_fb} not found in data.")
        return
    print(f"  Using feedback method for GSR scaling: {best_fb}")

    # Collect all grid sizes
    all_sizes = set()
    for key, df in raw_lookup.items():
        if "size" in df.columns:
            all_sizes.update(df["size"].unique())
        elif "size" in df.index.names:
            all_sizes.update(df.index.get_level_values("size").unique())
    all_sizes = sorted(all_sizes)

    if not all_sizes:
        print("  Skipping RQ2 GSR-by-Feedback scaling: no grid sizes found.")
        return

    # Define colors for LLMs
    llm_colors = {
        "GPT-5 Nano": "#3498DB",
        "GPT-5 Mini": "#9B59B6",
        "Gemini 2.5 Pro": "#E67E22",
    }

    # Build data series: 3 LLMs for the best feedback + RL
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^", "D"]
    line_styles = ["-", "--", "-.", ":"]

    series_idx = 0
    for llm in llms:
        key = (best_fb, llm)
        if key not in raw_lookup:
            continue

        df = raw_lookup[key]
        # Get final rows
        if "is_final" in df.columns:
            finals = df[df["is_final"]]
        else:
            finals = df.groupby("sample_id").tail(1)

        gsr_by_size = finals.groupby("size")["success"].mean() * 100
        sizes = [s for s in all_sizes if s in gsr_by_size.index]
        vals = [gsr_by_size.get(s, np.nan) for s in sizes]

        label = f"{best_fb} ({llm})"
        ax.plot(sizes, vals,
                marker=markers[series_idx % len(markers)],
                linestyle=line_styles[series_idx % len(line_styles)],
                color=llm_colors.get(llm, "#999"),
                label=label, linewidth=2, markersize=6)
        series_idx += 1

    # Add RL
    if ("RL", None) in raw_lookup:
        df = raw_lookup[("RL", None)]
        if "is_final" in df.columns:
            finals = df[df["is_final"]]
        else:
            finals = df.groupby("sample_id").tail(1)

        gsr_by_size = finals.groupby("size")["success"].mean() * 100
        sizes = [s for s in all_sizes if s in gsr_by_size.index]
        vals = [gsr_by_size.get(s, np.nan) for s in sizes]

        ax.plot(sizes, vals,
                marker=markers[series_idx % len(markers)],
                linestyle=line_styles[series_idx % len(line_styles)],
                color=RL_COLOR,
                label="RL", linewidth=2, markersize=6)

    ax.set_xlabel("Grid Size")
    ax.set_ylabel("$SR_{all}$ (%)")
    ax.set_title(f"RQ2: $SR_{{all}}$ vs Grid Size – {best_fb} + RL")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)
    _save(fig, os.path.join(out_dir, "rq2_gsr_by_feedback_scaling.png"))


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
        print("  No LLM methods found - skipping RQ3.")
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
RUN_FOLDER = "100_20260211_08-03-04-repaired-repaired"


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

    empty_models = [name for name, df in raw.items() if df.empty]
    if empty_models:
        print(f"WARNING: {len(empty_models)} model(s) have empty results and will be skipped: "
              f"{', '.join(empty_models)}")
    raw = {name: df for name, df in raw.items() if not df.empty}
    summaries = {name: summarize_samples(df) for name, df in raw.items()}

    out_dir = os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    rq1(summaries, out_dir, raw=raw)
    rq2(summaries, out_dir, raw=raw)
    rq3(summaries, out_dir)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()