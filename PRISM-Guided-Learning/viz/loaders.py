"""Load per-sample outcomes of legacy and symbolic runs into one tidy table.

Both loaders return a DataFrame indexed by sample_id with columns:
    approach, size, complete (all attempts ran), attempts, time_s, output_tokens (NaN if unknown),
    p_<requirement>            final probability (symbolic: worst case)
    p_best_<requirement>       symbolic only: best case
Symbolic runs are read from their parquet when present, otherwise reconstructed from the
worker logs, which is how partial/aborted runs (which never saved a parquet) are plotted.
"""
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from compare import _legacy_kept  # noqa: E402  (exact legacy keep-best rule)
from core.domain import load_domain  # noqa: E402
from legacy.requirements import get_threshold_for_key  # noqa: E402

_RESULTS = re.compile(r"(\w+): best=([\d.]+) worst=([\d.]+)")
_TIMESTAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+)")
_INSTANCE = re.compile(r"Instance (\d+): success=(\w+) iterations=(\d+) time=([\d.]+)s")


def load_legacy(run_dir: Path, dataset: str = "grid_20_balanced.csv") -> pd.DataFrame:
    instances = load_domain("gridworld").load_instances(dataset)
    df = pd.read_parquet(run_dir / "LEGACY_FEEDBACK_SIMPLIFIED_results.parquet").reset_index()
    final = df[df.is_final].set_index("sample_id")
    rows = {}
    for path in sorted((run_dir / "outputs").glob("sample_*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        sid = rec["sample_id"]
        probs = _legacy_kept(rec, instances[sid])
        f = final.loc[sid]
        rows[sid] = {"approach": "legacy", "size": int(f["size"]), "complete": True,
                     "attempts": int(df[df.sample_id == sid].iteration.max()), "time_s": f["total_time"],
                     "output_tokens": f["llm_output_tokens"], **{f"p_{k}": p for k, p in probs.items()}}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("sample_id")


def load_symbolic(run_dir: Path, sizes: Optional[dict] = None, include_partial: bool = False) -> pd.DataFrame:
    parquet = run_dir / "SYMBOLIC_results.parquet"
    if parquet.exists():
        df = pd.read_parquet(parquet).reset_index()
        final = df[df.is_final].set_index("sample_id")
        rows = {}
        for sid, f in final.iterrows():
            reqs = [c[len("final_worst_"):] for c in final.columns if c.startswith("final_worst_")]
            rows[sid] = {"approach": "symbolic", "size": sizes.get(sid) if sizes else np.nan, "complete": True,
                         "attempts": int(df[df.sample_id == sid].iteration.max()), "time_s": f["total_time"],
                         "output_tokens": f["llm_output_tokens"],
                         **{f"p_{r}": f[f"final_worst_{r}"] for r in reqs},
                         **{f"p_best_{r}": f[f"final_best_{r}"] for r in reqs}}
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis("sample_id")
    return _symbolic_from_logs(run_dir, sizes, include_partial)


def _symbolic_from_logs(run_dir: Path, sizes: Optional[dict], include_partial: bool) -> pd.DataFrame:
    finished = {}
    main_log = run_dir / "main.log"
    if main_log.exists():
        for line in main_log.read_text(encoding="utf-8").splitlines():
            m = _INSTANCE.search(line)
            if m:
                finished[int(m.group(1))] = float(m.group(4))
    rows = {}
    for log in sorted(run_dir.glob("worker_*.log")):
        sid = int(log.stem.split("_")[1])
        lines = log.read_text(encoding="utf-8").splitlines()
        attempts, kept, first_ts, last_ts = 0, None, None, None
        pending = None
        for line in lines:
            ts = _TIMESTAMP.match(line)
            if ts and first_ts is None:
                first_ts = pd.Timestamp(ts.group(1))
            if " - INFO - Results: " in line:
                pending = {r: (float(b), float(w)) for r, b, w in _RESULTS.findall(line)}
            elif " - INFO - Score " in line and pending is not None:
                attempts += 1
                last_ts = pd.Timestamp(_TIMESTAMP.match(line).group(1))
                if "(new best)" in line:
                    kept = pending
                pending = None
        complete = sid in finished
        if kept is None or (not complete and not include_partial):
            continue
        time_s = finished.get(sid, (last_ts - first_ts).total_seconds() if last_ts is not None else np.nan)
        rows[sid] = {"approach": "symbolic", "size": sizes.get(sid) if sizes else np.nan, "complete": complete,
                     "attempts": attempts, "time_s": time_s, "output_tokens": np.nan,
                     **{f"p_{r}": w for r, (b, w) in kept.items()},
                     **{f"p_best_{r}": b for r, (b, w) in kept.items()}}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("sample_id").sort_index()


def requirement_names(df: pd.DataFrame) -> list:
    return [c[2:] for c in df.columns if c.startswith("p_") and not c.startswith("p_best_")]


def add_summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """requirements met and total shortfall below thresholds (worst case for symbolic)."""
    reqs = requirement_names(df)
    df = df.copy()
    df["met"] = sum((df[f"p_{r}"] >= get_threshold_for_key(r)).astype(int) for r in reqs)
    df["shortfall"] = sum((get_threshold_for_key(r) - df[f"p_{r}"]).clip(lower=0) for r in reqs)
    if all(f"p_best_{r}" in df for r in reqs):
        df["met_best"] = sum((df[f"p_best_{r}"] >= get_threshold_for_key(r)).astype(int) for r in reqs)
    return df
