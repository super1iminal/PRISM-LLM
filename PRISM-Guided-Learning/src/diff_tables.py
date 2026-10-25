"""Every number that changed in the paper's tables against a previous version (git), for out/paper/certs.csv's report.

Each table is compared with its previous version: Asher's tables with git HEAD, Rajan's with his branch, and the
merged tables with the ones they replace. Rows are matched on the table's text columns; every numeric cell that moved
by more than TOL (or appeared or vanished) is listed.

Usage (from PRISM-Guided-Learning/): python src/diff_tables.py [--rajan-ref origin/rajanmaghera/fir-harness] [--out FILE]
"""
import argparse, io, subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from settings import PROJECT_ROOT

TABLES = PROJECT_ROOT / "out" / "paper" / "tables"
TOL = 1e-9
# new table -> (git ref, old path) where its previous version lives elsewhere or under another name
RENAMED = {"tab_rq1-coverage-llm.csv": [("RAJAN", "tab_rq1-coverage-llm.rajan.csv"), ("HEAD", "tab_rq1-coverage-llm.csv")],
           "tab_rq3-ablation-llm.csv": [("RAJAN", "tab_rq3-ablation-llm.rajan.csv")],
           "tab_rq1-llm-spend.csv": [("RAJAN", "tab_rq1-llm-spend.rajan.csv")]}


def old_version(ref: str, name: str):
    path = f"PRISM-Guided-Learning/out/paper/tables/{name}"
    out = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True, cwd=PROJECT_ROOT)
    return pd.read_csv(io.StringIO(out.stdout)) if out.returncode == 0 and out.stdout.strip() else None


def compare(old: pd.DataFrame, new: pd.DataFrame, table: str, origin: str) -> list:
    num_old = {c for c in old if pd.api.types.is_numeric_dtype(old[c])}
    num_new = {c for c in new if pd.api.types.is_numeric_dtype(new[c])}
    keys = [c for c in new if c in old and c not in num_old | num_new]
    values = sorted((num_old | num_new) & set(old) & set(new))
    if not keys:
        keys = ["_row"]
        old, new = old.assign(_row=range(len(old))), new.assign(_row=range(len(new)))
    o = old.groupby(keys, dropna=False)[values].first() if values else old.groupby(keys, dropna=False).size().to_frame("n")
    n = new.groupby(keys, dropna=False)[values].first() if values else new.groupby(keys, dropna=False).size().to_frame("n")
    m = o.join(n, how="outer", lsuffix="__old", rsuffix="__new")
    rows = []
    for col in (values or ["n"]):
        a = pd.to_numeric(m[f"{col}__old"], errors="coerce").astype(float)
        b = pd.to_numeric(m[f"{col}__new"], errors="coerce").astype(float)
        moved = ((a - b).abs() > TOL) | (a.isna() != b.isna())
        for key, x, y in zip(m.index[moved], a[moved], b[moved]):
            key = key if isinstance(key, tuple) else (key,)
            kind = "new" if np.isnan(x) else "removed" if np.isnan(y) else "changed"
            rows.append({"table": table, "compared_with": origin, "row": " / ".join(str(k) for k in key if str(k) != "nan"),
                         "column": col, "kind": kind, "old": x, "new": y,
                         "change": (y - x) if kind == "changed" else np.nan})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rajan-ref", default="origin/rajanmaghera/fir-harness")
    ap.add_argument("--ref", default="HEAD", help="git ref of the previous version of Asher's tables")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "out" / "paper" / "tables-changed.csv")
    a = ap.parse_args()
    refs = {"HEAD": a.ref, "RAJAN": a.rajan_ref}
    rows = []
    for path in sorted(TABLES.glob("*.csv")):
        new = pd.read_csv(path)
        candidates = RENAMED.get(path.name) or [("RAJAN" if ".rajan" in path.name or "synthesis" in path.name
                                                  or "random-sequences" in path.name else "HEAD", path.name)]
        found = False
        for ref, name in candidates:
            old = old_version(refs[ref], name)
            if old is not None:
                rows += compare(old, new, path.name, f"{refs[ref]}:{name}")
                found = True
        if not found:
            rows.append({"table": path.name, "compared_with": "", "row": "(new table)", "column": "", "kind": "new table",
                         "old": np.nan, "new": np.nan, "change": np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} changed cells in {df.table.nunique()} tables")
    print(df.groupby(["table", "kind"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
