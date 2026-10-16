"""Print each chain's attempts (score, error) from a method.py output directory."""
import json
import sys
from pathlib import Path

for d in sys.argv[1:]:
    for f in sorted(Path(d).glob("*.jsonl")):
        rows = [json.loads(l) for l in open(f, encoding="utf-8")]
        cells = []
        for r in rows:
            if r.get("error"):
                cells.append("ERR")
            else:
                s = r["score"]
                cells.append(f"{s[0]}/{-s[1]:.2f}/{s[2]:+.2f}")
        cost = sum(r["usage"]["cost"] for r in rows)
        print(f"{f.stem:14s} ${cost:.3f}  " + "  ".join(cells))
