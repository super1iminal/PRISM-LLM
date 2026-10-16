"""Re-run the final stage (repair + margin selection) on the attempts logged in method.py output directories.

    python src/fresh/finalize_dir.py <family> <dir> [<dir> ...] --out <file.json> [--top 8 --steps 40 --phases]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fresh.gw import FAMILIES, load  # noqa: E402
from fresh.method import finalize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("family")
ap.add_argument("dirs", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--top", type=int, default=8)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--phases", action="store_true")
args = ap.parse_args()
members = load(FAMILIES[args.family])
if args.phases:
    for m in members:
        m.all_phases()
attempts = []
for d in args.dirs:
    for f in sorted(Path(d).glob(f"{args.family}_c*.jsonl")):
        for line in open(f, encoding="utf-8"):
            a = json.loads(line)
            a.pop("prompt", None)
            a.pop("answer", None)
            if a.get("rules") and not a.get("error"):
                a["score"] = tuple(a["score"])
                attempts.append(a)
final, finals = finalize(members, attempts, args.top, args.steps)
Path(args.out).write_text(json.dumps({"best": final, "finals": finals}, indent=1, default=str), encoding="utf-8")
for f in finals:
    print(f["score"], f["from"], f["llm_score"])
print("best", final["score"])
for c, a in final["rules"]:
    print(f"  {c} -> {a}")
