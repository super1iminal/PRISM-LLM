"""Build cert_queue.py entries from finished runs (each run's kept list with `any` resolved).

    python src/fresh/make_queue.py out.json families <run_dir> [--tag inloop]
    python src/fresh/make_queue.py out.json grids <dir> --prefix r1 [--hide hazard,closer]
    python src/fresh/make_queue.py out.json mer <run_dir> --dataset mer_transfer.csv
    python src/fresh/make_queue.py out.json pacman <run_dir>
Appends to out.json if it exists.
"""
import argparse
import json
from pathlib import Path

FAMILIES = {"rooms": (["0", "4", "8"], "16-35"), "crossing": (["60", "64", "68"], "76-95"),
            "adjacent": (["120", "124", "128"], "136-155")}
SETS = [["0", "4", "8"], ["1", "5", "9"], ["2", "6", "10"], ["3", "7", "11"]]


def kept(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))["best"]["det_rules"]


ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("benchmark", choices=["families", "grids", "mer", "pacman"])
ap.add_argument("dir")
ap.add_argument("--prefix", default="")
ap.add_argument("--hide", default="hazard")
ap.add_argument("--dataset", default="mer_transfer.csv")
args = ap.parse_args()
d = Path(args.dir)
items = []
if args.benchmark == "families":
    config = {"approach": "symbolic", "domain": {"name": "gridworld", "dataset": "families_v0.csv"},
              "rules": {"extended": True, "general": True, "hidden_features": ["hazard"]}}
    for fam, (members, targets) in FAMILIES.items():
        items.append({"dir": str(d / "cert" / fam), "config": config, "targets": targets,
                      "samples": [{"rules": kept(d / f"{fam}_summary.json"), "members": members}]})
elif args.benchmark == "grids":
    config = {"approach": "symbolic", "domain": {"name": "gridworld", "dataset": "grid_20_large.csv"},
              "rules": {"extended": True, "general": True, "hidden_features": [h for h in args.hide.split(",") if h]}}
    items.append({"dir": str(d / f"cert_{args.prefix}"), "config": config, "targets": "12-19",
                  "samples": [{"rules": kept(d / f"{args.prefix}_set{i}_summary.json"), "members": m}
                              for i, m in enumerate(SETS)]})
elif args.benchmark == "mer":
    config = {"approach": "symbolic", "domain": {"name": "mer", "dataset": args.dataset},
              "rules": {"extended": True, "general": True}}
    items.append({"dir": str(d / "cert"), "config": config, "targets": "0-8",
                  "samples": [{"rules": kept(d / "mer_summary.json"), "members": ["0", "1", "2"]}]})
else:
    config = {"approach": "symbolic", "domain": {"name": "pacman", "dataset": "pacman_transfer.csv"},
              "rules": {"extended": True, "general": True}}
    items.append({"dir": str(d / "cert"), "config": config, "targets": "0-7,9",
                  "samples": [{"rules": kept(d / "pacman_summary.json"), "members": ["3", "4", "5"]}]})
out = Path(args.out)
queue = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
out.write_text(json.dumps(queue + items, indent=1), encoding="utf-8")
print(f"{len(items)} entries -> {out} ({len(queue) + len(items)} in total)")
