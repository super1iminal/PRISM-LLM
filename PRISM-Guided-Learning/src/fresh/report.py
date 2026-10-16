"""Summarize final runs: per family the kept list, its training margins, the LLM cost, and (once certified) the
validation members certified.

    python src/fresh/report.py out/fresh/final/run1 out/fresh/final/run2 out/fresh/final/run3
    python src/fresh/report.py --write-cert out/fresh/final/run1 ...   (write <run>/cert/ for transfer.py)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fresh.certify import write_run  # noqa: E402
from fresh.tally import tally  # noqa: E402

FAMS = ["rooms", "crossing", "adjacent"]

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--write-cert", action="store_true")
args = ap.parse_args()

counts = {f: [] for f in FAMS}
for run in map(Path, args.runs):
    lists, cost, tokens = {}, 0.0, 0
    print(f"## {run}")
    for fam in FAMS:
        s = json.loads((run / f"{fam}_summary.json").read_text(encoding="utf-8"))
        b = s["best"]
        lists[fam] = [tuple(r) for r in b["det_rules"]]
        cost += s["usage"]["cost"]
        tokens += s["usage"]["prompt_tokens"] + s["usage"]["output_tokens"]
        print(f"- {fam}: from {b['from']}, {len(b['rules'])} rules ({len(b['det_rules'])} after resolving `any` by "
              f"{b['det_order']}); training margin any {b['score'][2]:+.3f}, deterministic {b['det_score'][2]:+.3f}; "
              f"LLM ${s['usage']['cost']:.3f}, {s['usage']['calls']} calls, "
              f"{s['usage']['prompt_tokens'] + s['usage']['output_tokens']} tokens, {s['seconds']:.0f} s")
    print(f"- run cost ${cost:.3f}, {tokens} tokens")
    if args.write_cert:
        write_run(run / "cert", lists)
    if (run / "cert" / "transfer_families_v0.csv").exists():
        t = tally(run / "cert")
        for fam in FAMS:
            counts[fam].append(t[fam])
        print(f"- certified: rooms {t['rooms']}/20, crossing {t['crossing']}/20, adjacent {t['adjacent']}/20 "
              f"(on all 60: {t['rooms_all60']}, {t['crossing_all60']}, {t['adjacent_all60']})")
if all(counts[f] for f in FAMS):
    k = len(counts["rooms"])
    means = {f: sum(v) / k for f, v in counts.items()}
    print(f"\n| family | mean | range | runs |\n|---|---|---|---|")
    for f in FAMS:
        print(f"| {f} | {means[f]:.2f} | {min(counts[f])}-{max(counts[f])} | {counts[f]} |")
    per_run = [sum(counts[f][i] for f in FAMS) / 3 for i in range(k)]
    print(f"| mean over families | {sum(means.values()) / 3:.2f} | {min(per_run):.2f}-{max(per_run):.2f} | {per_run} |")
