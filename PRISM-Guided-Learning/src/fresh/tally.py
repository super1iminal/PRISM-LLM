"""Per list: how many of its own family's 20 validation members it certifies (from transfer_families_v0.csv)."""
import sys
from pathlib import Path

import pandas as pd

FAMS = [("rooms", range(16, 36)), ("crossing", range(76, 96)), ("adjacent", range(136, 156))]


def tally(run: Path):
    df = pd.read_csv(run / "transfer_families_v0.csv")
    out = {}
    for i, (fam, targets) in enumerate(FAMS):
        sub = df[(df.source == i) & df.target.isin(list(targets))]
        out[fam] = int(sub.certified.sum())
        out[fam + "_n"] = len(sub)
        out[fam + "_all60"] = int(df[df.source == i].certified.sum())
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        t = tally(Path(d))
        mean = sum(t[f] for f, _ in FAMS) / 3
        print(f"{d}: rooms {t['rooms']}/{t['rooms_n']}  crossing {t['crossing']}/{t['crossing_n']}  "
              f"adjacent {t['adjacent']}/{t['adjacent_n']}  mean {mean:.2f}   (all 60: "
              f"{t['rooms_all60']}, {t['crossing_all60']}, {t['adjacent_all60']})")
