"""Build sequential_v0.csv, the dataset of the sequential adaptation study: every member of families_v0.csv (ids
0-179, unchanged, so the core runs' training sets keep their ids) followed by the 20 random grids of
grid_20_large.csv (ids 180-199, family "random"). One dataset lets a generation run train on family members and a
random grid together. The families' sealed test ids stay sealed (configs/transfer/default.yaml).

Usage (from artifact/): python domains/gridworld/data/sequential.py
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent


def main():
    families = pd.read_csv(HERE / "families_v0.csv")
    grids = pd.read_csv(HERE / "grid_20_large.csv").assign(family="random", seed=-1)
    out = pd.concat([families, grids[families.columns]], ignore_index=True)
    out.to_csv(HERE / "sequential_v0.csv", index=False)
    print(f"{len(out)} instances: {len(families)} family members (0-{len(families) - 1}), "
          f"{len(grids)} random grids ({len(families)}-{len(out) - 1})")


if __name__ == "__main__":
    main()
