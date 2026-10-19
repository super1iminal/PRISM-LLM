"""Draw the sequences of a sequential study with a seeded RNG and write them into its study file (configs/sequential/),
so they are fixed and committed before the first episode runs (rule 7 of Rajan's SEAMS checklist).

    python src/draw_sequences.py grid --name random_v0 --count 10 --length 12 --seed 21001 \
        --seed-runs out/results/core/port_fam_history/seed_none_1 ..._2 ..._3
    python src/draw_sequences.py pacman --name pacman_v0 --count 3 --length 12 --seed 21003 \
        --seed-runs out/results/core/port_pacman_narrow_core/seed_none_1

Pools (validation instances only; never sealed test ids):
  grid: sequential_v0.csv's validation family members (ids 16-35 rooms, 76-95 crossing, 136-155 adjacent; sizes
        10-14) and the 20 random grids of grid_20_large (ids 180-199, sizes 9-13): shift_v0's pool. Each sequence draws
        `length` ids uniformly at random without repeats (numpy's default_rng(seed).choice(pool, length, replace=False)).
  pacman: pacman_transfer.csv's validation lengths 8, 10, 12, 20, 25, 30 (ids 0, 1, 2, 6, 7, 9). Each sequence holds
        every length once plus `length - 6` more drawn uniformly with replacement, in a random order, so every length
        occurs at least once and lengths below the training lengths (14-18) mix with lengths above them.
The sequences of one draw are the same in every study file it writes (one per seed run: `<name>`, `<name>_lib2`, ...),
so libraries are compared on identical sequences. The file's header comment records the seed, the pool and the
procedure (src/sequential.py reads only the study's own keys).
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

from settings import PROJECT_ROOT

POOLS = {
    "grid": dict(dataset="sequential_v0.csv", condition="port_fam_history",
                 ids=list(range(16, 36)) + list(range(76, 96)) + list(range(136, 156)) + list(range(180, 200))),
    "pacman": dict(dataset="pacman_transfer.csv", condition="port_pacman_narrow_core", ids=[0, 1, 2, 6, 7, 9]),
}


def draw(kind: str, count: int, length: int, seed: int):
    rng = np.random.default_rng(seed)
    pool = POOLS[kind]["ids"]
    seqs = []
    for k in range(count):
        if kind == "grid":
            ids = rng.choice(pool, size=length, replace=False)
        else:
            if length < len(pool):
                raise SystemExit("pacman: a sequence must hold every length at least once")
            ids = rng.permutation(list(pool) + list(rng.choice(pool, size=length - len(pool), replace=True)))
        seqs.append({"name": f"r{k + 1:02d}", "ids": [int(i) for i in ids]})
    return seqs


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("kind", choices=sorted(POOLS))
    p.add_argument("--name", required=True, help="study name (the first library's file; others get _lib<k>)")
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--seed-runs", nargs="+", required=True, help="one study file per seed library")
    p.add_argument("--note", default="", help="a line on what the study is for")
    a = p.parse_args()
    pool = POOLS[a.kind]
    seqs = draw(a.kind, a.count, a.length, a.seed)
    for k, run in enumerate(a.seed_runs, 1):
        name = a.name if k == 1 else f"{a.name}_lib{k}"
        head = [f"# {a.note}" if a.note else None,
                f"# Drawn by src/draw_sequences.py {a.kind} --count {a.count} --length {a.length} --seed {a.seed}"
                f" (numpy default_rng); same sequences in every library's file of this draw.",
                f"# Pool: {pool['dataset']} ids {pool['ids']} (validation only)."]
        study = {"name": name, "condition": pool["condition"], "dataset": pool["dataset"], "seed_run": run}
        if a.kind == "grid":
            study.update(generation_set=["run.workers=2"], max_training=4)
        study["sequences"] = seqs
        path = PROJECT_ROOT / "configs" / "sequential" / f"{name}.yaml"
        text = "\n".join(h for h in head if h) + "\n" + yaml.safe_dump(study, sort_keys=False, default_flow_style=None)
        path.write_text(text, encoding="utf-8")
        print(f"{path}: {len(seqs)} sequences x {a.length}")


if __name__ == "__main__":
    main()
