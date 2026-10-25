"""Verdicts from Storm's results on storm_prep.py's products: per requirement the rational result of `--exact`, else
the value of `--sound` (precision SOUND_EPS, absolute); value = 1 - P[F violation] for safety requirements. A check
is certified if every requirement holds, failed if one is violated (exactly, or beyond the sound precision), else
undecided. Usage: python storm_collect.py <storm dir> <out csv>"""
import json, re, sys
from fractions import Fraction
from pathlib import Path

import pandas as pd

SOUND_EPS = Fraction(1, 10 ** 12)
RESULT = re.compile(r"Result \(for initial states\): (\S+)")
SECONDS = re.compile(r"^rc=(\d+) seconds=(\S+)", re.M)


def read(path: Path):
    """(value as a Fraction or None, return code, seconds) of one Storm output file."""
    if not path.exists():
        return None, None, None
    text = path.read_text(errors="replace")
    m, s = RESULT.search(text), SECONDS.search(text)
    rc, secs = (int(s.group(1)), float(s.group(2)) if s.group(2) != "?" else None) if s else (None, None)
    if not m or rc != 0:
        return None, rc, secs
    return Fraction(m.group(1)), rc, secs


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    for cid, c in sorted(manifest.items()):
        margins, how, secs, missing = {}, set(), 0.0, []
        for name, e in c["requirements"].items():
            thr = Fraction(str(e["threshold"]))
            maximize = e["bound"] == ">="
            if e["fixed"] is not None:
                val, eps, kind = Fraction(str(e["fixed"])), Fraction(0), "fixed"
            else:
                val, rc, s = read(root / cid / f"{name}.exact.out")
                eps, kind = Fraction(0), "exact"
                if val is None:
                    val, rc, s = read(root / cid / f"{name}.sound.out")
                    eps, kind = SOUND_EPS, "sound"
                secs += s or 0.0
                if val is None:
                    missing.append(name)
                    continue
                if e["safety"]:
                    val = 1 - val
            how.add(kind)
            margins[name] = ((val - thr) if maximize else (thr - val), eps)
        violated = [k for k, (m, eps) in margins.items() if m < -eps]
        holds = not missing and all(m >= eps for m, eps in margins.values())
        outcome = "failed" if violated else "certified" if holds else "undecided"
        binding = (min(violated, key=lambda k: margins[k][0]) if violated else
                   min(margins, key=lambda k: margins[k][0]) if margins else "")
        rows.append({"check": cid, "file": c["file"], "source": c["source"], "target": c["target"],
                     "episode": c["episode"], "outcome": outcome,
                     "margin": float(margins[binding][0]) if binding else None, "binding": binding,
                     "method": "Storm " + "+".join(sorted(how - {"fixed"})), "storm_seconds": secs,
                     "missing": " ".join(missing), "states": c["states"]})
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.outcome.value_counts().to_dict(), df.method.value_counts().to_dict())


if __name__ == "__main__":
    main()
