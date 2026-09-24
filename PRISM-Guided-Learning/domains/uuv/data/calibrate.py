"""Print what the requirement thresholds in a UUV dataset are calibrated against.

For each scenario: the best and worst value of each requirement over all controllers (bare MDP),
and the worst-case values of the reference policies in reference_policies.json. Run from
PRISM-Guided-Learning/:
    python domains/uuv/data/calibrate.py [uuv_paper.csv]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from core.domain import load_domain  # noqa: E402
from core.rules import SymbolicPolicy  # noqa: E402
from core.verifier import PolicyVerifier  # noqa: E402

REFERENCE = json.loads((Path(__file__).parent / "reference_policies.json").read_text(encoding="utf-8"))


def main(dataset: str = "uuv_paper.csv") -> None:
    domain = load_domain("uuv")
    for instance in domain.load_instances(dataset):
        verifier = PolicyVerifier(domain, instance)
        spec = verifier.spec
        props = [f"{op}=? [ {r.formula} ]" for r in spec.requirements for op in ("Pmax", "Pmin")]
        bounds = verifier.runner.run(verifier.model, props).initial_values
        print(f"{instance.data['name']}: " + ", ".join(
            f"{r.name} in [{bounds[2 * i + 1]:.4f}, {bounds[2 * i]:.4f}] (threshold {r.threshold})"
            for i, r in enumerate(spec.requirements)))
        for name, rules in REFERENCE.items():
            v = verifier.verify(SymbolicPolicy.from_raw(spec.variables, list(spec.actions), rules), analysis=False)
            print(f"  {name:12s} " + "  ".join(
                f"{r.name}={v.worst[r.name]:.4f}{'' if r.satisfied(v.worst[r.name]) else ' (fails)'}"
                for r in spec.requirements))


if __name__ == "__main__":
    main(*sys.argv[1:])
