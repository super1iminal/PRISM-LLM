"""Print what a domain's requirement thresholds are calibrated against.

For each instance of a dataset: the best and worst value of each requirement over all controllers
(bare MDP; for a `<=` requirement such as energy or crash, best = lowest), and the worst-case values
of the domain's reference policies (`domains/<name>/data/reference_policies.json`), marking the
requirements each one fails.

Usage (from PRISM-Guided-Learning/): python src/calibrate.py --domain uuv --data uuv_paper.csv
"""
import argparse

from config import load_config
from core.domain import load_domain
from core.verifier import PolicyVerifier


def calibrate(domain_name: str, dataset: str) -> None:
    domain = load_domain(domain_name)
    cfg = load_config()
    references = domain.reference_policies()
    for instance in domain.load_instances(dataset):
        verifier = PolicyVerifier(domain, instance, cfg)
        spec = verifier.spec
        props = [f"{op}=? [ {r.formula} ]" for r in spec.requirements for op in (r.best_op(), r.worst_op())]
        bounds = verifier.runner.run(verifier.model, props).initial_values
        print(f"{instance.data.get('name', instance.id)}: " + ", ".join(
            f"{r.name}: best {bounds[2 * i]:.4f}, worst {bounds[2 * i + 1]:.4f} (threshold {r.bound} {r.threshold})"
            for i, r in enumerate(spec.requirements)), flush=True)
        for name, rules in references.items():
            v = verifier.verify(spec.policy(rules), analysis=False)
            print(f"  {name:16s} " + "  ".join(
                f"{r.name}={v.worst[r.name]:.4f}{'' if r.satisfied(v.worst[r.name]) else ' (fails)'}"
                for r in spec.requirements), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", required=True)
    parser.add_argument("--data", required=True, help="Dataset in domains/<domain>/data/")
    args = parser.parse_args()
    calibrate(args.domain, args.data)


if __name__ == "__main__":
    main()
