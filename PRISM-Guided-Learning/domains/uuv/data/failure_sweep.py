"""Is "one controller across failure rates" a non-trivial family? A PRISM-only check, no LLM.

Each member is a paper scenario with the dataset option `fail_scale` = k: every thruster-failure
probability (search at each altitude, and following) times k, the probability of staying in the same
state absorbing the difference. A member's thresholds keep the slack the paper scenario allows:
threshold = the member's best value over all controllers minus (plus, for energy) the scenario's slack at
k = 1. Then checks which reference policies (reference_policies.json) meet every requirement of every
member. If one of them does, the family is trivial: a fixed simple policy covers it.

Run from PRISM-Guided-Learning/:
    python domains/uuv/data/failure_sweep.py [uuv_paper.csv]
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from config import load_config  # noqa: E402
from core.domain import load_domain  # noqa: E402
from core.rules import SymbolicPolicy  # noqa: E402
from core.verifier import PolicyVerifier  # noqa: E402

REFERENCE = json.loads((Path(__file__).parent / "reference_policies.json").read_text(encoding="utf-8"))
FACTORS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def values(domain, instance, cfg, policies):
    """Best and worst over all controllers, and each policy's worst case, per requirement."""
    verifier = PolicyVerifier(domain, instance, cfg)
    reqs = verifier.spec.requirements
    props = [f"{op}=? [ {r.formula} ]" for r in reqs for op in (r.best_op(), r.worst_op())]
    bounds = verifier.runner.run(verifier.model, props).initial_values
    out = {"best": {r.name: bounds[2 * i] for i, r in enumerate(reqs)}, "bound": {r.name: r.bound for r in reqs},
           "threshold": {r.name: r.threshold for r in reqs}}
    for name, rules in policies.items():
        v = verifier.verify(SymbolicPolicy.from_raw(verifier.spec.variables, list(verifier.spec.actions), rules),
                            analysis=False)
        out[name] = dict(v.worst)
    return out


def main(dataset: str = "uuv_paper.csv") -> None:
    domain = load_domain("uuv")
    cfg = load_config()
    passes_all = {name: True for name in REFERENCE}
    for instance in domain.load_instances(dataset):
        base = values(domain, instance, cfg, {})
        slack = {r: abs(base["best"][r] - base["threshold"][r]) for r in base["best"]}
        print(f"{instance.data['name']}: slack at k=1 " + ", ".join(f"{r} {s:.4f}" for r, s in slack.items()))
        for k in FACTORS:
            v = values(domain, replace(instance, data={**instance.data, "fail_scale": k}), cfg, REFERENCE)
            thr = {r: v["best"][r] - slack[r] if v["bound"][r] == ">=" else v["best"][r] + slack[r] for r in slack}
            ok = lambda r, x: x >= thr[r] - 1e-9 if v["bound"][r] == ">=" else x <= thr[r] + 1e-9  # noqa: E731
            line = []
            for name in REFERENCE:
                fails = [r for r in slack if not ok(r, v[name][r])]
                passes_all[name] &= not fails
                line.append(f"{name} {'ok' if not fails else 'fails ' + '/'.join(fails)}")
            print(f"  k={k:<4} best " + ", ".join(f"{r} {v['best'][r]:.3f}" for r in slack) + " | " + "; ".join(line),
                  flush=True)
    print("passes every member: " + (", ".join(n for n, ok in passes_all.items() if ok) or "none"))


if __name__ == "__main__":
    main(*sys.argv[1:])
