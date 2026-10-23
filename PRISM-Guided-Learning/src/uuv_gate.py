"""The PRISM gate of the UUV family (Rajan's checklist, item 8), before any LLM run. On every member of
`uuv_family_v0.csv`, certify under rule 2 (exact solvers, `core.verifier.exact_verdict`):

- the reference controllers `always_low`, `always_med`, `always_high`, `stay` (domains/uuv/data/reference_policies.json)
  and the uniform random controller: each must fail on some member, or the family is trivial;
- the hand-written list `HAND` (one list in the extended vocabulary; its constants take each member's values): it
  must be certified on every member, so that transfer is possible. It is also each member's witness of joint
  solvability (checkpoint 2: a controller certified to meet every threshold at once).

Uniform random controller: a variable `pick`, redrawn uniformly on every transition, decides which altitude command
may fire, so the controller picks low, med or high with probability 1/3 at every decision. The first decision comes
before any redraw, so its value is the mean of the three runs that start with pick = 0, 1, 2 (every requirement is
linear in the initial distribution). Its "certified" means the mean of each requirement passes with the band.

Usage (from PRISM-Guided-Learning/):
    python src/uuv_gate.py [--data uuv_family_v0.csv] [--targets 0-29] [--out out/paper/rajan/item8/gate.csv]
    python src/uuv_gate.py --targets 30-41 --unseal --out out/paper/rajan/item8/gate_sealed.csv   (the test pass only)
Sealed members (configs/transfer/default.yaml) are refused without --unseal.
"""
import argparse
from pathlib import Path

import pandas as pd

from config import load_config
from core.domain import load_domain
from core.verifier import PolicyVerifier, exact_verdict
from settings import PROJECT_ROOT
from transfer import guard_sealed, parse_ids

START = "s = 11 | s = 0"     # start_task, start_search (STATES in domains/uuv/domain.py)
# Search at med; once forced down to low, stay low until the visibility is one level above the poor band; keep high.
HAND = [(f"alt = 0 & !({START}) & water_visib < {{FIRST_AVG}} + 1", "low"), ("alt = 2", "high"), ("true", "med")]


def first_levels(data):
    """The first visibility level of the average band and of the good band: the model's med_visib = W/3 and
    high_visib = 2W/3 (real division), rounded up to a level."""
    w = data["max_visib"] - data["min_visib"]
    return -(-w // 3), -(-2 * w // 3)


def instantiate(rules, data):
    a, g = first_levels(data)
    return [(c.replace("{FIRST_AVG} + 1", str(a + 1)).replace("{FIRST_AVG}", str(a)).replace("{FIRST_GOOD}", str(g)), x)
            for c, x in rules]


def uniform_model(model: str, first: int) -> str:
    labels = ("step", "low", "med", "high")
    draw = " + ".join(f"1/3:(pick'={k})" for k in range(3))
    return (model + f"\nmodule pickmod\n  pick : [0..2] init {first};\n"
            + "".join(f"  [{a}] true -> {draw};\n" for a in labels) + "endmodule\n"
            + "\nmodule uniform\n  [low] pick=0 -> true;\n  [med] pick=1 -> true;\n  [high] pick=2 -> true;\nendmodule\n")


def verdict(v, worst, solver, cfg):
    margins = {r.name: r.margin(worst[r.name]) for r in v.spec.requirements}
    return exact_verdict(margins, solver, cfg.prism), min(margins.values()), min(margins, key=margins.get)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default="uuv_family_v0.csv")
    p.add_argument("--targets", default="0-29", help="Member ids (default: the training and validation members)")
    p.add_argument("--unseal", action="store_true", help="Allow sealed test members (the test pass only)")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "out" / "paper" / "rajan" / "item8" / "gate.csv")
    a = p.parse_args()
    ids = parse_ids(a.targets)
    guard_sealed(a.data, ids, a.unseal)
    cfg = load_config("U1", [f"domain.dataset={a.data}"])
    domain = load_domain("uuv")
    refs = domain.reference_policies()
    rows = []
    for inst in (i for i in domain.load_instances(a.data) if int(i.id) in ids):
        v = PolicyVerifier(domain, inst, cfg)
        lists = {**refs, "hand": instantiate(HAND, inst.data)}
        for name, rules in lists.items():
            res, solver = v.verify_exact(v.spec.policy(rules))
            out, margin, binding = verdict(v, res.worst, solver, cfg) if res else ("not_certifiable", None, None)
            rows.append({"member": inst.data["name"], "instance": int(inst.id), "controller": name, "verdict": out,
                         "margin": margin, "binding": binding, "solver": solver,
                         **{f"worst_{k}": x for k, x in (res.worst.items() if res else [])}})
        # uniform: mean over the first pick
        base = v.model
        vals, solvers = [], set()
        for k in range(3):
            v.model = uniform_model(base, k)
            v._forced = None
            res, solver = v.verify_exact(v.empty_policy())
            vals.append(res.worst)
            solvers.add(solver)
        v.model, v._forced = base, None
        mean = {r.name: sum(x[r.name] for x in vals) / 3 for r in v.spec.requirements}
        solver = solvers.pop() if len(solvers) == 1 else "mixed"
        out, margin, binding = verdict(v, mean, solver, cfg)
        rows.append({"member": inst.data["name"], "instance": int(inst.id), "controller": "uniform", "verdict": out,
                     "margin": margin, "binding": binding, "solver": solver, **{f"worst_{k}": x for k, x in mean.items()}})
        print(inst.data["name"], {r["controller"]: r["verdict"][:4] for r in rows if r["member"] == inst.data["name"]},
              flush=True)
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    summary = df.pivot_table(index="controller", columns="verdict", values="member", aggfunc="count", fill_value=0)
    print(summary.to_string())
    print(a.out)


if __name__ == "__main__":
    main()
