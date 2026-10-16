"""Every round's candidate of every gridworld list (runs that resolve `any` in the loop): numpy values on the training
members and certified held-out targets. One JSON line per (run, sample)."""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
sys.path.insert(0, str(Path(__file__).parent))
import order_cache as oc
from order_cache import ROOT, TARGETS, evaluate, mdp_of

from config import load_run_config
from core.domain import load_domain
from results_io import sample_records
from tie_analysis import allowed
from transfer import parse_ids


def one(args):
    run, sid = args
    run = Path(run)
    cfg = load_run_config(run)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    instances = {int(i.id): i for i in domain.load_instances(cfg.domain.dataset)}
    rec = sample_records(run)[sid]
    trained = [int(m["instance"]) for m in rec["members"]]
    group = TARGETS[cfg.domain.dataset]
    targets = parse_ids(group[sid] if sid in group else group[None])
    out = {"run": run.relative_to(ROOT / "out/results").as_posix(), "sid": sid, "trained": trained,
           "targets": targets, "rounds": []}
    for it in rec["iterations"]:
        rules = [(r["condition"], r["action"]) for r in it["rules"]]
        entry = {"iteration": it["iteration"], "improved": it.get("improved"), "mode": it["mode"],
                 "num_rules": len(rules), "train": {}, "cert": None}
        try:
            for i in trained:
                mdp, mons = mdp_of(cfg.domain.dataset, instances[i])
                reqs = {r.name: r for r in domain.spec(instances[i]).requirements}
                v = evaluate(mdp, mons, allowed(mdp, rules), reqs, True)
                entry["train"][str(i)] = None if v is None else {n: [x, reqs[n].threshold, reqs[n].bound]
                                                                 for n, x in v.items()}
            cert = 0
            for t in targets:
                mdp, mons = mdp_of(cfg.domain.dataset, instances[t])
                reqs = {r.name: r for r in domain.spec(instances[t]).requirements}
                cert += bool(evaluate(mdp, mons, allowed(mdp, rules), reqs, False))
            entry["cert"] = cert
        except KeyError:
            entry["skipped"] = True
        out["rounds"].append(entry)
    return out


def main():
    out = Path(sys.argv[1])
    jobs = []
    runs = sorted(list((ROOT / "out/results/core").glob("*/seed*/config.json")) +
                  list((ROOT / "out/results/ablations").glob("*/seed*/config.json")))
    for cfgp in runs:
        c = json.loads(cfgp.read_text())
        if c.get("rules", {}).get("any_order", "none") not in ("all", "rotations"):
            continue
        if c["domain"]["dataset"] != "families_v0.csv" or "haiku" not in c["llm"]["model"] or "chains" in cfgp.parent.parent.name:
            continue
        for sid, rec in sample_records(cfgp.parent).items():
            if rec.get("iterations") and "members" in rec:
                jobs.append((str(cfgp.parent), sid))
    print(len(jobs), "lists", flush=True)
    with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 3) as pool, open(out, "w") as f:
        for k, d in enumerate(pool.map(one, jobs, chunksize=1)):
            f.write(json.dumps(d) + "\n")
            f.flush()
            if k % 20 == 0:
                print(k, d["run"], flush=True)


if __name__ == "__main__":
    main()
