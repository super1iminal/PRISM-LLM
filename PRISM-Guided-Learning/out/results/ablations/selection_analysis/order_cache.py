"""Per gridworld list (final rules of runs that resolve `any` in the loop): the values of every action order on its
training members, and whether it certifies each validation target, on the numpy model. One JSON line per list."""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
ROOT = Path(__file__).resolve().parents[4]          # PRISM-Guided-Learning/
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np

from baselines.gridworld_mdp import ACTIONS, GridMDP, RuleScorer, _reach
from config import load_run_config
from core.domain import load_domain
from core.rules import ANY, action_orders, open_ties
from tie_analysis import allowed
from transfer import final_rules, parse_ids, training_instances

TARGETS = {"grid_20_large.csv": {None: "12-19"},
           "families_v0.csv": {0: "16-35", 1: "76-95", 2: "136-155"}}
FAIL_FIRST = ["avoid_moving_seg1", "seq_1_before_2", "avoid_moving_seg2", "avoid_moving_seg3", "seq_2_before_3",
              "complete_sequence", "goal1", "goal2", "goal3"]
_MDPS = {}


def mdp_of(dataset, inst):
    key = (dataset, inst.id)
    if key not in _MDPS:
        mdp = GridMDP(inst.data)
        _MDPS[key] = (mdp, mdp.monitors())
    return _MDPS[key]


def reach_set(mdp, allow):
    seen = np.zeros(mdp.S, dtype=bool)
    seen[mdp.initial] = True
    frontier = np.array([mdp.initial])
    while frontier.size:
        nxt = mdp.succ[frontier][allow[frontier]].reshape(-1)
        frontier = np.unique(nxt[~seen[nxt]])
        seen[frontier] = True
    return seen


def evaluate(mdp, monitors, allow, reqs, full):
    """Values per requirement (full), or True/False whether every requirement holds (early exit); None if a
    reachable state is left open."""
    seen = reach_set(mdp, allow)
    if (allow[seen].sum(axis=1) > 1).any():
        return None
    actions = allow.argmax(axis=1)
    nxt = mdp.succ[np.arange(mdp.S), actions]
    p = mdp.prob[actions]
    names = sorted(monitors, key=lambda n: FAIL_FIRST.index(n) if n in FAIL_FIRST else 99)
    out = {}
    for name in names:
        mon, safety = monitors[name]
        out[name] = _reach(mon, nxt, p, mdp.initial, safety)
        if not full and not reqs[name].satisfied(out[name]):
            return False
    return out if full else True


def list_values(domain, cfg, inst, opened, final, orders, full):
    mdp, monitors = mdp_of(cfg.domain.dataset, inst)
    reqs = {r.name: r for r in domain.spec(inst).requirements}
    out, cache = [], {}
    try:
        allows = [allowed(mdp, [(c, a) for c, act in opened for a in (order if act == ANY else [act])])
                  for order in orders]
        fin_allow = allowed(mdp, final)
    except KeyError:
        spec = domain.spec(inst)
        scorer = RuleScorer(inst.data, spec.variables)
        vocab = spec.vocabulary(cfg.rules)
        allows, fin_allow = [], None
        pol = spec.policy(opened, vocab)
        table = []
        for state in scorer.valuations:
            i = pol.first_match(state)
            table.append(None if i is None else ([ACTIONS.index(a) for a in pol.allowed(state)],
                                                 pol.rules[i].action == ANY))
        for order in orders:
            rank = [ACTIONS.index(a) for a in order]
            allow = np.zeros((mdp.S, 4), dtype=bool)
            for i, entry in enumerate(table):
                if entry is None:
                    allow[i] = True
                elif entry[1]:
                    allow[i, next(a for a in rank if a in entry[0])] = True
                else:
                    allow[i, entry[0]] = True
            allows.append(allow)
        fpol = spec.policy(final, vocab)
        fin_allow = np.zeros((mdp.S, 4), dtype=bool)
        for i, state in enumerate(scorer.valuations):
            a = fpol.allowed(state)
            if a is None:
                fin_allow[i] = True
            else:
                fin_allow[i, [ACTIONS.index(x) for x in a]] = True
    for allow in allows:
        key = allow.tobytes()
        if key not in cache:
            cache[key] = evaluate(mdp, monitors, allow, reqs, full)
        out.append(cache[key])
    return out, evaluate(mdp, monitors, fin_allow, reqs, full)


def one(args):
    run, sid = args
    run = Path(run)
    cfg = load_run_config(run)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    instances = {int(i.id): i for i in domain.load_instances(cfg.domain.dataset)}
    final = final_rules(run)[sid]
    trained = training_instances(run)[sid]
    group = TARGETS[cfg.domain.dataset]
    targets = parse_ids(group[sid] if sid in group else group[None])
    opened = open_ties(final, ACTIONS)
    orders = action_orders(ACTIONS, "all")
    reqs = {str(i): [(r.name, r.threshold, r.bound) for r in domain.spec(instances[i]).requirements] for i in trained}
    vals, fins = {}, {}
    for i in trained + targets:
        vals[str(i)], fins[str(i)] = list_values(domain, cfg, instances[i], opened, final, orders, i in trained)
    return {"run": run.relative_to(ROOT / "out/results").as_posix(), "sid": sid,
            "condition": cfg.condition, "dataset": cfg.domain.dataset,
            "hidden_features": cfg.rules.hidden_features, "trained": trained, "targets": targets,
            "orders": ["-".join(o) for o in orders], "num_any": sum(a == ANY for _, a in opened),
            "num_rules": len(final), "reqs": reqs, "values": vals, "final": fins}


def priority(run: Path) -> int:
    s = run.as_posix()
    return 0 if "/core/" in s else 1


def main():
    out = Path(sys.argv[1])
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            d = json.loads(line)
            done.add((d["run"], d["sid"]))
    jobs = []
    runs = sorted(list((ROOT / "out/results/core").glob("*/seed*/config.json")) +
                  list((ROOT / "out/results/ablations").glob("*/seed*/config.json")), key=lambda p: priority(p))
    for cfgp in runs:
        c = json.loads(cfgp.read_text())
        if c.get("rules", {}).get("any_order", "none") not in ("all", "rotations"):
            continue
        if c["domain"]["dataset"] not in TARGETS or "haiku" not in c["llm"]["model"]:
            continue
        run = cfgp.parent
        for sid in final_rules(run):
            if (run.relative_to(ROOT / "out/results").as_posix(), sid) not in done:
                jobs.append((str(run), sid))
    print(len(jobs), "lists", flush=True)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    with ProcessPoolExecutor(workers) as pool, open(out, "a") as f:
        for k, d in enumerate(pool.map(one, jobs, chunksize=1)):
            f.write(json.dumps(d) + "\n")
            f.flush()
            if k % 10 == 0:
                print(k, d["run"], d["sid"], flush=True)


if __name__ == "__main__":
    main()
