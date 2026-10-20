"""Item 1 of Rajan's SEAMS checklist: synthesis time against certification time, per instance, on one machine.

Synthesis (a lower bound: the query decides whether a strategy exists without exporting one):
  * grid worlds: the multi-objective query `src/ceilings.py` runs (`PolicyVerifier.joint_query`, PRISM's sparse engine
    and exact LP, `prism.multi_engine` / `prism.multi_method`) on the bare MDP of every instance of the RQ2 pool: the
    20 grids of grid_20_large (9x9-13x13) and the validation family members (10x10-14x14);
  * Pac-Man: the single `Pmin` query on the unrestricted (bare) model at every validation length (8, 10, 12, 20, 25,
    30), with the exact solvers the certification uses (`prism.exact_methods` in turn: interval iteration at
    `prism.exact_epsilon`, then linear programming).
Certification, on the same instances in the same session, under the certification rule (`src/recertify.py`'s
`certify_rules`, unchanged): the three lists of `core/port_fam_history/seed_none_1` (frozen: `any` resolved,
determinized.json) on every grid-world instance, and the three core Pac-Man lists (`core/port_pacman_narrow_core`
seed_none_1-3, narrowed, as kept) on every Pac-Man length.

One check at a time (no other PRISM process from this script), so the times are not contended. Per check: states,
model-construction time (PRISM's "Time for model construction"), check time (the call's wall time minus construction:
it includes the JVM's start and, for lists, composing the policy module), outcome (done | timeout | out_of_memory |
error; lists: the certification verdict too) and, for lists, Asher's verdict for the same list and instance where his
checks have one (`asher_outcome`).

Usage (from PRISM-Guided-Learning/):
    python src/synthesis_time.py [--only families grids pacman] [--limit N] [--out out/paper/tables/tab_rq2-synthesis-time.csv]
(The three parts can run as separate jobs, each alone on its own cores, into separate --out files; the paper table is
their concatenation.)
Rows already in the output are skipped, so an interrupted run resumes with the same command. Validation instances
only: the sealed guard of src/transfer.py refuses anything else.
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import load_run_config
from core.domain import load_domain
from core.prism import PrismCall, PrismRunner, run_call
from core.verifier import PolicyVerifier, exact_solvers
from recertify import certify_rules
from settings import PROJECT_ROOT
from transfer import determinized_rules, final_rules, guard_sealed

RESULTS = PROJECT_ROOT / "out" / "results"
FAM_RUN = RESULTS / "core" / "port_fam_history" / "seed_none_1"
PAC_RUNS = [RESULTS / "core" / "port_pacman_narrow_core" / f"seed_none_{k}" for k in (1, 2, 3)]
FAMILY_IDS = list(range(16, 36)) + list(range(76, 96)) + list(range(136, 156))   # validation members, 10x10-14x14
GRID_IDS = list(range(20))                                                       # grid_20_large: all 20 (9x9-13x13)
PACMAN_IDS = [0, 1, 2, 6, 7, 9]                                                  # lengths 8, 10, 12, 20, 25, 30
COLUMNS = ["instance", "domain", "size", "method", "states", "construction_s", "check_s", "outcome"]

STATES = re.compile(r"^States:\s+(\d+)", re.M)
CONSTRUCTION = re.compile(r"Time for model construction:\s*([\d.]+)")
CHECKING = re.compile(r"Time for model checking:\s*([\d.]+)")
RESULT = re.compile(r"^Result: (\S+)", re.M)


def outcome_of(stdout: str) -> str:
    if "OutOfMemory" in stdout or "out of memory" in stdout.lower():
        return "out_of_memory"
    if "Error" in stdout or not RESULT.search(stdout):
        return "error"
    return "done"


def timed_call(cmd: List[str], timeout: float) -> Dict:
    """One PRISM process, run here and timed: its output fields, or the timeout."""
    start = time.time()
    try:
        out = run_call(PrismCall(tuple(cmd), timeout, None))
    except subprocess.TimeoutExpired:
        return {"outcome": "timeout", "wall_s": time.time() - start}
    wall = time.time() - start
    construction = sum(float(x) for x in CONSTRUCTION.findall(out))
    states = STATES.search(out)
    result = RESULT.search(out)
    return {"outcome": outcome_of(out), "wall_s": wall, "construction_s": construction,
            "check_s": wall - construction, "prism_check_s": sum(float(x) for x in CHECKING.findall(out)),
            "states": int(states.group(1)) if states else None, "result": result.group(1) if result else None,
            "error": None if outcome_of(out) == "done" else out.strip().splitlines()[-1][:200] if out.strip() else ""}


def synthesis_grid(verifier: PolicyVerifier, runner: PrismRunner) -> Dict:
    """`ceilings.py`'s joint query on the bare MDP (as `PrismRunner.check_steps` runs it), timed."""
    with tempfile.TemporaryDirectory(prefix="synth_") as tmp:
        paths = runner._write_inputs(tmp, verifier.model, [verifier.joint_query()])
        return timed_call(runner._command(*paths, runner.multi_args), runner.timeout)


def synthesis_pmin(verifier: PolicyVerifier, runner: PrismRunner, prism_cfg) -> Dict:
    """The single best-case query (Pmin for Pac-Man's crash bound) on the bare model, with the certification's exact
    solvers in turn; the first that finishes is reported."""
    props = [f"{r.best_op()}=? [ {r.formula} ]" for r in verifier.spec.requirements]
    exact = replace(prism_cfg, method="", max_iters=prism_cfg.exact_max_iters)
    exact_runner = PrismRunner(exact, prism_path=runner.prism_path)
    tries = []
    with tempfile.TemporaryDirectory(prefix="synth_") as tmp:
        paths = exact_runner._write_inputs(tmp, verifier.model, props)
        for name, args in exact_solvers(prism_cfg)[:-1]:          # sound solvers only, as the certification rule
            row = timed_call(exact_runner._command(*paths, ["-explicit"]) + args, exact_runner.timeout)
            tries.append(f"{name}: {row['outcome']}")
            if row["outcome"] == "done":
                return {**row, "solver": name, "tries": " | ".join(tries)}
        return {**row, "solver": None, "tries": " | ".join(tries)}


def asher_verdicts() -> Dict[tuple, str]:
    """(list id, domain, instance) -> Asher's outcome for the same frozen list and instance, from his checks under the
    same rule: the families run's transfer_exact_*_determinized.csv, the Pac-Man runs' transfer_exact_*.csv, and the
    shift_v0 frozen sequences' checks.csv for the random grids (sequential_v0 ids 180-199 = grid_20_large 0-19)."""
    out = {}
    fam = FAM_RUN / "transfer_exact_families_v0_determinized.csv"
    if fam.exists():
        for r in pd.read_csv(fam).itertuples():
            out[(f"fam1_list{r.source}", "families", int(r.target))] = r.outcome
    for k, run in enumerate(PAC_RUNS, 1):
        p = run / "transfer_exact_pacman_transfer.csv"
        if p.exists():
            for r in pd.read_csv(p).itertuples():
                out[(f"pacman{k}_list{r.source}", "pacman", int(r.target))] = r.outcome
    for p in (RESULTS / "sequential" / "shift_v0" / "frozen").glob("*/checks.csv"):
        for r in pd.read_csv(p).itertuples():
            if str(r.rule_list).startswith("seed") and 180 <= int(r.instance) <= 199:
                out[(f"fam1_list{str(r.rule_list)[4:]}", "grids", int(r.instance) - 180)] = r.outcome
    return out


def hardware() -> Dict:
    info = {"host": platform.node(), "os": platform.platform(), "python": platform.python_version(),
            "cpus_visible": os.cpu_count(), "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
            "slurm_job": os.environ.get("SLURM_JOB_ID")}
    try:
        cpu = [l for l in open("/proc/cpuinfo") if l.startswith("model name")]
        info["cpu"] = cpu[0].split(":", 1)[1].strip() if cpu else None
        mem = [l for l in open("/proc/meminfo") if l.startswith("MemTotal")]
        info["ram_gb"] = round(int(mem[0].split()[1]) / 2**20, 1) if mem else None
    except OSError:
        pass
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="+", choices=["families", "grids", "pacman"], default=["families", "grids", "pacman"])
    p.add_argument("--limit", type=int, default=None, help="at most N instances per dataset (a smoke test)")
    p.add_argument("--java-max-mem", default=None, help="PRISM's Java heap (default: the run's, 4g); Pac-Man needs 8g "
                                                        "to build the bare model at length 30")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "out" / "paper" / "tables" / "tab_rq2-synthesis-time.csv")
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.out.exists():
        old = pd.read_csv(a.out)
        done = {(r.domain, int(r.instance), str(r.method)) for r in old.itertuples()}
    meta = {**hardware(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "argv": sys.argv[1:],
            "java_max_mem": a.java_max_mem or "the run's (4g)"}
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    asher = asher_verdicts()

    jobs = []   # (domain, dataset, ids, run config dir, lists)
    fam_lists = {f"fam1_list{s}": r for s, r in determinized_rules(FAM_RUN).items()}
    if "families" in a.only:
        jobs += [("families", "families_v0.csv", FAMILY_IDS, FAM_RUN, fam_lists)]
    if "grids" in a.only:
        jobs += [("grids", "grid_20_large.csv", GRID_IDS, FAM_RUN, fam_lists)]
    if "pacman" in a.only:
        pac_lists = {f"pacman{k}_list{s}": r for k, run in enumerate(PAC_RUNS, 1) for s, r in final_rules(run).items()}
        jobs += [("pacman", "pacman_transfer.csv", PACMAN_IDS, PAC_RUNS[0], pac_lists)]

    for domain_name, dataset, ids, run_dir, lists in jobs:
        cfg = load_run_config(run_dir)
        cfg = replace(cfg, domain=replace(cfg.domain, dataset=dataset))
        if a.java_max_mem:
            cfg = replace(cfg, prism=replace(cfg.prism, java_max_mem=a.java_max_mem))
        if cfg.rules.early_narrow:          # narrowings name places with numbers (as transfer.py, recertify.py)
            cfg = replace(cfg, rules=replace(cfg.rules, shared_literals=True))
        domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
        guard_sealed(dataset, ids)
        instances = {int(i.id): i for i in domain.load_instances(dataset)}
        runner = PrismRunner(cfg.prism)
        for iid in ids[:a.limit] if a.limit else ids:
            inst = instances[iid]
            size = inst.data.get("n", inst.data.get("max_steps"))
            base = {"instance": iid, "domain": domain_name, "size": size, "dataset": dataset, "split": "validation"}
            rows = []
            if (domain_name, iid, "synthesis") not in done:
                t0 = time.time()
                verifier = PolicyVerifier(domain, inst, cfg, runner)
                render = time.time() - t0
                r = synthesis_pmin(verifier, runner, cfg.prism) if domain_name == "pacman" \
                    else synthesis_grid(verifier, runner)
                rows.append({**base, "method": "synthesis", **r, "render_s": render,
                             "query": "Pmin (exact solvers)" if domain_name == "pacman" else "multi(...) sparse LP"})
            todo = {k: v for k, v in lists.items() if (domain_name, iid, k) not in done}
            for c in certify_rules(domain, cfg, inst, todo) if todo else []:      # one list at a time, in order
                seconds, model_s = c.get("seconds"), c.get("model_seconds")
                rows.append({**base, "method": c["source"], "states": c.get("states"), "construction_s": model_s,
                             "check_s": seconds - model_s if seconds is not None and model_s is not None else None,
                             "wall_s": seconds,
                             "outcome": "done" if c["outcome"] in ("certified", "failed", "inconclusive")
                             else c["outcome"], "verdict": c["outcome"], "solver": c.get("solver"),
                             "margin": c.get("margin"), "binding": c.get("binding"), "error": c.get("error"),
                             "asher_outcome": asher.get((c["source"], domain_name, iid))})
            for r in rows:
                agree = "" if not r.get("asher_outcome") else \
                    (" (Asher: same)" if r["asher_outcome"] == r.get("verdict") else f" (Asher: {r['asher_outcome']}!)")
                print(f"{domain_name} {iid} (size {size}) {r['method']}: {r.get('outcome')} {r.get('verdict') or r.get('result') or ''}"
                      f" states={r.get('states')} construction={r.get('construction_s')} check={r.get('check_s')}{agree}",
                      flush=True)
            if rows:
                new = pd.DataFrame(rows)
                old = pd.read_csv(a.out) if a.out.exists() else pd.DataFrame()
                df = pd.concat([old, new], ignore_index=True)
                df = df[COLUMNS + [c for c in df.columns if c not in COLUMNS]]
                df.to_csv(a.out, index=False)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    a.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    df = pd.read_csv(a.out)
    diff = df[df.asher_outcome.notna() & (df.asher_outcome != df.verdict)] if "asher_outcome" in df else df.iloc[:0]
    print(f"\n{a.out}: {len(df)} rows; outcomes {df.outcome.value_counts().to_dict()}; "
          f"verdicts differing from Asher's: {len(diff)}")
    if len(diff):
        print(diff[["instance", "domain", "method", "verdict", "asher_outcome", "margin"]].to_string(index=False))


if __name__ == "__main__":
    main()
