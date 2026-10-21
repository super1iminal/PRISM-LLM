"""Inputs for Storm (Oct 21): for each check in <checks csv> (file, source, target, episode), PRISM exports the
composed model it checks (as direct_recheck.py; per-state policies: the explicit model under the policy), and each
requirement becomes its monitor product (ltl_exact.product_model: PRISM language, rational probabilities) with one
reachability query. Writes <out dir>/<check>/<requirement>.prism and .props, and <out dir>/manifest.json.

Usage (from PRISM-Guided-Learning/): python storm_prep.py <checks csv> <out dir> [--workers 6]"""
import argparse, json, sys, tempfile, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import exact_recheck as E
from core.prism import PrismCall, PrismRunner, run_call
from core.verifier import PolicyVerifier
from ltl_exact import product_model, read_explicit

LOCK = threading.Lock()


def export(domain, cfg, instance, controller, source, tmp: Path):
    """(spec, .tra, .lab) of the model the check certifies."""
    if isinstance(controller, tuple):
        from baselines.explicit_mdp import ExplicitMDP
        import re
        mdp = ExplicitMDP(domain, instance, cfg)
        m = mdp.under(controller[1].policies(mdp)[source])
        m = m.part(m.reachable())
        names = sorted({n for r in mdp.spec.requirements for n in re.findall(r'"([^"]+)"', r.formula)})
        m.write(tmp / "model", names)
        return mdp.spec, tmp / "model.tra", tmp / "model.lab"
    runner = PrismRunner(cfg.prism)
    v = PolicyVerifier(domain, instance, cfg, runner)
    spec = v.spec
    (tmp / "m.prism").write_text(v.compose(spec.policy(controller(int(instance.id))[source],
                                                        spec.vocabulary(cfg.rules))), encoding="utf-8")
    log = run_call(PrismCall((runner.prism_path, str(tmp / "m.prism"), "-exportmodel", str(tmp / "out.tra,lab"),
                              "-exportmodelprecision", "17"), 1800, cfg.prism.max_concurrent))
    if not (tmp / "out.tra").exists():
        raise RuntimeError("export failed: " + log[-300:])
    return spec, tmp / "out.tra", tmp / "out.lab"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checks"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    rows = pd.read_csv(a.checks)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    contexts, locks = {}, {f: threading.Lock() for f in rows.file.unique()}
    manifest = {}

    def ctx(file):
        with locks[file]:
            if file not in contexts:
                contexts[file] = E.context(file)
            return contexts[file]

    def one(item):
        k, row = item
        cid = f"c{k:03d}"
        domain, cfg, inst, controller = ctx(row.file)
        with tempfile.TemporaryDirectory(prefix="storm_") as tmp:
            spec, tra, lab = export(domain, cfg, inst[int(row.target)], controller, str(row.source), Path(tmp))
            choices, init, labels = read_explicit(tra, lab)
        d = out / cid; d.mkdir(exist_ok=True)
        reqs = {}
        for r in spec.requirements:
            text, props, finish, sizes = product_model(choices, init, labels, [r])
            entry = {"threshold": r.threshold, "bound": r.bound, "safety": None, "fixed": None}
            if not props[0]:
                entry["fixed"] = float(finish[0])          # the monitor cannot decide it: a constant
            else:
                (d / f"{r.name}.prism").write_text(text, encoding="utf-8")
                (d / f"{r.name}.props").write_text(props[0], encoding="utf-8")
                entry["safety"] = finish[0](0) == 1         # safety: value = 1 - P[F violation]
            reqs[r.name] = entry
        with LOCK:
            manifest[cid] = {"file": row.file, "source": str(row.source), "target": int(row.target),
                             "episode": None if pd.isna(row.episode) else int(row.episode),
                             "states": len(choices), "requirements": reqs}
        print(cid, row.file, row.source, row.target, len(choices), flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(one, rows.iterrows()))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
