"""Regression: legacy per-state policies, re-expressed as symbolic rules, must verify identically.

For every policy the legacy run verified (every sample, every iteration), translate it to atomic
rules, compose it with the gridworld domain's MDP, and check with PRISM:

  1. stored:  new best/worst (default PRISM settings, like the legacy run) vs the probabilities
              the legacy run reported;
  2. exact:   new best/worst vs the legacy DTMC recomputed for the same policy, both solved to a
              tight convergence epsilon. This isolates model equivalence from solver tolerance;
  3. coverage: best == worst and no uncovered situations (the translation is a complete policy).

Usage: python src/regression.py --legacy-run out/results/legacy_grid20 [--data grid_20_balanced.csv]
"""
import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from core.domain import load_domain
from core.prism import PrismRunner
from core.rules import SymbolicPolicy
from core.verifier import PolicyVerifier
from legacy.gridworld import GridWorld as LegacyGridWorld
from legacy.prism_model import PrismModelGenerator
from legacy.requirements import SimplifiedVerifier
from settings import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))
from domains.gridworld.legacy_translate import legacy_policy_to_rules  # noqa: E402

_quiet = logging.getLogger("regression")
_quiet.addHandler(logging.NullHandler())
_quiet.propagate = False


def legacy_dtmc_values(instance, legacy_policy, runner: PrismRunner):
    """Recompute the legacy DTMC's requirement values for a policy with `runner`'s settings."""
    d = instance.data
    gw = LegacyGridWorld(d["n"], d["goals"], d["static"], d["moving"])
    policy = {tuple(state): int(action) for state, action in legacy_policy}
    model = PrismModelGenerator(gw, _quiet).generate_prism_model(policy)
    requirements = SimplifiedVerifier(None, gw, _quiet).requirements
    return runner.run(model, [r.property for r in requirements]).initial_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-run", required=True)
    parser.add_argument("--data", default="grid_20_balanced.csv")
    parser.add_argument("--epsilon", default="1e-10", help="Tight PRISM convergence epsilon for the exact check")
    parser.add_argument("--tol-stored", type=float, default=1e-4,
                        help="Solver-tolerance check: legacy values were computed at PRISM's default epsilon")
    parser.add_argument("--tol-exact", type=float, default=1e-7)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run_dir = Path(args.legacy_run)
    domain = load_domain("gridworld")
    instances = domain.load_instances(args.data)
    default_runner = PrismRunner()
    tight_runner = PrismRunner(extra_args=["-epsilon", args.epsilon])

    jobs = []
    for path in sorted((run_dir / "outputs").glob("sample_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        instance = instances[record["sample_id"]]
        for it, (policy, probs) in enumerate(zip(record["iteration_policies"], record["iteration_prism_probs"]), 1):
            jobs.append((record["sample_id"], it, instance, policy, probs))

    def check(job):
        sample_id, iteration, instance, legacy_policy, stored = job
        rules = legacy_policy_to_rules(legacy_policy, len(instance.data["goals"]))
        verifier = PolicyVerifier(domain, instance, default_runner)
        policy = SymbolicPolicy.from_raw(verifier.spec.variables, list(verifier.spec.actions), rules)
        v = verifier.verify(policy, analysis=False)
        v_tight = PolicyVerifier(domain, instance, tight_runner).verify(policy, analysis=False)
        names = [r.name for r in verifier.spec.requirements]
        exact_legacy = dict(zip(names, legacy_dtmc_values(instance, legacy_policy, tight_runner)))
        rows = []
        for name in names:
            rows.append({
                "sample_id": sample_id, "iteration": iteration, "requirement": name, "num_rules": len(rules),
                "uncovered_situations": v.uncovered_situations,
                "stored_legacy": stored[name], "best": v.best[name], "worst": v.worst[name],
                "exact_legacy": exact_legacy[name], "exact_best": v_tight.best[name], "exact_worst": v_tight.worst[name],
                "diff_stored": max(abs(v.best[name] - stored[name]), abs(v.worst[name] - stored[name])),
                "diff_exact": max(abs(v_tight.best[name] - exact_legacy[name]),
                                  abs(v_tight.worst[name] - exact_legacy[name])),
                "best_minus_worst_exact": abs(v_tight.best[name] - v_tight.worst[name]),
            })
        return rows

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for rows in pool.map(check, jobs) for row in rows]

    df = pd.DataFrame(rows)
    out_dir = run_dir / "regression"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_dir / "regression.csv", index=False)

    stored_fail = df[df.diff_stored > args.tol_stored]
    exact_fail = df[df.diff_exact > args.tol_exact]
    uncovered = int(df.uncovered_situations.max())
    passed = len(stored_fail) == 0 and len(exact_fail) == 0 and uncovered == 0
    summary = "\n".join([
        "# Regression: legacy policies as symbolic rules",
        "",
        f"- Legacy run: `{run_dir}`",
        f"- Policies checked: {len(jobs)} (all iterations of {df.sample_id.nunique()} samples); "
        f"requirement values: {len(df)}",
        f"- Uncovered situations in translated policies (max): {uncovered}",
        f"- **Stored check** (default PRISM settings vs the legacy run's reported values): "
        f"max diff {df.diff_stored.max():.2e}, {len(stored_fail)} above {args.tol_stored}",
        f"- **Exact check** (legacy DTMC vs new MDP, both at epsilon {args.epsilon}): "
        f"max diff {df.diff_exact.max():.2e}, {len(exact_fail)} above {args.tol_exact}",
        f"- Max |best - worst| at epsilon {args.epsilon}: {df.best_minus_worst_exact.max():.2e}",
        "",
        f"**{'PASS' if passed else 'FAIL'}**",
    ])
    for title, failures in (("Stored check failures", stored_fail), ("Exact check failures", exact_fail)):
        if len(failures):
            summary += f"\n\n## {title}\n\n```\n{failures.to_string(index=False)}\n```"
    (out_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
