"""How much do the action orders of a run's rule lists decide? Gridworld only, on the numpy model
(src/baselines/gridworld_mdp.py, checked against PRISM), so it takes seconds per list and target.

For each final rule list of the runs (a run of rules with one condition and every action is read as `c -> any`,
core.rules.open_ties) and each target:
  * how many actions the deciding rule allows, over the states reachable when any allowed action may be taken;
  * whether the list passes with every choice left open, taking the worst allowed action (what
    `transfer.py --rules open` certifies with PRISM) and the best one per state (an upper bound: a solver choosing
    per state, which the method never does);
  * how many targets it passes under each of the 24 fixed action orders.

Usage (from PRISM-Guided-Learning/):
    python src/tie_analysis.py <run dir> ... [--targets 12-19] [--data <dataset>] [--out <file.md>]
Lists whose conditions are not conjunctions of feature literals and goal flags are reported as skipped.
"""
import argparse
import itertools
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from baselines.gridworld_mdp import ACTIONS, GridMDP
from config import load_run_config
from core.domain import load_domain
from core.rules import ANY, open_ties
from transfer import final_rules, parse_ids

Rules = List[Tuple[str, str]]


def literals(condition: str) -> Dict[str, bool]:
    if condition.strip() == "true":
        return {}
    return {t.strip().lstrip("!").strip(): not t.strip().startswith("!") for t in condition.split("&")}


def allowed(mdp: GridMDP, rules: Rules) -> np.ndarray:
    """(S, 4) bool: the actions the deciding rule allows in each state (all where no rule decides)."""
    stage = np.array([s for (_, _, s, _) in mdp.states])
    out = np.zeros((mdp.S, 4), dtype=bool)
    decided = np.zeros(mdp.S, dtype=bool)
    for cond, action in rules:
        ok = np.ones((mdp.S, 4), dtype=bool)
        for name, want in literals(cond).items():
            if name[0] == "g" and name[1:].isdigit():            # goal flag g_k: the first k goals are reached
                ok &= ((stage >= int(name[1:])) == want)[:, None]
            else:
                ok &= mdp.features[name] == want
        if action != ANY:
            ok &= np.arange(4) == ACTIONS.index(action)
        fire = ~decided & ok.any(axis=1)
        out[fire] = ok[fire]
        decided |= fire
    out[~decided] = True
    return out


def reachable(mdp: GridMDP, allow: np.ndarray) -> np.ndarray:
    seen = np.zeros(mdp.S, dtype=bool)
    seen[mdp.initial] = True
    frontier = [mdp.initial]
    while frontier:
        nxt = {int(j) for i in frontier for a in np.flatnonzero(allow[i]) for j in mdp.succ[i, a]} - \
              set(np.flatnonzero(seen))
        seen[list(nxt)] = True
        frontier = list(nxt)
    return seen


def extreme(mdp: GridMDP, allow: np.ndarray, mon, target: np.ndarray, minimize: bool, tol: float = 1e-12) -> float:
    """Min or max probability of reaching `target` on the product with the monitor, by value iteration from 0."""
    Q, S = mon.step.shape
    fixed = np.repeat(mon.accept | mon.reject, S)
    goal = np.repeat(target, S).astype(float)
    idx = (mon.step[:, mdp.succ] * S + mdp.succ[None]).reshape(Q * S, 4, 3)
    blocked = np.tile(~allow, (Q, 1))
    x = goal.copy()
    for _ in range(200000):
        vals = np.where(blocked, np.inf if minimize else -np.inf, (mdp.prob[None] * x[idx]).sum(axis=2))
        new = np.where(fixed, goal, vals.min(axis=1) if minimize else vals.max(axis=1))
        done = np.abs(new - x).max() < tol
        x = new
        if done:
            break
    return float(x[mon.step[0, mdp.initial] * S + mdp.initial])


def worst_best(mdp: GridMDP, allow: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
    worst, best = {}, {}
    for name, (mon, safety) in mdp.monitors().items():
        if safety:                                                # P(G !reject) = 1 - P(F reject)
            worst[name] = 1 - extreme(mdp, allow, mon, mon.reject, minimize=False)
            best[name] = 1 - extreme(mdp, allow, mon, mon.reject, minimize=True)
        else:
            worst[name] = extreme(mdp, allow, mon, mon.accept, minimize=True)
            best[name] = extreme(mdp, allow, mon, mon.accept, minimize=False)
    return worst, best


def analyse(rules: Rules, targets, domain) -> Optional[dict]:
    """The numbers of one list over `targets` (instances), or None if it is not a list of feature literals."""
    rules = open_ties(rules, ACTIONS)
    sizes, worst_pass, best_pass, per_order = Counter(), 0, 0, Counter()
    orders = list(itertools.permutations(ACTIONS))
    for target in targets:
        mdp = GridMDP(target.data)
        reqs = domain.spec(target).requirements
        try:
            allow = allowed(mdp, rules)
        except KeyError:
            return None
        stage = np.array([s for (_, _, s, _) in mdp.states])
        live = reachable(mdp, allow) & (stage < mdp.K)
        sizes.update(allow[live].sum(axis=1).tolist())
        worst, best = worst_best(mdp, allow)
        worst_pass += all(r.satisfied(worst[r.name]) for r in reqs)
        best_pass += all(r.satisfied(best[r.name]) for r in reqs)
        monitors = mdp.monitors()
        for order in orders:
            values = mdp.evaluate(allowed(mdp, [(c, a) for c, act in rules
                                                for a in (order if act == ANY else [act])]).argmax(axis=1), monitors)
            per_order[order] += all(r.satisfied(values[r.name]) for r in reqs)
    total = sum(sizes.values()) or 1
    counts = sorted(per_order.values())
    return {"one": sizes[1] / total, "two": sizes[2] / total, "more": (sizes[3] + sizes[4]) / total,
            "worst": worst_pass, "best": best_pass, "order_min": counts[0], "order_median": counts[len(counts) // 2],
            "order_max": counts[-1]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories (gridworld)")
    parser.add_argument("--targets", default="12-19", help="Target ids")
    parser.add_argument("--data", default=None, help="Target dataset (default: each run's own)")
    parser.add_argument("--out", type=Path, default=None, help="Also write the table to this markdown file")
    args = parser.parse_args(argv)
    lines = [f"Targets {args.targets}. Allowed: share of reachable decision states where the deciding rule allows 1 / "
             "2 / 3-4 actions. Open, worst / best: targets passed with every choice left open, taking the worst / "
             "best allowed action per state. Orders: targets passed under each of the 24 fixed orders.", "",
             "| run | list | allowed 1 / 2 / 3-4 | open, worst | open, best | orders min / median / max |",
             "|---|---|---|---|---|---|"]
    for run in args.runs:
        cfg = load_run_config(run)
        domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
        wanted = set(parse_ids(args.targets))
        targets = [t for t in domain.load_instances(args.data or cfg.domain.dataset) if int(t.id) in wanted]
        for source, rules in final_rules(run).items():
            n = analyse(rules, targets, domain)
            name = f"{run.parent.name}/{run.name}"
            if n is None:
                lines.append(f"| {name} | {source} | skipped (not feature literals) | | | |")
                continue
            lines.append(f"| {name} | {source} | {n['one']:.0%} / {n['two']:.0%} / {n['more']:.0%} | {n['worst']} | "
                         f"{n['best']} | {n['order_min']} / {n['order_median']} / {n['order_max']} |")
            print(lines[-1], flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(args.out)


if __name__ == "__main__":
    main()
