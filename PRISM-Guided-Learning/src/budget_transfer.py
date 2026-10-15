"""Budget curves from finished runs (no LLM calls): after each round, the rule list keep-best holds, what the run had
spent by then, and how many held-out targets that list certifies, by PRISM through src/transfer.py's `certify`.
Feedback and restarts are compared on these curves at equal cost (configs/experiments/rq2_feedback.yaml): keep-best
makes every curve rise with the budget, so only the comparison at equal cost is evidence.

Per training set (a source: one sample of the run) and round k:
  * the kept list: the rules of the last improved iteration up to k. Iterations store their rules with `any`
    already resolved into the order the verifier chose, so this is the list keep-best holds after round k;
  * the cost so far: prompt and output tokens of rounds 1..k (re-asks after invalid answers included), and dollars
    at the experiment's prices;
  * solved: whether the kept list passes every training member in the worst case (the loop's values);
  * its certification on each target, cached by a hash of the list: most rounds keep the same list, and the final
    list's rows come from the run's transfer CSV when it has them. Other lists are re-checked with the exact solvers
    only when the values near a threshold decide the verdict (`certify(deciding_only=True)`): the same verdict as
    src/transfer.py on its own assumption that values outside the band are accurate, without the exact checks
    (minutes each) of lists that fail elsewhere anyway.
Round 1 is the LLM's first answer, before any feedback (`first_answer`): an "LLM alone" point. A set the loop
stopped early (solved) has no rows after its last round and keeps its last list at any larger budget.

Usage (from PRISM-Guided-Learning/):
    python src/budget_transfer.py configs/experiments/rq2_feedback.yaml [--cells "F,R 18"] [--max-parallel 2]
        [--workers 2]
    python src/budget_transfer.py <run dir> ... [--data <dataset>] [--targets 12-19] [--source-targets JSON]
        [--price 0.10 0.50] [--workers 2]
An experiment file checks every run of its cells on its benchmarks' targets (each family's list on its own family's
targets) at its prices. `--source-targets` maps a source to its own targets, e.g. '{"0": "16-35"}'.
Writes <run dir>/budget_transfer.csv: one row per source, round and target. Sealed targets are refused.
"""
import argparse
import hashlib
import itertools
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import load_file, load_run_config
from core.domain import load_domain
from core.prism import PrismRunner
from results_io import sample_records
from transfer import DEFAULT_CONFIG, TransferConfig, certify, csv_path, guard_sealed, parse_ids, training_instances

BUDGET_CSV = "budget_transfer.csv"
CHECK_COLUMNS = ["parsed", "certified", "met", "requirements", "error"]
COSTS = {"dollars": "dollars per training set", "output_tokens": "output tokens per training set",
         "prompt_tokens": "prompt tokens per training set"}


def rules_hash(rules: Sequence) -> str:
    """A short hash of a rule list ((condition, action) pairs, in order)."""
    return hashlib.sha1(json.dumps([list(r) for r in rules]).encode()).hexdigest()[:12]


def training_met(it: dict, instance, requirements_of: Callable[[int], list]) -> Tuple[int, int]:
    """Requirements an iteration's rules meet in the worst case, summed over the training members shown to the LLM
    (the sample's own instance when it has no members), and how many requirements that is: the training score."""
    members = [(m["instance"], m["worst"]) for m in it.get("members", []) if not m.get("hidden")] or \
              [(instance, it["worst"])]
    met = total = 0
    for iid, worst in members:
        reqs = requirements_of(int(iid))
        met += sum(r.satisfied(worst[r.name]) for r in reqs)
        total += len(reqs)
    return met, total


def kept_lists(record: dict, requirements_of: Optional[Callable[[int], list]] = None) -> List[dict]:
    """Per round of one sample record: the round whose rules keep-best holds after it, those rules, the prompt and
    output tokens spent so far, and whether the kept rules pass every member in the worst case. With
    `requirements_of` (instance id -> its requirements), also the kept rules' training score (`training_met`)."""
    out, kept, prompt, output = [], None, 0, 0
    for it in record.get("iterations", []):
        prompt += it.get("llm_prompt_tokens") or 0
        output += it.get("llm_output_tokens") or 0
        if kept is None or it.get("improved"):
            kept = it
        entry = {"round": int(it["iteration"]), "kept_round": int(kept["iteration"]),
                 "rules": [(r["condition"], r["action"]) for r in kept["rules"]],
                 "prompt_tokens": prompt, "output_tokens": output, "solved": bool(kept.get("worst_case_success"))}
        if requirements_of is not None:
            entry["train_met"], entry["train_requirements"] = training_met(kept, record.get("instance"),
                                                                           requirements_of)
        out.append(entry)
    return out


def _check_row(row: dict) -> dict:
    certified = row.get("certified")
    return {"parsed": bool(row.get("parsed", True)),
            "certified": bool(certified) if certified == certified and certified is not None else False,
            "met": row.get("met"), "requirements": row.get("requirements"), "error": row.get("error")}


def _seed_cache(run_dir: Path, dataset: str, records: Dict[int, dict]) -> Dict[tuple, dict]:
    """(rules hash, target) -> certification, from an earlier budget CSV and the run's transfer CSV (final lists)."""
    cache = {}
    path = csv_path(run_dir, dataset, "final")
    if path.exists():
        finals = {sid: rules_hash([(r["condition"], r["action"]) for r in rec["final_rules"]])
                  for sid, rec in records.items() if "final_rules" in rec}
        for row in pd.read_csv(path).to_dict("records"):
            if int(row["source"]) in finals:
                cache[(finals[int(row["source"])], int(row["target"]))] = _check_row(row)
    earlier = run_dir / BUDGET_CSV
    if earlier.exists():
        for row in pd.read_csv(earlier).to_dict("records"):
            cache[(row["rules_hash"], int(row["target"]))] = _check_row(row)
    return cache


def budget_table(run_dir: Path, targets: Optional[str] = None, dataset: Optional[str] = None,
                 source_targets: Optional[Dict[str, str]] = None, price: Optional[Sequence[float]] = None,
                 workers: int = 2) -> pd.DataFrame:
    """Every source's kept list after every round of the run in `run_dir`, certified on `targets` of `dataset`
    (default: the run's own); a source in `source_targets` only on its own targets (sample id -> id spec, a subset
    of `targets`). `price`: [$ per M prompt tokens, $ per M output tokens]. Writes <run dir>/budget_transfer.csv."""
    run_dir = Path(run_dir)
    cfg = load_run_config(run_dir)
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    dataset = dataset or cfg.domain.dataset
    instances = {int(t.id): t for t in domain.load_instances(dataset)}
    wanted = [i for i in parse_ids(targets) if i in instances] if targets else sorted(instances)
    guard_sealed(dataset, wanted)
    records = {sid: rec for sid, rec in sample_records(run_dir).items() if rec.get("iterations")}
    own = {sid: [t for t in parse_ids(source_targets[str(sid)]) if t in wanted]
           if source_targets and str(sid) in source_targets else wanted for sid in records}
    trained = training_instances(run_dir) if Path(dataset).name == Path(cfg.domain.dataset).name else {}
    own_instances = instances if Path(dataset).name == Path(cfg.domain.dataset).name else \
        {int(t.id): t for t in domain.load_instances(cfg.domain.dataset)}
    requirements = {}

    def requirements_of(iid: int) -> list:
        if iid not in requirements:
            requirements[iid] = list(domain.spec(own_instances[iid]).requirements)
        return requirements[iid]

    rounds = [{"source": sid, **entry} for sid, rec in records.items() for entry in kept_lists(rec, requirements_of)]
    lists = {rules_hash(r["rules"]): r["rules"] for r in rounds}
    cache = _seed_cache(run_dir, dataset, records)
    todo: Dict[int, Dict[str, list]] = {}
    for r in rounds:
        h = rules_hash(r["rules"])
        for t in own[r["source"]]:
            if (h, t) not in cache:
                todo.setdefault(t, {})[h] = lists[h]
    if todo:
        settings = load_file(DEFAULT_CONFIG, TransferConfig)
        runner = PrismRunner(cfg.prism)
        with ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(certify, domain, cfg, runner, instances[t], sources, settings.exact_near, True)
                       for t, sources in todo.items()]
            for future in as_completed(futures):
                for row in future.result():
                    cache[(row["source"], int(row["target"]))] = _check_row(row)

    price_in, price_out = price if price else (float("nan"), float("nan"))
    rows = []
    for r in rounds:
        h = rules_hash(r["rules"])
        dollars = (r["prompt_tokens"] * price_in + r["output_tokens"] * price_out) / 1e6
        for t in own[r["source"]]:
            rows.append({"source": r["source"], "round": r["round"], "kept_round": r["kept_round"],
                         "first_answer": r["round"] == 1, "rules_hash": h, "num_rules": len(r["rules"]),
                         "prompt_tokens": r["prompt_tokens"], "output_tokens": r["output_tokens"],
                         "dollars": dollars, "solved": r["solved"], "train_met": r["train_met"],
                         "train_requirements": r["train_requirements"], "target": t,
                         "trained": t in trained.get(r["source"], []), **cache[(h, t)]})
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / BUDGET_CSV, index=False)
    return df


# ---------------------------------------------------------------- reading the curves

def rounds_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per source and round of a budget CSV: the cost so far, solved, the kept list's training score (`train_met`
    of `train_requirements`) and the held-out targets it certifies (`certified`, of `held_out` checked; a source's
    own training instances left out)."""
    keys = ["source", "round"]
    meta = df.groupby(keys)[["kept_round", "first_answer", "prompt_tokens", "output_tokens", "dollars", "solved",
                             "train_met", "train_requirements"]].first()
    held = df[~df.trained.astype(bool)]
    counts = held.assign(certified=held.certified.fillna(False).astype(bool)).groupby(keys).certified.agg(
        certified="sum", held_out="size")
    return meta.join(counts).fillna({"certified": 0, "held_out": 0}).reset_index()


def at_budget(table: pd.DataFrame, budget: float, cost: str = "dollars") -> pd.DataFrame:
    """Per source, the list keep-best holds with `budget` to spend per training set (in column `cost`): that of its
    last round whose cumulative cost is within the budget. A source whose first round costs more has nothing yet:
    no round, nothing certified, unsolved."""
    rows = []
    for source, g in table.sort_values("round").groupby("source"):
        within = g[g[cost] <= budget * (1 + 1e-9)]
        if within.empty:
            rows.append({"source": source, "round": np.nan, "spent": 0.0, "certified": 0.0, "solved": False,
                         "train_met": 0.0})
        else:
            last = within.iloc[-1]
            rows.append({"source": source, "round": last["round"], "spent": last[cost],
                         "certified": float(last.certified), "solved": bool(last.solved),
                         "train_met": float(last.train_met)})
    return pd.DataFrame(rows)


def at_round(table: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per source, its row after round k (its last round when it stopped earlier)."""
    return table[table["round"] <= k].sort_values("round").groupby("source").tail(1)


def run_value(table: pd.DataFrame, budget: float, metric: str, cost: str = "dollars") -> float:
    """A run's value at a budget per training set: mean certified per list, or share of training sets solved."""
    picked = at_budget(table, budget, cost)
    return float(picked[metric].astype(float).mean())


def bootstrap_test(a, b, reps: int = 10000, level: float = 0.95, seed: int = 0):
    """Mean of `a` minus mean of `b` (one value per run), a percentile bootstrap interval at `level` (each side's
    runs resampled independently), and a two-sided bootstrap p-value: twice the smaller share of resampled
    differences at or beyond 0 (an interval at level 1 - p just reaches 0)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if not len(a) or not len(b):
        return (float("nan"),) * 4
    rng = np.random.default_rng(seed)
    diffs = rng.choice(a, (reps, len(a))).mean(axis=1) - rng.choice(b, (reps, len(b))).mean(axis=1)
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(diffs, [tail, 100 - tail])
    p = min(1.0, 2 * (min((diffs <= 0).sum(), (diffs >= 0).sum()) + 1) / (reps + 1))
    return float(a.mean() - b.mean()), float(lo), float(hi), float(p)


def holm(p: Sequence[float]) -> np.ndarray:
    """Holm-adjusted p-values (step-down; controls the chance of any false positive among the comparisons)."""
    p = np.asarray(p, dtype=float)
    adjusted, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def permutation_test(a, b, exact_limit: int = 200_000, samples: int = 100_000, seed: int = 0) -> Tuple[float, str]:
    """Two-sided permutation p for the difference of means of `a` and `b` (one value per run): the share of splits of
    the pooled runs into groups of their sizes whose difference is at least as far from 0 as the observed one. Every
    split when there are at most `exact_limit` ("exact"), else `samples` random splits with a fixed seed ("random",
    counting the observed split). With 5 runs a side the smallest p is 2/252 = 0.008."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if not len(a) or not len(b):
        return float("nan"), ""
    pooled, n = np.concatenate([a, b]), len(a)
    observed = abs(a.mean() - b.mean()) - 1e-9
    if math.comb(len(pooled), n) <= exact_limit:
        firsts = np.array(list(itertools.combinations(range(len(pooled)), n)))
        method, extra = "exact", 0
    else:
        rng = np.random.default_rng(seed)
        firsts = np.argsort(rng.random((samples, len(pooled))), axis=1)[:, :n]
        method, extra = "random", 1
    sums = pooled[firsts].sum(axis=1)
    diffs = np.abs(sums / n - (pooled.sum() - sums) / (len(pooled) - n))
    return float(((diffs >= observed).sum() + extra) / (len(diffs) + extra)), method


# ---------------------------------------------------------------- experiments

def experiment_runs(exp, cells: Optional[Sequence[str]] = None) -> Dict[Path, dict]:
    """Run dir -> how to check it (targets, dataset, own targets per source, price), for every run of an experiment's
    cells (src/experiments/base.py), or only of the cells with these codes."""
    from experiments.metrics import model_of
    jobs = {}
    for u in exp.units if cells is None else [u for u in exp.units if u.cell.code in cells]:
        b = u.bench
        own = {s: b.groups[g] for s, g in b.source_groups.items() if g in b.groups} if b.groups else None
        for r in u.runs:
            jobs.setdefault(r, {"targets": b.targets, "dataset": b.dataset, "source_targets": own,
                                "price": exp.prices.get(model_of(r))})
    return jobs


def unit_tables(unit) -> Dict[Path, pd.DataFrame]:
    """Run dir -> its rounds table, for the runs of an experiment unit that have a budget CSV."""
    return {r: rounds_table(pd.read_csv(r / BUDGET_CSV)) for r in unit.runs if (r / BUDGET_CSV).exists()}


def reference_spend(tables: Dict[Path, pd.DataFrame], cost: str = "dollars") -> float:
    """Mean spend per training set of a cell's runs over all their rounds (each run's mean over its sets, then the
    mean over runs)."""
    per_run = [t.sort_values("round").groupby("source")[cost].last().mean() for t in tables.values()]
    return float(np.mean(per_run)) if per_run else float("nan")


def curves(exp, cost: str = "dollars", points: int = 121) -> pd.DataFrame:
    """Per benchmark, cell and run: the training score (requirements met on the training members), certified per
    list and share of sets solved, at budgets up to the largest spend of the benchmark's runs, from the first at
    which every set of the run holds a list."""
    rows = []
    for bench in exp.spec["benchmarks"].values():
        units = [u for u in exp.units if u.bench.name == bench.name]
        tables = {u.cell.name: unit_tables(u) for u in units}
        spends = [t[cost].max() for ts in tables.values() for t in ts.values()]
        if not spends:
            continue
        grid = np.linspace(0, max(spends), points)
        for u in units:
            for run, table in tables[u.cell.name].items():
                start = table[table["round"] == 1][cost].max()     # every set has its first list from here on
                for x in grid[grid >= start]:
                    picked = at_budget(table, x, cost)
                    rows.append({"bench": bench.name, "cell": u.cell.name, "code": u.cell.code,
                                 "run": f"{run.parent.name}/{run.name}", "x": x,
                                 "train_met": picked.train_met.mean(),
                                 "train_requirements": table.train_requirements.max(),
                                 "certified": picked.certified.mean(),
                                 "solved": picked.solved.astype(float).mean()})
    return pd.DataFrame(rows)


def first_answers(exp) -> pd.DataFrame:
    """Per benchmark: certified per list and sets solved by the LLM's first answer (round 1, before any feedback),
    pooled over every list of every cell (round 1 is the same initial prompt in each), and its mean cost."""
    rows = []
    for bench in exp.spec["benchmarks"].values():
        firsts = [t[t["round"] == 1] for u in exp.units if u.bench.name == bench.name
                  for t in unit_tables(u).values()]
        if firsts:
            f = pd.concat(firsts)
            rows.append({"bench": bench.name, "lists": len(f), "train_met": f.train_met.mean(),
                         "train_requirements": f.train_requirements.max(), "certified": f.certified.mean(),
                         "solved": f.solved.astype(float).mean(), "dollars": f.dollars.mean(),
                         "output_tokens": f.output_tokens.mean()})
    return pd.DataFrame(rows)


def overfitting(exp) -> pd.DataFrame:
    """Per benchmark and cell: the kept lists' training score and held-out certified count after round 1 and at the
    reference cell's mean spend (mean over runs of each run's mean over its lists). A training score that rises
    while held-out certification stays flat is the loop patching the training instances."""
    spec = exp.spec.get("budget") or {}
    ref = spec.get("reference") or exp.reference()
    cost = spec.get("cost", "dollars")
    rows = []
    for bench in exp.spec["benchmarks"].values():
        units = [u for u in exp.units if u.bench.name == bench.name]
        ref_units = [u for u in units if u.cell.name == ref]
        if not ref_units:
            continue
        spend = reference_spend(unit_tables(ref_units[0]), cost)
        for u in units:
            tables = list(unit_tables(u).values())
            if not tables:
                continue
            first = [at_round(t, 1) for t in tables]
            later = [at_budget(t, spend, cost) for t in tables]
            rows.append({"bench": bench.name, "cell": u.cell.name, "runs": len(tables), "budget": spend,
                         "train_requirements": max(t.train_requirements.max() for t in tables),
                         "train_first": np.mean([f.train_met.mean() for f in first]),
                         "train_at": np.mean([x.train_met.mean() for x in later]),
                         "certified_first": np.mean([f.certified.mean() for f in first]),
                         "certified_at": np.mean([x.certified.mean() for x in later])})
    return pd.DataFrame(rows)


def planned(exp) -> pd.DataFrame:
    """The experiment's planned comparisons (`budget.planned`) on every benchmark: the reference cell against another
    cell, each run read at a fraction of the reference's mean spend. Two tests over runs, each Holm-corrected over
    every comparison on every benchmark together (m fixed at their number; one not yet computed counts as p = 1): the
    bootstrap fixed before the runs (`p`, with its interval) and the exact permutation test (`p_exact`)."""
    spec = exp.spec["budget"]
    ref = spec.get("reference") or exp.reference()
    cost = spec.get("cost", "dollars")
    rows = []
    for bench in exp.spec["benchmarks"].values():
        units = {u.cell.name: u for u in exp.units if u.bench.name == bench.name}
        if ref not in units:
            continue
        ref_tables = unit_tables(units[ref])
        spend = reference_spend(ref_tables, cost)
        for comp in spec["planned"]:
            other = units.get(comp["cell"])
            budget = comp["at"] * spend
            a = [run_value(t, budget, comp["metric"], cost) for t in ref_tables.values()]
            b = [run_value(t, budget, comp["metric"], cost) for t in (unit_tables(other) if other else {}).values()]
            diff, lo, hi, p = bootstrap_test(a, b, level=spec.get("level", 0.95))
            p_exact, method = permutation_test(a, b)
            rows.append({"bench": bench.name, "label": bench.label or bench.name, "comparison": comp["name"],
                         "metric": comp["metric"], "cell": comp["cell"], "at": comp["at"], "budget": budget,
                         "runs_ref": len(a), "runs_other": len(b),
                         "ref_value": np.mean(a) if a else np.nan, "other_value": np.mean(b) if b else np.nan,
                         "diff": diff, "lo": lo, "hi": hi, "p": p, "p_exact": p_exact, "permutations": method})
    table = pd.DataFrame(rows)
    if not table.empty:
        m = len(spec["planned"]) * len(exp.spec["benchmarks"])
        for col in ("p", "p_exact"):
            padded = list(table[col].fillna(1.0)) + [1.0] * (m - len(table))
            table[f"{col}_holm"] = holm(padded)[:len(table)]
            table.loc[table[col].isna(), f"{col}_holm"] = np.nan
    return table


def report_section(exp) -> List[str]:
    """The report's budget section (an experiment file with `budget:`): the planned comparisons, the LLM's first
    answers, the overfitting readout (with the file's `budget.overfitting` sentence), and the budget figure
    (viz/plot_budget_transfer.py)."""
    import sys
    from settings import PROJECT_ROOT
    sys.path.insert(0, str(PROJECT_ROOT / "viz"))
    from plot_budget_transfer import PlotConfig, draw

    spec = exp.spec["budget"]
    ref = spec.get("reference") or exp.reference()
    plot = load_file(PROJECT_ROOT / spec["plot"], PlotConfig) if spec.get("plot") else PlotConfig(
        "plot_budget_transfer", str(exp.spec["path"]))
    cost = spec.get("cost", plot.cost)
    level = spec.get("level", 0.95)
    table = planned(exp)
    m = len(spec["planned"]) * len(exp.spec["benchmarks"])
    methods = set(table.permutations.dropna()) - {""} if not table.empty else set()
    exact_how = ("every split of the pooled runs enumerated" if methods <= {"exact"} else
                 "100,000 random splits with a fixed seed where the splits are too many to enumerate"
                 + (", every split elsewhere" if "exact" in methods else ""))
    lines = ["## Planned comparisons: feedback against restarts at equal cost", "",
             spec.get("note", "").strip(), "",
             f"Δ = **{ref}** minus the other cell, each run read at the given budget per training set (the list "
             f"keep-best holds after its last round within the budget; {cost} at the prices above). Interval: "
             f"{level:.0%} bootstrap over runs. Two tests, each Holm-corrected over all {m} comparisons together "
             "(one without runs yet counts as p = 1); below 0.05 counts. p (bootstrap): two-sided bootstrap over "
             "runs, the test fixed before the runs. p (exact): two-sided permutation test over runs, "
             f"{exact_how}. At these run counts the exact test is the more reliable one: the bootstrap understates "
             "the spread with 5 to 10 runs a side (with 5 v 5 the smallest exact p is 2/252 = 0.008, so nothing "
             f"could pass Holm over {m}).", "",
             "| benchmark | comparison | against | budget | runs | reference | other | Δ [interval] | p (bootstrap) "
             "| p (Holm) | p (exact) | p (exact, Holm) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        fmt = ".0%" if r.metric == "solved" else ".2f"
        money = f"${r.budget:.4f}" if cost == "dollars" else f"{r.budget:,.0f}"
        delta = "–" if np.isnan(r.diff) else (f"{r.diff * 100:+.0f} pts [{r.lo * 100:+.0f}, {r.hi * 100:+.0f}]"
                                              if r.metric == "solved" else f"{r.diff:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]")
        lines.append(f"| {r.label.split(' (')[0]} | {r.comparison} | {r.cell} | {money} | "
                     f"{r.runs_ref} v {r.runs_other} | "
                     f"{format(r.ref_value, fmt) if r.runs_ref else '–'} | "
                     f"{format(r.other_value, fmt) if r.runs_other else '–'} | {delta} | "
                     + " | ".join("–" if pd.isna(v) else f"{v:.3f}" for v in (r.p, r.p_holm, r.p_exact, r.p_exact_holm))
                     + " |")
    firsts = first_answers(exp)
    if not firsts.empty:
        lines += ["", "The LLM's first answer (round 1, before any feedback; the same initial prompt in every cell, "
                  "so pooled over all their lists): an \"LLM alone\" point.", "",
                  "| benchmark | lists | certified per list | sets solved | dollars per set |", "|---|---|---|---|---|"]
        labels = {b.name: (b.label or b.name).split(" (")[0] for b in exp.spec["benchmarks"].values()}
        for r in firsts.itertuples():
            lines.append(f"| {labels[r.bench]} | {r.lists} | {r.certified:.2f} | {r.solved:.0%} | ${r.dollars:.4f} |")
    over = overfitting(exp)
    if not over.empty:
        lines += ["", "Training against held-out (overfitting readout): the kept lists' requirements met in the worst "
                  "case, summed over the training members, beside the held-out targets they certify, after round 1 "
                  f"and at **{ref}**'s mean spend. Training rising while held-out stays flat means the loop patches "
                  "the training instances.", "",
                  "| benchmark | condition | runs | training met: round 1 → at spend | held-out certified: round 1 → "
                  "at spend |", "|---|---|---|---|---|"]
        labels = {b.name: (b.label or b.name).split(" (")[0] for b in exp.spec["benchmarks"].values()}
        for r in over.itertuples():
            lines.append(f"| {labels[r.bench]} | {r.cell} | {r.runs} | {r.train_first:.1f} → {r.train_at:.1f} "
                         f"(of {r.train_requirements:.0f}) | {r.certified_first:.2f} → {r.certified_at:.2f} |")
        if spec.get("overfitting"):
            lines += ["", spec["overfitting"].strip()]
    data = curves(exp, cost, plot.points)
    if not data.empty:
        marks = {b: sorted(set(g.budget.dropna())) for b, g in table.groupby("bench")} if not table.empty else {}
        plot.cost = cost
        path = draw(data, exp, plot, marks, firsts)
        data.to_csv(exp.out / "budget_transfer.csv", index=False)
        lines += ["", f"![training score, certified and solved against spend]({path.name})", "",
                  "Rows: training requirements met per list, held-out targets certified per list, training sets "
                  "solved. Each line: a cell's mean over runs, the band its range over runs; vertical lines: the "
                  "budgets of the planned comparisons; ×: the LLM's first answer. Curve data: `budget_transfer.csv`."]
    return lines + [""]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="configs/experiments/<name>.yaml, or run dirs")
    parser.add_argument("--data", default=None, help="Run dirs: target dataset (default: the run's own)")
    parser.add_argument("--targets", default=None, help='Run dirs: target ids, e.g. "12-19" (default: all)')
    parser.add_argument("--source-targets", default=None, help='Run dirs: JSON, source -> its own target ids')
    parser.add_argument("--price", type=float, nargs=2, default=None, metavar=("IN", "OUT"),
                        help="Run dirs: $ per M prompt tokens and per M output tokens")
    parser.add_argument("--cells", default=None, help='Experiment files: only these cell codes, e.g. "F,R 18"')
    parser.add_argument("--workers", type=int, default=2, help="Targets certified at once per run")
    parser.add_argument("--max-parallel", type=int, default=2, help="Runs checked at once")
    args = parser.parse_args(argv)

    jobs: Dict[Path, dict] = {}
    for path in args.paths:
        if path.suffix in (".yaml", ".yml"):
            from experiments.base import experiment
            jobs.update(experiment_runs(experiment(path), args.cells.split(",") if args.cells else None))
        else:
            jobs[path] = {"targets": args.targets, "dataset": args.data, "price": args.price,
                          "source_targets": json.loads(args.source_targets) if args.source_targets else None}
    print(f"{len(jobs)} runs", flush=True)

    def one(run: Path, job: dict) -> str:
        df = budget_table(run, workers=args.workers, **job)
        table = rounds_table(df)
        last = table.sort_values("round").groupby("source").tail(1)
        return (f"[{run.parent.name}/{run.name}] {table['round'].max()} rounds, {df.rules_hash.nunique()} lists; "
                f"round 1: {at_round(table, 1).certified.mean():.2f} certified per list, "
                f"last: {last.certified.mean():.2f}")

    with ThreadPoolExecutor(args.max_parallel) as pool:
        futures = {pool.submit(one, run, job): run for run, job in jobs.items()}
        for future in as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as e:     # one run's failure must not stop the others
                print(f"[{futures[future]}] failed: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
