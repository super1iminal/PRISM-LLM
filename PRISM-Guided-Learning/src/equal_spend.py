"""The feedback loop at another arm's spend (RQ2, feedback against pure restarts): every chain of a feedback run cut to
its first R rounds, R the most that keeps the training set's LLM spend (all its chains, run side by side) within a
budget, then the run's own final pick (`planner.pick: candidates`, with repair) on what is left, verified on the
training members. No LLM calls. Writes a run of its own (config.json, outputs/, determinized.json, norepair.json) that
transfer.py certifies like any other.

Usage (from PRISM-Guided-Learning/):
    python src/equal_spend.py <feedback run dir> (--budget DOLLARS | --budget-from CONDITION)
        [--prices EXPERIMENT_YAML | --price IN,OUT] [--out DIR] [--no-exact] [--repair-score members_first] [--as-run]
then, for example,
    python src/transfer.py out/results/equal_spend/port_fam/seed_none_1 --rules determinized --targets 16-35,76-95,136-155

--budget-from: the mean LLM spend per training set over that condition's runs (e.g. port_fam_resample). Prices ($ per
million prompt and output tokens) of the run's model come from an experiment config's `prices` (default
configs/experiments/rq1_comparison.yaml) or from --price. --out defaults to out/results/equal_spend/<condition>/<seed
dir>, outside core/ and ablations/: the run copies the feedback run's config.json, so the run store would count it
as one more feedback run. --no-exact: the picked list's check on the members with PRISM's loop solver instead of
interval iteration (on the families that check can take an hour per list; the held-out checks are unchanged).
--repair-score: the pick's repair objective (`planner.repair_score`) instead of the run's, recorded in the new run's
config.json. --as-run: the new run stands in for the feedback run itself (e.g. its pick with repair, at a budget that
keeps every round): each record keeps the kept rounds and the run's other fields, and the run gets a results parquet
and the LLM task log (llm_tasks.jsonl, when every round is kept), so reports read it as any finished run.
"""
import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from config import load_run_config
from core.domain import load_domain
from core.planner import SymbolicPlanner
from core.training import training_sets
from experiments.metrics import model_of, run_numbers
from run_ablation import existing_root
from run_symbolic import save_results, write_determinized
from settings import PROJECT_ROOT, RESULTS_PATH

OUT_ROOT = RESULTS_PATH / "equal_spend"
DEFAULT_PRICES = PROJECT_ROOT / "configs" / "experiments" / "rq1_comparison.yaml"


def cost(it: Dict[str, Any], price: Sequence[float]) -> float:
    """One round's LLM spend in dollars (as experiments.metrics.run_numbers)."""
    return ((it.get("llm_prompt_tokens") or 0) * price[0] + (it.get("llm_output_tokens") or 0) * price[1]) / 1e6


def condition_spend(condition: str, prices: Dict[str, List[float]]) -> float:
    """The mean LLM spend per training set over a condition's finished runs."""
    runs = sorted(p for p in (existing_root(condition) / condition).glob("seed_*") if (p / "outputs").is_dir())
    if not runs:
        raise SystemExit(f"no finished runs of {condition}")
    return sum(run_numbers(r, prices)["dollars"] for r in runs) / len(runs)


def equal_spend(run: Path, budget: float, price: Sequence[float], out: Path, exact: bool = True,
                repair_score: Optional[str] = None, as_run: bool = False) -> List[Dict]:
    """The re-picked records of `run` at `budget` dollars per training set, written to `out`."""
    cfg = load_run_config(run)
    if not exact:
        cfg.prism.exact_check = False
    if repair_score:
        cfg.planner.repair_score = repair_score
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    sets = training_sets(domain.load_instances(cfg.domain.dataset), cfg.run.train_sets, cfg.run.hidden_sets,
                         cfg.run.pool_sets)[:cfg.run.limit]
    planner = SymbolicPlanner(domain, None, cfg)
    planner.select_chain = lambda target, results: {"chain": None, "members": []}   # the pick replaces it
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "config.json", out / "config.json")
    if repair_score:
        recorded = json.loads((out / "config.json").read_text(encoding="utf-8"))
        recorded.setdefault("planner", {})["repair_score"] = repair_score
        (out / "config.json").write_text(json.dumps(recorded, indent=1), encoding="utf-8")
    records = []
    for sid, target in enumerate(sets):
        rec = json.loads((run / "outputs" / f"sample_{sid:03d}.json").read_text(encoding="utf-8"))
        its = rec["iterations"]
        full, top = sum(cost(it, price) for it in its), max(it["chain_iteration"] for it in its)
        fits = [r for r in range(1, top + 1) if sum(cost(it, price) for it in its if it["chain_iteration"] <= r) <= budget]
        rounds = max(fits) if fits else 1
        kept = [it for it in its if it["chain_iteration"] <= rounds]
        results = [{"iterations": [dict(it, iteration=it["chain_iteration"]) for it in kept if it["chain"] == c]}
                   for c in range(cfg.planner.chains)]
        res = planner.pick_candidates(target, results, log=lambda line: None)
        spent = sum(cost(it, price) for it in kept)
        record = {**res, "instance": rec["instance"],
                  "equal_spend": {"rounds": rounds, "of_rounds": top, "spent": spent, "full_spend": full,
                                  "budget": budget, "source": f"{run.parent.name}/{run.name}"}}
        if as_run:      # the run's own record, with the new pick's fields over it, and the kept rounds
            record = {**rec, **record, "iterations": kept, "chain": rec.get("chain"), "chains": rec.get("chains")}
        records.append(record)
        (out / "outputs" / f"sample_{sid:03d}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        chosen = res["pick"]["candidates"][res["pick"]["chosen"]]["from"] if "pick" in res else None
        print(f"{run.parent.name} {run.name} set {sid}: rounds 1-{rounds} of {top} (${spent:.3f} of ${full:.3f}), "
              f"picked {chosen}, training {'passes' if res.get('success') else 'fails'}", flush=True)
    write_determinized(records, str(out))
    if as_run:
        save_results(records, sets, str(out))
        if all(r["equal_spend"]["rounds"] == r["equal_spend"]["of_rounds"] for r in records) and \
                (run / "llm_tasks.jsonl").exists():
            shutil.copy(run / "llm_tasks.jsonl", out / "llm_tasks.jsonl")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="A finished feedback run (planner.pick: candidates)")
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--budget", type=float, help="Dollars per training set")
    budget.add_argument("--budget-from", metavar="CONDITION", help="The mean spend per set of this condition's runs")
    prices = parser.add_mutually_exclusive_group()
    prices.add_argument("--prices", type=Path, default=DEFAULT_PRICES, help="Experiment config with a `prices` table")
    prices.add_argument("--price", help="IN,OUT: $ per million prompt and output tokens of the run's model")
    parser.add_argument("--out", type=Path, default=None, help="Default: out/results/equal_spend/<condition>/<seed dir>")
    parser.add_argument("--no-exact", action="store_true", help="Members' check with PRISM's loop solver")
    parser.add_argument("--as-run", action="store_true", help="Records, results parquet and task log of a finished run")
    parser.add_argument("--repair-score", choices=["capped_sum", "members_first"],
                        help="The pick's repair objective (planner.repair_score; default: the run's)")
    args = parser.parse_args(argv)

    run = args.run_dir.resolve()
    model = model_of(run)
    table = ({model: [float(x) for x in args.price.split(",")]} if args.price
             else yaml.safe_load(args.prices.read_text(encoding="utf-8")).get("prices") or {})
    if model not in table:
        raise SystemExit(f"no price for {model!r} (give --price IN,OUT)")
    dollars = args.budget if args.budget is not None else condition_spend(args.budget_from, table)
    out = args.out or OUT_ROOT / run.parent.name / run.name
    print(f"{run.parent.name} {run.name}: budget ${dollars:.4f} per set"
          + (f" (the mean of {args.budget_from})" if args.budget_from else ""))
    equal_spend(run, dollars, table[model], out, exact=not args.no_exact, repair_score=args.repair_score,
                as_run=args.as_run)


if __name__ == "__main__":
    main()
