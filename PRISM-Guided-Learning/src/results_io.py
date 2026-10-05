"""Reading finished runs: their result files, what each run was, and the policy the legacy loop kept."""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from legacy.gridworld import GridWorld as LegacyGridWorld
from legacy.requirements import SimplifiedVerifier, get_threshold_for_key

SYMBOLIC_RESULTS = "SYMBOLIC_results.parquet"
LEGACY_RESULTS = "LEGACY_FEEDBACK_SIMPLIFIED_results.parquet"
RESULT_FILES = {"symbolic": SYMBOLIC_RESULTS, "legacy": LEGACY_RESULTS}

# What every run made before runs recorded config.json used (out/results/legacy_grid20 and
# symbolic_grid20_*). A historical record, not a setting. out/results/symbolic_uuv also predates
# config.json but is a UUV run; readers that need its domain must not rely on this.
PRE_CONFIG_RUNS = {"domain": "gridworld", "dataset": "grid_20_balanced.csv", "model": "qwen3:14b-q4_K_M",
                   "max_rounds": 5}

_quiet = logging.getLogger("results_io")
_quiet.addHandler(logging.NullHandler())
_quiet.propagate = False


@dataclass
class RunFacts:
    """What a finished run was, as far as reports and figures need to know."""
    approach: str          # symbolic | legacy
    domain: str
    dataset: str
    model: str
    max_rounds: int
    recorded: bool         # False: the run predates config.json, the facts are PRE_CONFIG_RUNS


def run_facts(run_dir) -> RunFacts:
    """Facts of the run in `run_dir`, from its config.json (or PRE_CONFIG_RUNS for older runs)."""
    run_dir = Path(run_dir)
    approach = "legacy" if (run_dir / LEGACY_RESULTS).exists() else "symbolic"
    path = run_dir / "config.json"
    if not path.exists():
        return RunFacts(approach, recorded=False, **PRE_CONFIG_RUNS)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return RunFacts(approach, cfg["domain"]["name"], cfg["domain"]["dataset"], cfg["llm"]["model"],
                    cfg["legacy" if approach == "legacy" else "planner"]["max_rounds"], recorded=True)


def legacy_kept(record, instance) -> dict:
    """Probabilities of the policy the legacy keep-best loop ended with, for one sample's output record.

    Newer legacy runs store them directly. Otherwise, replay the rule: fewest failed
    requirements, ties broken by the higher weighted score, earliest iteration first.
    """
    if record.get("final_prism_probs"):
        return record["final_prism_probs"]
    iterations = record.get("iteration_prism_probs", [])
    if not iterations:
        return {}
    d = instance.data
    verifier = SimplifiedVerifier(None, LegacyGridWorld(d["n"], d["goals"], d["static"], d["moving"]), _quiet)
    best, best_key = iterations[0], None
    for probs in iterations:
        mistakes = sum(1 for k, p in probs.items() if p < get_threshold_for_key(k))
        key = (mistakes, -verifier._calculate_score(list(probs.values())))
        if best_key is None or key < best_key:
            best, best_key = probs, key
    return best
