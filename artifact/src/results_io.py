"""Reading finished runs: their result files, what each run was, and the requirements each sample was
checked against."""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from core.domain import Requirement, load_domain

SYMBOLIC_RESULTS = "SYMBOLIC_results.parquet"
LEGACY_RESULTS = "LEGACY_FEEDBACK_SIMPLIFIED_results.parquet"
RESULT_FILES = {"symbolic": SYMBOLIC_RESULTS, "legacy": LEGACY_RESULTS}

# What every run made before runs recorded config.json used (older runs). A historical record, not a setting.
PRE_CONFIG_RUNS = {"domain": "gridworld", "dataset": "grid_20_balanced.csv", "model": "qwen3:14b-q4_K_M",
                   "max_rounds": 5}


def sample_records(run_dir) -> Dict[int, dict]:
    """Sample id -> the run's per-sample record (`outputs/sample_<id>.json`), in sample order."""
    return {int(path.stem.split("_")[1]): json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((Path(run_dir) / "outputs").glob("sample_*.json"))}


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


@lru_cache(maxsize=None)
def _requirements(domain: str, dataset: str) -> Tuple[Tuple[Requirement, ...], ...]:
    d = load_domain(domain)
    return tuple(tuple(d.spec(instance).requirements) for instance in d.load_instances(dataset))


def requirements_by_sample(run_dir) -> List[List[Requirement]]:
    """Per sample id (the instance's row in the run's dataset), the requirements it was checked against,
    with the thresholds and bounds the domain defines for that instance."""
    facts = run_facts(run_dir)
    return [list(reqs) for reqs in _requirements(facts.domain, facts.dataset)]


def met_and_shortfall(requirements: Sequence[Requirement], values: Dict[str, float]) -> Tuple[int, float]:
    """How many of `values` (requirement name -> value) meet their requirement, and their total shortfall
    (`Requirement.shortfall`, summed in the order of `values`)."""
    by_name = {r.name: r for r in requirements}
    unknown = set(values) - set(by_name)
    if unknown:
        raise ValueError(f"values for {sorted(unknown)}, which are not requirements of this instance "
                         f"({', '.join(by_name)}). A run without config.json from a domain other than "
                         f"{PRE_CONFIG_RUNS['domain']}?")
    met = sum(by_name[name].satisfied(value) for name, value in values.items())
    return met, sum(by_name[name].shortfall(value) for name, value in values.items())
