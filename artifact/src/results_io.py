"""Reading finished runs: their result files, what each run was, and the requirements each sample was
checked against."""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from core.domain import Requirement, load_domain

SYMBOLIC_RESULTS = "SYMBOLIC_results.parquet"


def sample_records(run_dir) -> Dict[int, dict]:
    """Sample id -> the run's per-sample record (`outputs/sample_<id>.json`), in sample order."""
    return {int(path.stem.split("_")[1]): json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((Path(run_dir) / "outputs").glob("sample_*.json"))}


@dataclass
class RunFacts:
    """What a finished run was, as far as reports and figures need to know."""
    domain: str
    dataset: str
    model: str
    max_rounds: int


def run_facts(run_dir) -> RunFacts:
    """Facts of the run in `run_dir`, from its config.json."""
    cfg = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    return RunFacts(cfg["domain"]["name"], cfg["domain"]["dataset"], cfg["llm"]["model"], cfg["planner"]["max_rounds"])


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
                         f"({', '.join(by_name)})")
    met = sum(by_name[name].satisfied(value) for name, value in values.items())
    return met, sum(by_name[name].shortfall(value) for name, value in values.items())
