"""Runs indexed by what they did, so that experiments share them instead of repeating them.

A run's identity is its resolved config (recorded in <run>/config.json; keys it lacks take their default) without the
settings that cannot change its results: the condition's name, parallelism and scheduler, timeouts and client
retries, pruning the orders for `any` rules, and the seed, which only labels a repeat. Runs made under different condition names with the same identity
are repeats of one configuration. A code change that alters what a run does must therefore come with a config key
(as every behaviour change here does), or the store would mix runs made before and after it.

A run's optional <run>/identity.json corrects what its config.json could not record, e.g. domain features its
prompts never showed although its config does not hide them (recorded as hidden): {"note": ..., "<section>": {"<key>": value}}.
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config import Config, _build, _merge, _resolve_scheduler, _read_yaml, RUN_CONFIG_DIR, validate
from results_io import RESULT_FILES
from settings import RESULTS_PATH

CORE_PATH = RESULTS_PATH / "core"
ABLATIONS_PATH = RESULTS_PATH / "ablations"
ROOTS = (CORE_PATH, ABLATIONS_PATH)

IGNORED = {"condition", "run.workers", "run.scheduler", "llm.seed", "llm.openrouter.timeout_s",
           "llm.openrouter.max_retries", "prism.java_max_mem", "prism.timeout_s", "rules.any_prune"}


def _flat(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out.update(_flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def identity(cfg: Config) -> Dict[str, Any]:
    """The settings that make a run what it is (flattened `section.key` -> value)."""
    return {k: v for k, v in _flat(cfg.to_dict()).items() if k not in IGNORED}


def fingerprint(cfg: Config) -> str:
    """A short hash of `identity(cfg)`."""
    return hashlib.sha1(json.dumps(identity(cfg), sort_keys=True, default=str).encode()).hexdigest()[:12]


def recorded_config(run_dir: Path) -> Config:
    """The settings a run had: its config.json over the run defaults, corrected by its identity.json if any."""
    data = _merge(_read_yaml(RUN_CONFIG_DIR / "default.yaml"),
                  json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    fix = run_dir / "identity.json"
    if fix.is_file():
        data = _merge(data, {k: v for k, v in json.loads(fix.read_text(encoding="utf-8")).items() if k != "note"})
    cfg = _build(Config, _resolve_scheduler(data))
    validate(cfg)
    return cfg


def finished(run_dir: Path, cfg: Config) -> bool:
    return (run_dir / RESULT_FILES[cfg.approach]).exists()


class RunStore:
    """Every run under `roots` (<root>/<name>/seed_*/ with a config.json), by fingerprint. Runs whose config no
    longer loads (keys since removed) are listed in `unreadable` and never matched."""

    def __init__(self, roots=ROOTS):
        self.roots = tuple(roots)
        self._index: Dict[str, List[Tuple[Path, bool]]] = {}
        self.unreadable: List[Path] = []
        self.refresh()

    def refresh(self) -> None:
        self._index, self.unreadable = {}, []
        for root in self.roots:
            for run in sorted(root.glob("*/seed_*")):
                if not (run / "config.json").is_file():
                    continue
                try:
                    cfg = recorded_config(run)
                except (ValueError, KeyError, TypeError):
                    self.unreadable.append(run)
                    continue
                self._index.setdefault(fingerprint(cfg), []).append((run, finished(run, cfg)))

    def runs(self, cfg: Config) -> List[Path]:
        """Finished runs with `cfg`'s identity, in path order."""
        return [run for run, done in self._index.get(fingerprint(cfg), []) if done]

    def pending(self, cfg: Config) -> List[Path]:
        """Started but unfinished runs with `cfg`'s identity: still running elsewhere, or failed (delete those)."""
        return [run for run, done in self._index.get(fingerprint(cfg), []) if not done]
