"""Run configuration: one schema, defaults in configs/run/default.yaml, one file per named condition in
configs/run/conditions/<name>.yaml.

    cfg = load_config("R2", overrides=["llm.seed=1", "run.limit=1"])

A condition file is a partial override of the default; CLI overrides (`section.key=value`, value parsed
as YAML) apply last. Unknown or missing keys are errors, so a typo cannot silently fall back to
a default. `docs/config.md` documents every key.

The other tools keep their settings in their own folders (configs/regression/, configs/ablation/,
configs/plot/) with their own schemas, loaded the same way by `load_file`.
"""
import argparse
import json
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

from settings import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "configs"
RUN_CONFIG_DIR = CONFIG_DIR / "run"
CONDITIONS_DIR = RUN_CONFIG_DIR / "conditions"


# Each section mirrors a block of configs/run/default.yaml, which holds every default. No key has a
# default here, so a key missing from the YAML is an error rather than a silent second default.

@dataclass
class DomainConfig:
    name: str
    dataset: str
    visible_extra: List[str]                  # extra state variables rules may read, if in the model


@dataclass
class OpenRouterConfig:
    base_url: str
    api_key_env: str                          # environment variable holding the API key (never the key itself)
    providers: List[str]                      # provider slugs to use, in order, with no fallback; [] = OpenRouter routes
    quantizations: List[str]                  # accepted weight precisions, e.g. [fp8, bf16]; [] = any
    timeout_s: float                          # per request
    max_retries: int                          # the client's retries on rate limits and server errors


@dataclass
class LLMConfig:
    model: str
    think: bool
    num_ctx: int
    num_predict: int
    seed: Optional[int]
    temperature: Optional[float]              # None = the model's default
    backend: str                              # serving engine (core/backends)
    openrouter: OpenRouterConfig              # used by llm.backend=openrouter only


@dataclass
class PlannerConfig:
    max_rounds: int
    max_fixups: int                           # extra calls per round when the answer has invalid rules
    retry: str                                # stall:k | never | every:k | gain:eps | always
    branch: str                               # joint | per_requirement
    feedback: str                             # blame (REFINE/EXTEND with blame) | table (S1: results table only)
    max_rules: int
    max_condition_chars: int


@dataclass
class FeedbackConfig:
    blame: str                                # mass | regret (S5) | random (V1) | none (V2)
    horizon: Union[int, str]                  # occupancy steps in the mass analysis, or "domain" (Domain.horizon)
    top_k: int                                # hotspots / rules shown in feedback
    states_per_rule: int
    space: str                                # states (valuations) | features (feature values per action)
    trajectory: bool                          # show a likely way to fail the worst-failing requirement
    optimal: bool                             # name the optimal strategy's actions at the states the feedback names
    stuck: bool                               # say where the rules' action there changes nothing (a self-loop)
    deviations: bool                          # list the states the rules reach where they lose most against the optimum

    def horizon_for(self, domain, instance) -> int:
        """Occupancy horizon for `instance`: `horizon` steps, or with "domain" the domain's own
        (e.g. UUV's mission deadline)."""
        if self.horizon != "domain":
            return int(self.horizon)
        value = domain.horizon(instance)
        if value is None:
            raise ValueError(f"domain {domain.name!r} has no horizon of its own; set feedback.horizon to a number")
        return int(value)


@dataclass
class PromptConfig:
    catch_all_instruction: bool
    examples: bool
    generality: bool                # say the rules will be certified unchanged on unseen, larger instances
    strategy_first: bool            # the answer starts with the strategy in words (`strategy`), then the rules
    tie_break: bool                 # explain that `any` is resolved against the policy, and how to fix an order
    hint: str                       # extra advice for the policy section ("" = none), set by a condition
    demonstration: bool             # the initial prompt shows one optimal run on the first training member


@dataclass
class PrismConfig:
    method: str                               # iterative method; plain value iteration oscillates on periodic chains
    fallback_methods: List[str]               # tried in order when `method` does not converge
    java_max_mem: str
    max_iters: int
    timeout_s: float
    multi_engine: str                         # PRISM's explicit engine has no multi-objective support
    multi_method: str                         # exact LP; value iteration fails to converge on periodic chains
    exact_check: bool                         # re-verify the final policy with sound solvers and report those values
    exact_epsilon: str                        # interval iteration's precision (passed to PRISM as written)
    exact_fallback_epsilon: str               # Gauss-Seidel's, where interval iteration does not converge
    exact_max_iters: int
    max_concurrent: Optional[int]            # PRISM processes at once on the whole machine (lock-file slots); null: no cap


@dataclass
class RulesConfig:
    max_enumeration: int
    extended: bool                  # instance constants, domain features and `any` rules
    general: bool                   # only the numbers 0 and 1 in conditions (with extended)
    hidden_features: List[str]      # domain features left out of the vocabulary
    any_order: str                  # none | rotations | all: resolve `any` rules into the order that does best
                                    # on the members each round; declared: always the declared action order;
                                    # report: keep `any`, tell the LLM the best order (none: worst case over them)
    any_scorer: str                 # prism | domain: what scores the orders (domain: the domain's fast model; PRISM
                                    # then verifies only the chosen order, which can choose another order on ties)
    any_prune: bool                 # the domain's fast model rules out orders that cannot score best, PRISM verifies
                                    # the rest: the same choice (run store ignores it)


@dataclass
class LegacyConfig:
    max_rounds: int
    retry: str                                # never | stall:k (L1: blind retry with the initial prompt)
    examples: bool                            # the two worked examples in the initial prompt (L2: off)


@dataclass
class RunConfig:
    workers: int                              # instances in flight (threads, lockstep batch size, event slots)
    limit: Optional[int]
    scheduler: str                            # threads | lockstep (batched) | event (as results arrive); docs/scheduler.md
    train_sets: List[List[int]]               # groups of instance ids, one rule set each; [] = one per instance
    hidden_sets: List[List[int]]              # per training set, ids checked and scored but never shown; [] = none
    pool_sets: List[List[int]]                # per training set, counterexample candidates added when they fail


@dataclass
class Config:
    approach: str                             # symbolic | legacy
    domain: DomainConfig
    llm: LLMConfig
    planner: PlannerConfig
    feedback: FeedbackConfig
    prompt: PromptConfig
    prism: PrismConfig
    rules: RulesConfig
    legacy: LegacyConfig
    run: RunConfig
    condition: Optional[str] = None           # name of the applied condition (set by load_config, not a setting)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (update or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _build(cls, data: Dict[str, Any], path: str = ""):
    """Instantiate dataclass `cls` from `data`, rejecting unknown and missing keys."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown config key(s) {sorted(path + k for k in unknown)}")
    missing = [n for n, f in known.items() if n not in data and f.default is MISSING and f.default_factory is MISSING]
    if missing:
        raise ValueError(f"missing config key(s) {[path + k for k in missing]}")
    kwargs = {}
    for name, value in data.items():
        if is_dataclass(known[name].type):
            if not isinstance(value, dict):
                raise ValueError(f"config section {path}{name} must be a mapping, got {value!r}")
            kwargs[name] = _build(known[name].type, value, f"{path}{name}.")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _parse_override(text: str) -> Dict[str, Any]:
    key, sep, value = text.partition("=")
    if not sep:
        raise ValueError(f"override {text!r} must look like section.key=value")
    nested: Dict[str, Any] = yaml.safe_load(value) if value.strip() else None
    for part in reversed(key.strip().split(".")):
        nested = {part: nested}
    return nested


def _read_yaml(path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_file(path, schema: type, overrides: Sequence[str] = ()):
    """A tool's config file as dataclass `schema`, then `key=value` overrides (same rules as the run config)."""
    data = _read_yaml(path)
    for text in overrides:
        data = _merge(data, _parse_override(text))
    return _build(schema, data)


def plot_config(schema: type, script: str, argv=None):
    """Parse `CONFIG [--set key=value ...]` and load that plot config as `schema`. The file names the
    script it is for (`script:`), so a figure's config cannot be fed to the wrong script."""
    parser = argparse.ArgumentParser(description=f"Plot from a config in configs/plot/ (script: {script})")
    parser.add_argument("config", type=Path, help="Plot config file, e.g. configs/plot/<figure>.yaml")
    parser.add_argument("--set", action="append", default=[], help="Override key=value (repeatable)")
    args = parser.parse_args(argv)
    named = (yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}).get("script")
    if named != script:
        raise SystemExit(f"{args.config} is a config for {named}.py, not {script}.py")
    return load_file(args.config, schema, args.set)


def conditions() -> Dict[str, Dict[str, Any]]:
    """Every named condition (file stem -> its overrides of the default), sorted by name."""
    return {path.stem: _read_yaml(path) for path in sorted(CONDITIONS_DIR.glob("*.yaml"))}


def load_config(condition: Optional[str] = None, overrides: Sequence[str] = ()) -> Config:
    """Default config, then the named condition's file, then `section.key=value` overrides."""
    data = _read_yaml(RUN_CONFIG_DIR / "default.yaml")
    if condition:
        path = CONDITIONS_DIR / f"{condition}.yaml"
        if not path.is_file():
            raise ValueError(f"unknown condition {condition!r}; known: {', '.join(conditions())}")
        data = _merge(data, _read_yaml(path))
        data["condition"] = condition
    for text in overrides:
        data = _merge(data, _parse_override(text))
    cfg = _build(Config, _resolve_scheduler(data))
    validate(cfg)
    return cfg



def load_run_config(run_dir) -> Config:
    """The settings a finished run recorded in its config.json, over the run defaults (keys the run did not
    record take their default values)."""
    recorded = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    cfg = _build(Config, _resolve_scheduler(_merge(_read_yaml(RUN_CONFIG_DIR / "default.yaml"), recorded)))
    validate(cfg)
    return cfg


def _resolve_scheduler(data: Dict[str, Any]) -> Dict[str, Any]:
    """`run.scheduler: null` is the event scheduler for symbolic runs and threads for legacy runs (the only
    scheduler legacy has)."""
    if data["run"].get("scheduler") is None:
        data["run"]["scheduler"] = "threads" if data.get("approach") == "legacy" else "event"
    return data


def validate(cfg: Config) -> None:
    from core.backends import BACKENDS   # local imports: core depends on config, not the reverse
    from core.scheduler import SCHEDULERS
    from core.retry import RetryPolicy
    RetryPolicy.parse(cfg.planner.retry)
    RetryPolicy.parse(cfg.legacy.retry)
    checks = {
        "approach": (cfg.approach, {"symbolic", "legacy"}),
        "planner.branch": (cfg.planner.branch, {"joint", "per_requirement"}),
        "planner.feedback": (cfg.planner.feedback, {"blame", "table"}),
        "feedback.blame": (cfg.feedback.blame, {"mass", "regret", "random", "none"}),
        "feedback.space": (cfg.feedback.space, {"states", "features"}),
        "rules.any_order": (cfg.rules.any_order, {"none", "rotations", "all", "declared", "report"}),
        "rules.any_scorer": (cfg.rules.any_scorer, {"prism", "domain"}),
        "llm.backend": (cfg.llm.backend, set(BACKENDS)),
        "run.scheduler": (cfg.run.scheduler, {"threads", *SCHEDULERS}),
    }
    for key, (value, allowed) in checks.items():
        if value not in allowed:
            raise ValueError(f"{key}={value!r} is not supported (allowed: {', '.join(sorted(allowed))})")
    horizon = cfg.feedback.horizon
    if horizon != "domain" and not (isinstance(horizon, int) and not isinstance(horizon, bool) and horizon >= 1):
        raise ValueError(f"feedback.horizon={horizon!r} must be a number of steps (>= 1) or \"domain\"")
    for key in ("providers", "quantizations"):
        value = getattr(cfg.llm.openrouter, key)
        if not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
            raise ValueError(f"llm.openrouter.{key}={value!r} must be a list of names, e.g. [deepinfra]")
    limit = cfg.prism.max_concurrent
    if limit is not None and not (isinstance(limit, int) and not isinstance(limit, bool) and limit >= 1):
        raise ValueError(f"prism.max_concurrent={limit!r} must be a number of processes (>= 1) or null")
    if cfg.prism.exact_max_iters < 1:
        raise ValueError(f"prism.exact_max_iters={cfg.prism.exact_max_iters!r} must be >= 1")
    if cfg.approach == "legacy" and cfg.run.scheduler != "threads":
        raise ValueError(f"run.scheduler={cfg.run.scheduler} drives the symbolic planner only; legacy runs use threads")
