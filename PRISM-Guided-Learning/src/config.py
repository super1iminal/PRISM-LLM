"""Run configuration: one schema, defaults in configs/default.yaml, named conditions in configs/conditions.yaml.

    cfg = load_config("R2", overrides=["llm.seed=1", "run.limit=1"])

A condition is a partial override of the default; CLI overrides (`section.key=value`, value parsed
as YAML) apply last. Unknown sections or keys are errors, so a typo cannot silently fall back to
a default. `docs/config.md` documents every key.
"""
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Sequence

import yaml

from settings import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "configs"


@dataclass
class DomainConfig:
    name: str = "gridworld"
    dataset: str = "grid_20_balanced.csv"
    visible_extra: List[str] = field(default_factory=list)   # extra state variables rules may read, if in the model


@dataclass
class LLMConfig:
    model: str = "qwen3:14b-q4_K_M"
    think: bool = False
    num_ctx: int = 16384
    num_predict: int = 8192
    seed: Optional[int] = None
    temperature: Optional[float] = None       # None = the model's default


@dataclass
class PlannerConfig:
    max_rounds: int = 5
    max_fixups: int = 2                       # extra calls per round when the answer has invalid rules
    retry: str = "gain:0.05"                  # stall:k | never | every:k | gain:eps | always
    branch: str = "joint"                     # joint | per_requirement
    feedback: str = "blame"                   # blame (REFINE/EXTEND with blame) | table (S1: results table only)
    max_rules: int = 64
    max_condition_chars: int = 200


@dataclass
class FeedbackConfig:
    blame: str = "mass"                       # mass | regret (S5) | random (V1) | none (V2)
    horizon: int = 100                        # steps for the occupancy in the mass analysis
    horizon_by_domain: Dict[str, Any] = field(default_factory=dict)   # steps, or "domain" (Domain.horizon)
    top_k: int = 10                           # hotspots / rules shown in feedback
    states_per_rule: int = 3

    def horizon_for(self, domain, instance=None) -> int:
        """Occupancy horizon for `domain` (a Domain, or its name). The value "domain" asks the domain
        (e.g. UUV's mission deadline), falling back to `horizon`."""
        name = getattr(domain, "name", domain)
        value = self.horizon_by_domain.get(name, self.horizon)
        if value == "domain":
            value = (domain.horizon(instance) if hasattr(domain, "horizon") and instance is not None else None)
            return int(value) if value else self.horizon
        return int(value)


@dataclass
class PromptConfig:
    catch_all_instruction: bool = True
    examples: bool = True


@dataclass
class PrismConfig:
    method: str = "gaussseidel"               # iterative method; plain value iteration oscillates on periodic chains
    fallback_methods: List[str] = field(default_factory=lambda: ["modpoliter"])   # tried when `method` does not converge
    java_max_mem: str = "4g"
    max_iters: int = 1_000_000
    timeout_s: float = 900
    multi_engine: str = "sparse"              # PRISM's explicit engine has no multi-objective support
    multi_method: str = "lp"                  # exact LP; value iteration fails to converge on periodic chains


@dataclass
class RulesConfig:
    max_enumeration: int = 200_000


@dataclass
class LegacyConfig:
    max_rounds: int = 5
    retry: str = "never"                      # never | stall:k (L1: blind retry with the initial prompt)
    examples: bool = True                     # the two worked examples in the initial prompt (L2: off)


@dataclass
class RunConfig:
    workers: int = 2
    limit: Optional[int] = None


@dataclass
class Config:
    approach: str = "symbolic"                # symbolic | legacy
    domain: DomainConfig = field(default_factory=DomainConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    prism: PrismConfig = field(default_factory=PrismConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    legacy: LegacyConfig = field(default_factory=LegacyConfig)
    run: RunConfig = field(default_factory=RunConfig)
    condition: Optional[str] = None           # name of the applied condition, for the record

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (update or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _build(cls, data: Dict[str, Any], path: str = ""):
    """Instantiate dataclass `cls` from `data`, rejecting unknown keys."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"unknown config key(s) {sorted(path + k for k in unknown)}")
    kwargs = {}
    for name, value in data.items():
        default = known[name].default_factory() if callable(known[name].default_factory) else None
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[name] = _build(type(default), value, f"{path}{name}.")
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


def conditions() -> Dict[str, Dict[str, Any]]:
    with open(CONFIG_DIR / "conditions.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(condition: Optional[str] = None, overrides: Sequence[str] = ()) -> Config:
    """Default config, then the named condition, then `section.key=value` overrides."""
    with open(CONFIG_DIR / "default.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if condition:
        known = conditions()
        if condition not in known:
            raise ValueError(f"unknown condition {condition!r}; known: {', '.join(known)}")
        data = _merge(data, known[condition] or {})
        data["condition"] = condition
    for text in overrides:
        data = _merge(data, _parse_override(text))
    cfg = _build(Config, data)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    from core.retry import RetryPolicy  # local import: core depends on config, not the reverse
    RetryPolicy.parse(cfg.planner.retry)
    RetryPolicy.parse(cfg.legacy.retry)
    checks = {
        "approach": (cfg.approach, {"symbolic", "legacy"}),
        "planner.branch": (cfg.planner.branch, {"joint", "per_requirement"}),
        "planner.feedback": (cfg.planner.feedback, {"blame", "table"}),
        "feedback.blame": (cfg.feedback.blame, {"mass", "regret", "random", "none"}),
    }
    for key, (value, allowed) in checks.items():
        if value not in allowed:
            raise ValueError(f"{key}={value!r} is not supported (allowed: {', '.join(sorted(allowed))})")
