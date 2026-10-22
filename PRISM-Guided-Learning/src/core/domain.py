"""Case-study plumbing: a Domain turns dataset rows into fully specified planning problems.

A case study lives in `domains/<name>/` and consists of

    domain.py            a `Domain` subclass: loads instances and builds the template context
    model.prism.j2       the MDP in PRISM syntax. Every policy action must be an action label
                         ([up], [down], ...) on the commands it controls
    spec.yaml.j2         policy-visible variables, actions and requirements (see `Spec`), plus
                         optional_variables: descriptions for variables a config may make visible,
                         and optionally constants and features: the extended rule vocabulary
    description.md.j2    English description of the MDP for the prompt
    visual.txt.j2        visual representation of the state space for the prompt
    examples.md.j2       (optional) domain-specific rule examples for the prompt
    examples_extended.md.j2
                         (optional) examples for runs with the extended rule vocabulary
    initial.md.j2, refine.md.j2, extend.md.j2
                         (optional) override the core prompt templates in src/core/templates

All templates are Jinja2 and are rendered with `Domain.context(instance)`.
"""
import importlib.util
import inspect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import jinja2
import pandas as pd
import yaml

from core.rules import Constant, Feature, SymbolicPolicy, Value, Variable, Vocabulary
from settings import DOMAINS_PATH

CORE_TEMPLATES = Path(__file__).resolve().parent / "templates"


@dataclass
class Requirement:
    """A bound on a probability (`P⋈θ [formula]`) or, with `reward`, on an expected reward
    (`R{"reward"}⋈θ [formula]`, e.g. expected energy until `F "done"`)."""
    name: str
    formula: str              # PRISM path formula, e.g.  F "at_goal1"
    threshold: float
    bound: str = ">="         # ">=": value must be at least threshold, "<=": at most
    description: str = ""
    reward: Optional[str] = None   # reward structure name; None for a probability requirement

    @property
    def maximize(self) -> bool:
        return self.bound == ">="

    def satisfied(self, value: float) -> bool:
        return value >= self.threshold if self.maximize else value <= self.threshold

    def shortfall(self, value: float) -> float:
        """How far `value` is from satisfying the requirement (0 if satisfied). Reward requirements
        use the gap relative to the threshold, so they are on the same scale as probabilities."""
        gap = self.threshold - value if self.maximize else value - self.threshold
        if self.reward is not None:
            gap /= abs(self.threshold) or 1.0
        return max(0.0, gap)

    def margin(self, value: float) -> float:
        """How far `value` is on the passing side of the threshold (negative: the shortfall), on the scale of
        `shortfall`."""
        gap = value - self.threshold if self.maximize else self.threshold - value
        return gap / (abs(self.threshold) or 1.0) if self.reward is not None else gap

    def _op(self, best: bool) -> str:
        high = self.maximize == best
        if self.reward is not None:
            return f'R{{"{self.reward}"}}{"max" if high else "min"}'
        return "Pmax" if high else "Pmin"

    def best_op(self) -> str:
        return self._op(best=True)

    def worst_op(self) -> str:
        return self._op(best=False)

    def bounded(self) -> str:
        """The requirement as a bounded PRISM property, e.g. `P>=0.8 [ F "goal" ]`."""
        op = f'R{{"{self.reward}"}}' if self.reward is not None else "P"
        return f"{op}{self.bound}{self.threshold} [ {self.formula} ]"

    def bound_text(self) -> str:
        """How the bound reads in the prompt."""
        what = "Probability" if self.reward is None else f"Expected {self.reward}"
        return f"{what} must be {self.bound} {self.threshold}"


def failing(requirements: Sequence[Requirement], values: Dict[str, float]) -> List[Requirement]:
    """The requirements whose value in `values` (requirement name -> value) misses the threshold."""
    return [r for r in requirements if not r.satisfied(values[r.name])]


@dataclass
class Spec:
    variables: List[Variable]          # policy-visible state variables
    actions: Dict[str, str]            # action label -> English description
    requirements: List[Requirement]
    constants: List[Constant] = field(default_factory=list)   # named instance values rules may use
    features: List[Feature] = field(default_factory=list)     # conditions the domain computes for rules

    def vocabulary(self, rules, literal_vars: Optional[Sequence[str]] = None) -> Vocabulary:
        """The vocabulary a run's `rules` config allows: with `rules.extended`, this instance's constants and
        features (minus `rules.hidden_features`; optional ones only if named in `rules.extra_features`) and `any`
        rules; otherwise the base language. With `rules.shared_literals`, numbers may also be compared with
        `literal_vars` (the planner passes the variables whose range is the same on every training member; by
        default every variable, for checking frozen rules)."""
        if not rules.extended:
            return Vocabulary()
        features = [f for f in self.features if f.name not in rules.hidden_features
                    and (not f.optional or f.name in rules.extra_features)]
        shared = (tuple(literal_vars if literal_vars is not None else [v.name for v in self.variables])
                  if rules.general and getattr(rules, "shared_literals", False) else ())
        return Vocabulary(list(self.constants), features, allow_any=True, general=rules.general, literal_vars=shared)

    def policy(self, rules: Iterable[Tuple[str, str]], vocabulary: Optional[Vocabulary] = None) -> SymbolicPolicy:
        """`(condition, action)` pairs parsed into a policy over these variables and actions, with `vocabulary`'s
        names if given; raises RuleError listing every bad rule."""
        return SymbolicPolicy.from_raw(self.variables, list(self.actions), rules, vocabulary)


@dataclass
class Instance:
    id: str
    data: Dict[str, Any] = field(default_factory=dict)


class Domain:
    """Base class for case studies. Subclasses implement `load_instances` and usually `context`."""

    def __init__(self, root: Path, visible_extra: Sequence[str] = ()):
        self.root = Path(root)
        self.name = self.root.name
        # Extra state variables rules may read (config `domain.visible_extra`), on top of the spec's
        # `variables`. Ignored for instances whose model lacks them. Templates see them as `visible_extra`.
        self.visible_extra = list(visible_extra)
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader([str(self.root), str(CORE_TEMPLATES)]),
            undefined=jinja2.StrictUndefined, trim_blocks=True, lstrip_blocks=True,
            keep_trailing_newline=True, autoescape=False,
        )

    # ---------------------------------------------------------------- to implement

    def load_instances(self, dataset: str) -> List[Instance]:
        raise NotImplementedError

    def context(self, instance: Instance) -> Dict[str, Any]:
        return dict(instance.data)

    def horizon(self, instance: Instance) -> Optional[int]:
        """Optional occupancy horizon for the mass analysis (used when the config asks the domain)."""
        return None

    def rule_scorer(self, instance: Instance) -> Optional[Any]:
        """Optional fast model of `instance` that scores rule lists without PRISM (`rules.any_scorer: domain`): its
        `order_values(policy, orders)` gives, for each action order, the (best, worst) values of `policy` with its
        `any` rules resolved in that order. None if the domain has none."""
        return None

    def strict(self, instance: Instance) -> Dict[str, Any]:
        """What `rules.strict` withholds from conditions: `hide`, names of policy variables and constants that
        conditions may not use (and the prompt leaves out); `literal_sums`, whether number literals may be added or
        subtracted (`1 + 1`); `note`, a sentence for the prompt saying what is withheld and why. By default nothing."""
        return {"hide": [], "literal_sums": True, "note": ""}

    def objective(self, instance: Instance) -> Optional[Tuple[str, Requirement]]:
        """Optional single objective for the baselines that learn one requirement's optimal actions on the explicit
        model (`baselines/explicit_mdp.py`, which handles `F` / `U` over labels only), for domains whose requirements
        are not of that form: a PRISM model of the instance (the domain's model plus whatever the objective needs,
        such as a step counter) and the requirement on it. The policy-visible variables stay the spec's. None: the
        baselines take one of the spec's own requirements (the default)."""
        return None

    def fast_checker(self, instance: Instance, config, vocabulary) -> Optional[Any]:
        """Optional fast exact checker of `instance` for rule lists parsed with `vocabulary` (`planner.repair`,
        `planner.pick: candidates`): its `worst(rules, order=None)` gives each requirement's worst-case value of the
        `(condition, action)` pairs (with `any` taking the first allowed action of `order` if given), and raises
        `core.rules.RuleError` if they do not parse. None if the domain has none."""
        return None

    def repair_literals(self, instance: Instance, vocabulary) -> List[str]:
        """Conjuncts `planner.repair` may add to conditions: by default each boolean policy variable and each feature
        of `vocabulary` that is boolean, and their negations (domains add comparisons of their own)."""
        spec = self.spec(instance)
        names = [v.name for v in spec.variables if v.type == "bool"] + [f.name for f in vocabulary.features
                                                                         if f.is_action_feature]
        return [lit for n in names for lit in (n, f"!{n}")]

    def worst_case_notes(self, instance: Instance) -> str:
        """The domain's own notes on the worst case for `prompt.worst_case` (worst_case.md.j2), or ''."""
        return self.render("worst_case.md.j2", instance).strip() if self.has_template("worst_case.md.j2") else ""

    def met(self, instance: Instance, requirement: Requirement) -> Optional[Callable[[Dict[str, Any]], bool]]:
        """Optional test of a state's variables showing `requirement` already met (`feedback.history`): a formula over
        events (goal cells) is checked from every state anew, so a state after it was met can read as lost there. None
        where the variables do not record it (the default)."""
        return None

    def state_words(self, instance: Instance, valuation: Dict[str, Any]) -> Optional[str]:
        """Optional plain description of a state for the failure feedback (`feedback.words`); None: `format_state`."""
        return None

    def loss_words(self, requirement: Requirement) -> Optional[str]:
        """Optional words for losing `requirement` in one step, for the failure feedback's risks (`feedback.words`);
        None: "loses <name>"."""
        return None

    def optimal_strategy(self, instance: Instance, config) -> Optional[Any]:
        """Optional optimal strategy of `instance` (`feedback.optimal`, `prompt.demonstration`; core/optimal.py): a
        function from a state (every model variable) to its optimal actions and chance of success. None if the
        domain has none."""
        return None

    # ---------------------------------------------------------------- the domain's data files

    def read_dataset(self, dataset: str) -> pd.DataFrame:
        """A dataset's rows: a CSV file, given by absolute path or by name in `domains/<name>/data/`."""
        path = Path(dataset) if Path(dataset).is_absolute() else self.root / "data" / dataset
        return pd.read_csv(path, index_col=False)

    def reference_policies(self) -> Dict[str, List[Tuple[str, str]]]:
        """Name -> `(condition, action)` rules of the domain's hand-written reference policies
        (`data/reference_policies.json`); {} if it has none."""
        path = self.root / "data" / "reference_policies.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    # ---------------------------------------------------------------- template-backed defaults

    def render(self, template: str, instance: Optional[Instance] = None, **extra) -> str:
        ctx = self.context(instance) if instance is not None else {}
        ctx.setdefault("visible_extra", self.visible_extra)
        ctx.update(extra)
        return self.env.get_template(template).render(**ctx)

    def has_template(self, template: str) -> bool:
        return (self.root / template).exists() or (CORE_TEMPLATES / template).exists()

    def model(self, instance: Instance) -> str:
        return self.render("model.prism.j2", instance)

    def description(self, instance: Instance) -> str:
        return self.render("description.md.j2", instance).strip()

    def visual(self, instance: Instance) -> str:
        return self.render("visual.txt.j2", instance).rstrip()

    def examples(self, instance: Instance, extended: bool = False) -> str:
        """The domain's example rules; with `extended`, its examples_extended.md.j2 if it has one."""
        for template in (["examples_extended.md.j2"] if extended else []) + ["examples.md.j2"]:
            if self.has_template(template):
                return self.render(template, instance).strip()
        return ""

    def spec(self, instance: Instance) -> Spec:
        raw = yaml.safe_load(self.render("spec.yaml.j2", instance))
        declared = parse_model_variables(self.model(instance))
        variables = []
        optional = {v["name"]: v for v in raw.get("optional_variables") or []}
        listed = [({"name": v} if isinstance(v, str) else v) for v in raw["variables"]]
        names = {v["name"] for v in listed}
        listed += [optional.get(name, {"name": name}) for name in self.visible_extra
                   if name in declared and name not in names]
        for v in listed:
            name = v["name"]
            if "type" in v:
                var = Variable(name, v["type"], v.get("low", 0), v.get("high", 1), v.get("description", ""))
            elif name in declared:
                d = declared[name]
                var = Variable(name, d.type, d.low, d.high, v.get("description", ""))
            else:
                raise ValueError(f"policy variable {name!r} is not declared in the model; give its type/range in the spec")
            variables.append(var)
        requirements = [Requirement(r["name"], r["formula"], float(r["threshold"]), r.get("bound", ">="),
                                    r.get("description", ""), r.get("reward")) for r in raw["requirements"]]
        constants = [Constant(c["name"], int(c["value"]), c.get("description", "")) for c in raw.get("constants") or []]
        features = [Feature(f["name"], f.get("description", ""), f.get("expr"), f.get("per_action"),
                            bool(f.get("optional", False))) for f in raw.get("features") or []]
        return Spec(variables, dict(raw["actions"]), requirements, constants, features)

    def format_state(self, valuation: Dict[str, Value]) -> str:
        """How a (policy-variable) state is shown to the LLM; by default in rule syntax."""
        return " & ".join(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in valuation.items())


def load_domain(name: str, visible_extra: Sequence[str] = ()) -> Domain:
    """Load `domains/<name>/domain.py` and instantiate its Domain subclass."""
    root = DOMAINS_PATH / name
    spec = importlib.util.spec_from_file_location(f"domains.{name}.domain", root / "domain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    classes = [c for _, c in inspect.getmembers(module, inspect.isclass)
               if issubclass(c, Domain) and c is not Domain and c.__module__ == module.__name__]
    if len(classes) != 1:
        raise ValueError(f"{root / 'domain.py'} must define exactly one Domain subclass")
    return classes[0](root, visible_extra)


# ---------------------------------------------------------------- model parsing helpers

_CONST_RE = re.compile(r"^\s*const\s+(?:int\s+)?(\w+)\s*=\s*([^;]+);", re.M)   # untyped constants are int
_INT_VAR_RE = re.compile(r"^\s*(\w+)\s*:\s*\[([^\]]+?)\.\.([^\]]+?)\]", re.M)
_BOOL_VAR_RE = re.compile(r"^\s*(\w+)\s*:\s*bool\b", re.M)


def _eval_int(expr: str, consts: Dict[str, int]) -> int:
    expr = expr.strip()
    if not re.fullmatch(r"[\w\s+\-*/()]+", expr):
        raise ValueError(f"cannot evaluate bound {expr!r}")
    return int(eval(expr, {"__builtins__": {}}, dict(consts)))


def parse_model_variables(model: str) -> Dict[str, Variable]:
    """State variables declared in a PRISM model (int ranges resolved through `const int`)."""
    consts: Dict[str, int] = {}
    for name, expr in _CONST_RE.findall(model):
        consts[name] = _eval_int(expr, consts)
    variables = {}
    for name, low, high in _INT_VAR_RE.findall(model):
        variables[name] = Variable(name, "int", _eval_int(low, consts), _eval_int(high, consts))
    for name in _BOOL_VAR_RE.findall(model):
        variables[name] = Variable(name, "bool")
    return variables
