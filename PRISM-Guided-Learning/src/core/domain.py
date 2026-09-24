"""Case-study plumbing: a Domain turns dataset rows into fully specified planning problems.

A case study lives in `domains/<name>/` and consists of

    domain.py            a `Domain` subclass: loads instances and builds the template context
    model.prism.j2       the MDP in PRISM syntax. Every policy action must be an action label
                         ([up], [down], ...) on the commands it controls
    spec.yaml.j2         policy-visible variables, actions and requirements (see `Spec`)
    description.md.j2    English description of the MDP for the prompt
    visual.txt.j2        visual representation of the state space for the prompt
    examples.md.j2       (optional) domain-specific rule examples for the prompt
    initial.md.j2, refine.md.j2, extend.md.j2
                         (optional) override the core prompt templates in src/core/templates

All templates are Jinja2 and are rendered with `Domain.context(instance)`.
"""
import importlib.util
import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import jinja2
import yaml

from core.rules import Value, Variable
from settings import DOMAINS_PATH

CORE_TEMPLATES = Path(__file__).resolve().parent / "templates"


@dataclass
class Requirement:
    name: str
    formula: str              # PRISM path formula, e.g.  F "at_goal1"
    threshold: float
    bound: str = ">="         # ">=": probability must be at least threshold, "<=": at most
    description: str = ""

    @property
    def maximize(self) -> bool:
        return self.bound == ">="

    def satisfied(self, probability: float) -> bool:
        return probability >= self.threshold if self.maximize else probability <= self.threshold

    def shortfall(self, probability: float) -> float:
        """How far `probability` is from satisfying the requirement (0 if satisfied)."""
        gap = self.threshold - probability if self.maximize else probability - self.threshold
        return max(0.0, gap)

    def best_op(self) -> str:
        return "Pmax" if self.maximize else "Pmin"

    def worst_op(self) -> str:
        return "Pmin" if self.maximize else "Pmax"


@dataclass
class Spec:
    variables: List[Variable]          # policy-visible state variables
    actions: Dict[str, str]            # action label -> English description
    requirements: List[Requirement]


@dataclass
class Instance:
    id: str
    data: Dict[str, Any] = field(default_factory=dict)


class Domain:
    """Base class for case studies. Subclasses implement `load_instances` and usually `context`."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.name = self.root.name
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

    # ---------------------------------------------------------------- template-backed defaults

    def render(self, template: str, instance: Optional[Instance] = None, **extra) -> str:
        ctx = self.context(instance) if instance is not None else {}
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

    def examples(self, instance: Instance) -> str:
        return self.render("examples.md.j2", instance).strip() if self.has_template("examples.md.j2") else ""

    def spec(self, instance: Instance) -> Spec:
        raw = yaml.safe_load(self.render("spec.yaml.j2", instance))
        declared = parse_model_variables(self.model(instance))
        variables = []
        for v in raw["variables"]:
            v = {"name": v} if isinstance(v, str) else v
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
                                    r.get("description", "")) for r in raw["requirements"]]
        return Spec(variables, dict(raw["actions"]), requirements)

    def format_state(self, valuation: Dict[str, Value]) -> str:
        """How a (policy-variable) state is shown to the LLM; by default in rule syntax."""
        return " & ".join(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in valuation.items())


def load_domain(name: str) -> Domain:
    """Load `domains/<name>/domain.py` and instantiate its Domain subclass."""
    root = DOMAINS_PATH / name
    spec = importlib.util.spec_from_file_location(f"domains.{name}.domain", root / "domain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    classes = [c for _, c in inspect.getmembers(module, inspect.isclass)
               if issubclass(c, Domain) and c is not Domain and c.__module__ == module.__name__]
    if len(classes) != 1:
        raise ValueError(f"{root / 'domain.py'} must define exactly one Domain subclass")
    return classes[0](root)


# ---------------------------------------------------------------- model parsing helpers

_CONST_RE = re.compile(r"^\s*const\s+int\s+(\w+)\s*=\s*([^;]+);", re.M)
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
