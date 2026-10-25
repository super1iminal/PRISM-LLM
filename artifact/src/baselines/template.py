"""The hand-written gridworld template (domains/gridworld/data/template_policy.py): one policy over relative
features, compiled on each grid to rules over x, y and the goal flags. Not fitted. Needs a condition without
`rules.general` (the compiled rules name cells).

params:
  variants: [greedy, cautious]   the template's variants to certify, one source each
"""
import importlib.util
from typing import Dict

from baselines.base import Baseline, Rules
from core.domain import Instance
from settings import DOMAINS_PATH


def _template_module():
    path = DOMAINS_PATH / "gridworld" / "data" / "template_policy.py"
    spec = importlib.util.spec_from_file_location("gridworld_template_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Template(Baseline):

    def __init__(self, domain, params):
        super().__init__(domain, params)
        self.module = _template_module()
        self.variants = params.get("variants") or list(self.module.VARIANTS)

    def rules(self, target: Instance) -> Dict[str, Rules]:
        return {v: self.module.template_rules(target.data, v) for v in self.variants}
