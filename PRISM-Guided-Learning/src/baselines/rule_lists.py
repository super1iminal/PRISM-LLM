"""Fixed rule lists, the same on every target: hand-written lists, rule sets taken from runs, or lists another
tool produced (e.g. a search over the rule language). Not fitted.

params:
  lists: {name: [[condition, action], ...]}   inline lists, and/or
  files: [path, ...]                          JSON or YAML files of the same shape (relative to PRISM-Guided-Learning/)
"""
import json
from pathlib import Path
from typing import Dict

import yaml

from baselines.base import Baseline, Rules
from core.domain import Instance
from settings import PROJECT_ROOT


def read_lists(path: Path) -> Dict[str, Rules]:
    """name -> rules from a JSON or YAML file of {name: [[condition, action], ...]}."""
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    return {str(name): [tuple(rule) for rule in rules] for name, rules in data.items()}


class RuleLists(Baseline):

    def __init__(self, domain, params):
        super().__init__(domain, params)
        self.lists: Dict[str, Rules] = {str(k): [tuple(r) for r in v] for k, v in (params.get("lists") or {}).items()}
        for path in params.get("files") or []:
            self.lists.update(read_lists(PROJECT_ROOT / path))
        if not self.lists:
            raise ValueError("rule_lists needs params.lists or params.files")

    def rules(self, target: Instance) -> Dict[str, Rules]:
        return self.lists
