"""Baselines for the transfer search (docs/transfer_plan.md): one module per method, each a `Baseline`
(base.py), run from configs/baselines/<name>.yaml by src/run_baseline.py."""
from importlib import import_module

from baselines.base import Baseline

METHODS = {
    "rule_lists": "baselines.rule_lists:RuleLists",
    "template": "baselines.template:Template",
    "feature_search": "baselines.feature_search:FeatureSearch",
    "decision_tree": "baselines.decision_tree:DecisionTree",
    "ppo": "baselines.ppo:PPO",
}


def load_method(name: str):
    """The Baseline subclass registered under `name`."""
    if name not in METHODS:
        raise ValueError(f"unknown baseline method {name!r}; known: {', '.join(METHODS)}")
    module, cls = METHODS[name].split(":")
    return getattr(import_module(module), cls)


__all__ = ["Baseline", "METHODS", "load_method"]
