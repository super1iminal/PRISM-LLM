"""Mars Exploration Rovers (MER): schedule a resource arbiter and its two users so that both resources end up granted.

The MDP is the MER case study (Feng, Kwiatkowska and Parker, FASE 2011) as 1-2-3-Go! uses it (Azeem et al., VMCAI 2025,
arXiv:2410.18293): an arbiter shares communication and driving between two users. Every step of the system is a
policy action, so the policy is a scheduler, as in 1-2-3-Go!. The requirement asks for a high probability of the
monitored error (both resources granted at once), 1-2-3-Go!'s property `p1` (Pmax). Instances vary the probability
`x` that a user keeps a resource after using it (n = 10), the parameter on which 1-2-3-Go! fails to generalize, and
`n`, how many of user 1's comm requests while drive is held may take the arbiter's faulty shortcut.

Dataset rows (CSV): name, n, x, optimum (Pmax from PRISM), err_threshold (the optimum minus 5% of it).
"""
import re
from pathlib import Path
from typing import Any, Dict, List

from core.domain import Domain, Instance

USERS = (1, 2)
RESOURCES = {1: "comm", 2: "drive"}


class Mer(Domain):

    def load_instances(self, dataset: str) -> List[Instance]:
        df = self.read_dataset(dataset)
        return [Instance(id=str(idx), data={"name": str(row["name"]), "n": int(row["n"]), "x": float(row["x"]),
                                            "optimum": float(row["optimum"]),
                                            "err_threshold": float(row["err_threshold"])})
                for idx, row in df.iterrows()]

    def context(self, instance: Instance) -> Dict[str, Any]:
        d = dict(instance.data)
        d.update({"resources": RESOURCES, **rule_vocabulary(d["n"], d["x"], self.root)})
        return d


# ---------------------------------------------------------------- extended rule vocabulary

_ENABLED_RE = re.compile(r"^formula en_(\w+) = (.+);$", re.M)


def enabled_conditions(root: Path) -> Dict[str, str]:
    """Action -> the condition under which it changes the state, read from the model's `en_<action>` formulas, with
    the rescind limit `n` written as the rule constant `N`."""
    text = (root / "model.prism.j2").read_text(encoding="utf-8")
    return {a: re.sub(r"\bn\b", "N", expr) for a, expr in _ENABLED_RE.findall(text)}


def rule_vocabulary(n: int, x: float, root: Path) -> Dict[str, Any]:
    """Constants N and X_PCT, the state feature shortcuts_left and the per-action feature enabled."""
    features = [
        {"name": "shortcuts_left", "description": "comm requests of user 1 while drive is held that can still take "
                                                  "the arbiter's shortcut (N - k)",
         "expr": "N - k"},
        {"name": "enabled", "description": "true if this action is possible here (otherwise it leaves the state "
                                           "unchanged)",
         "per_action": enabled_conditions(root)},
    ]
    return {"rule_constants": [
                {"name": "N", "value": n, "description": "how many of user 1's comm requests while drive is held "
                                                         "may take the arbiter's shortcut"},
                {"name": "X_PCT", "value": round(100 * x),
                 "description": "the probability x that a user keeps a resource after using it, in percent"}],
            "rule_features": features}
