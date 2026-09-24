"""UUV case study: pipeline inspection by an underwater vehicle that adapts its search altitude.

Based on Paessler et al., "Formal Modelling and Analysis of a Self-Adaptive Robotic System"
(iFM 2023, arXiv:2308.14663). The MDP re-expresses the paper's ProFeat model (artifact branch
`scp-ifm_artifact` of github.com/remaro-network/auv_profeat) in plain PRISM. The policy plays the
paper's managing subsystem: while searching, it picks the altitude (low, med, high) that the water
visibility allows. Everything else is forced, exactly as in the paper.

Dataset rows (CSV): name, min_visib, max_visib, current_prob, inspect, deadline,
safe_threshold, on_time_threshold.
"""
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from core.domain import Domain, Instance

INFL_TF = 10   # influence of thruster failures on losing the pipeline (same in both paper scenarios)

# Altitude code -> (name, search state, recovery state, P(found), P(stay searching), P(thruster failure))
ALTITUDES = [
    {"code": 0, "name": "low", "search": "search_low", "recover": "recover_low",
     "p_found": 0.42, "p_stay": 0.55, "p_fail": 0.03},
    {"code": 1, "name": "med", "search": "search_med", "recover": "recover_med",
     "p_found": 0.48, "p_stay": 0.5, "p_fail": 0.02},
    {"code": 2, "name": "high", "search": "search_high", "recover": "recover_high",
     "p_found": 0.59, "p_stay": 0.4, "p_fail": 0.01},
]

# State numbering as in the paper's artifact
STATES = {
    "start_search": 0, "search_high": 1, "search_med": 2, "search_low": 3, "found": 4,
    "following": 5, "recover_high": 6, "recover_med": 7, "recover_low": 8, "recover_following": 9,
    "done": 10, "start_task": 11, "lost_pipe": 12,
}

FOLLOW = {"p_follow": 0.92, "p_lost": 0.05, "p_fail": 0.03}


class UUV(Domain):

    def load_instances(self, dataset: str) -> List[Instance]:
        path = Path(dataset) if Path(dataset).is_absolute() else self.root / "data" / dataset
        df = pd.read_csv(path, index_col=False)
        instances = []
        for idx, row in df.iterrows():
            instances.append(Instance(id=str(idx), data={
                "name": str(row["name"]),
                "min_visib": int(row["min_visib"]),
                "max_visib": int(row["max_visib"]),
                "current_prob": float(row["current_prob"]),
                "inspect": int(row["inspect"]),
                "deadline": int(row["deadline"]),
                "safe_threshold": float(row["safe_threshold"]),
                "on_time_threshold": float(row["on_time_threshold"]),
            }))
        return instances

    def context(self, instance: Instance) -> Dict[str, Any]:
        d = dict(instance.data)
        span = d["max_visib"] - d["min_visib"]
        # Same thresholds as the paper's formulas med_visib = span/3, high_visib = 2*span/3 (real division)
        med_visib, high_visib = span / 3, 2 * span / 3
        visib = list(range(d["min_visib"], d["max_visib"] + 1))
        d.update({
            "infl_tf": INFL_TF,
            "altitudes": ALTITUDES,
            "states": STATES,
            "follow": FOLLOW,
            "med_visib": med_visib,
            "high_visib": high_visib,
            "init_visib": int(span / 2 + 0.5),   # PRISM's round((max_visib-min_visib)/2)
            "visib_bands": [
                {"name": "poor", "levels": [v for v in visib if v < med_visib], "allowed": ["low"]},
                {"name": "average", "levels": [v for v in visib if med_visib <= v < high_visib],
                 "allowed": ["low", "med"]},
                {"name": "good", "levels": [v for v in visib if v >= high_visib], "allowed": ["low", "med", "high"]},
            ],
            "p_visib_down": d["current_prob"],
            "p_visib_up": (1 - d["current_prob"]) / 2,
        })
        return d
