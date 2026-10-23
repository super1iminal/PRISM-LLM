"""UUV case study: pipeline inspection by an underwater vehicle that adapts its search altitude.

Based on Paessler et al., "Formal Modelling and Analysis of a Self-Adaptive Robotic System"
(iFM 2023, arXiv:2308.14663). The MDP re-expresses the paper's ProFeat model (artifact branch
`scp-ifm_artifact` of github.com/remaro-network/auv_profeat) in plain PRISM. The policy plays the
paper's managing subsystem: while searching, it picks the altitude (low, med, high) that the water
visibility allows. Everything else is forced, exactly as in the paper.

Dataset rows (CSV): name, min_visib, max_visib, current_prob, inspect, deadline,
safe_threshold, on_time_threshold, and optionally energy_threshold and fail_scale (every thruster-failure
probability times this factor; default 1, the paper's values).
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.domain import Domain, Instance, Requirement

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

# Mission states in words (`feedback.words`), from the model's comments and the description's state list
PHASE_WORDS = {
    "start_task": "starting a task (the first search altitude is chosen here)",
    "start_search": "starting a search", "search_low": "searching at low altitude",
    "search_med": "searching at medium altitude", "search_high": "searching at high altitude",
    "found": "pipeline found (descending to follow it)", "following": "following the pipeline",
    "recover_low": "recovering a failed thruster while searching low",
    "recover_med": "recovering a failed thruster while searching at medium altitude",
    "recover_high": "recovering a failed thruster while searching high",
    "recover_following": "recovering a failed thruster while following", "done": "inspection done",
    "lost_pipe": "pipeline lost (searching again from low altitude)",
}


def scaled_failures(rows: List[Dict[str, Any]], absorbing: str, k: float) -> List[Dict[str, Any]]:
    """`rows` with every `p_fail` times `k`; the probability `absorbing` (staying in the same state) takes up the
    difference. k = 1 returns the rows unchanged."""
    if k == 1:
        return rows
    out = [dict(r) for r in rows]
    for r in out:
        r[absorbing] = round(r[absorbing] - (k - 1) * r["p_fail"], 10)
        r["p_fail"] = round(k * r["p_fail"], 10)
        if r[absorbing] < 0:
            raise ValueError(f"fail_scale {k} makes {absorbing} negative")
    return out


class UUV(Domain):

    def load_instances(self, dataset: str) -> List[Instance]:
        df = self.read_dataset(dataset)
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
                # Optional expected-energy budget (blank or missing column: no energy requirement)
                "energy_threshold": (float(row["energy_threshold"]) if "energy_threshold" in row
                                     and pd.notna(row["energy_threshold"]) else None),
                "fail_scale": (float(row["fail_scale"]) if "fail_scale" in row
                               and pd.notna(row["fail_scale"]) else 1.0),
            }))
        return instances

    def horizon(self, instance: Instance) -> int:
        """Mass-analysis horizon: the mission deadline (used when the run config sets `feedback.horizon: domain`)."""
        return instance.data["deadline"]

    def context(self, instance: Instance) -> Dict[str, Any]:
        d = dict(instance.data)
        span = d["max_visib"] - d["min_visib"]
        # Same thresholds as the paper's formulas med_visib = span/3, high_visib = 2*span/3 (real division)
        med_visib, high_visib = span / 3, 2 * span / 3
        visib = list(range(d["min_visib"], d["max_visib"] + 1))
        d.update({
            "infl_tf": INFL_TF,
            "altitudes": scaled_failures(ALTITUDES, "p_stay", d.get("fail_scale", 1.0)),
            "states": STATES,
            "follow": scaled_failures([FOLLOW], "p_follow", d.get("fail_scale", 1.0))[0],
            "init_visib": int(span / 2 + 0.5),   # PRISM's round((max_visib-min_visib)/2)
            "visib_bands": [
                {"name": "poor", "levels": [v for v in visib if v < med_visib], "allowed": ["low"]},
                {"name": "average", "levels": [v for v in visib if med_visib <= v < high_visib],
                 "allowed": ["low", "med"]},
                {"name": "good", "levels": [v for v in visib if v >= high_visib], "allowed": ["low", "med", "high"]},
            ],
            "p_visib_down": d["current_prob"],
            "p_visib_up": (1 - d["current_prob"]) / 2,
            **rule_vocabulary(d),
        })
        return d

    def objective(self, instance: Instance) -> Optional[Tuple[str, Requirement]]:
        """The baselines' objective (Rajan's checklist, item 8): finish the inspection within the deadline with no
        thruster failure, `F "goal"` on the product of the model with a step counter `t` (+1 on every transition,
        saturating at deadline + 1) and a flag `ok` (false once a thruster-failure state has been left), where
        "goal" = done & t <= deadline & ok. Since "done" is reached with probability 1 under every scheduler and no
        failure follows it, this is the probability of meeting both probability requirements on the same path."""
        return product_model(self.model(instance), instance.data["deadline"]), Requirement(
            name="goal", formula='F "goal"', threshold=0.0, bound=">=",
            description="the inspection is done within the deadline and no thruster ever fails")

    def fast_checker(self, instance: Instance, config, vocabulary):
        """The explicit model's numpy checker (baselines/explicit_mdp.RuleChecker), as for MER and Pac-Man: each
        requirement's worst case over the completions a rule list allows, `G` as one minus the opposite `F`, the
        deadline by backward induction, and expected energy until "done" from the model's state rewards. It lets the
        pick repair candidates and resolve `any` (Asher's UUV plan, step 2)."""
        from baselines.explicit_mdp import RuleChecker
        return RuleChecker(self, instance, config, vocabulary)

    def repair_literals(self, instance: Instance, vocabulary) -> List[str]:
        """The default (action features and their negations), the visibility bands and mission phases, and the
        altitude and decision states by their codes."""
        states = ["START_TASK", "START_SEARCH", "SEARCH_LOW", "SEARCH_MED", "SEARCH_HIGH", "RECOVER_LOW", "RECOVER_MED",
                  "RECOVER_HIGH"]
        return super().repair_literals(instance, vocabulary) + [
            lit for f in ("poor_visib", "average_visib", "good_visib", "starting", "searching", "recovering")
            for lit in (f, f"!{f}")] + [f"alt = {a}" for a in ("LOW", "MED", "HIGH")] \
            + [f"alt != {a}" for a in ("LOW", "MED", "HIGH")] + [f"s = {c}" for c in states]

    def state_words(self, instance: Instance, valuation: Dict[str, Any]) -> Optional[str]:
        """A state in the mission's words (`feedback.words`; Asher's UUV plan, step 5), from the model and spec alone:
        the mission phase, the active altitude, and the visibility with its band and what the band allows."""
        d = instance.data
        span = d["max_visib"] - d["min_visib"]
        first_avg, first_good = -(-span // 3), -(-2 * span // 3)
        names = {code: st for st, code in STATES.items()}
        phase = PHASE_WORDS.get(names.get(valuation.get("s")), f"mission state {valuation.get('s')}")
        alt = {a["code"]: a["name"] for a in ALTITUDES}.get(valuation.get("alt"), str(valuation.get("alt")))
        v = valuation.get("water_visib")
        band = ("poor (only low altitude allowed)" if v < first_avg else
                "average (low or medium allowed)" if v < first_good else "good (every altitude allowed)")
        return f"{phase}, active altitude {alt}, visibility {v} of {d['min_visib']}-{d['max_visib']}: {band}"

    def loss_words(self, requirement) -> Optional[str]:
        """What losing a requirement in one step means, for the failure feedback's risks (`feedback.words`)."""
        return {"no_thruster_failure": "a thruster failure", "done_in_time": "missing the deadline",
                "energy_budget": "exceeding the energy budget"}.get(requirement.name)

    def format_state(self, valuation: Dict[str, Any]) -> str:
        """Rule syntax, with the names of mission states and altitudes added."""
        names = {"s": {code: st for st, code in STATES.items()}, "alt": {a["code"]: a["name"] for a in ALTITUDES}}
        return " & ".join(f"{k}={v}" + (f" ({names[k][v]})" if k in names else "") for k, v in valuation.items())


LABELS = ("step", "low", "med", "high")
FAILURE = "(s=recover_high | s=recover_med | s=recover_low | s=recover_following)"


def product_model(model: str, deadline: int) -> str:
    """`model` with the module `stepcount` (counter `t`, flag `ok`; every command synchronizes on one of LABELS, so
    every transition counts) and the label "goal" (done within `deadline` steps and no thruster failure). Rewards are
    dropped: the objective is a probability."""
    m = re.sub(r'rewards "[^"]+".*?endrewards\n?', "", model, flags=re.S)
    counter = (f"\nmodule stepcount\n  t : [0..{deadline + 1}] init 0;\n  ok : bool init true;\n"
               + "".join(f"  [{a}] true -> (t'=min(t+1, {deadline + 1})) & (ok'=ok & !{FAILURE});\n" for a in LABELS)
               + "endmodule\n\n")
    i = m.index('label "done"')
    return m[:i] + counter + f'label "goal" = s=done & t<={deadline} & ok;\n' + m[i:]

# ---------------------------------------------------------------- extended rule vocabulary (Rajan's checklist, item 8)
# Written from the model and the requirements alone (model.prism.j2, spec.yaml.j2), before any held-out check: the
# model's codes and visibility bands as named constants (the general rule language allows only the numbers 0 and 1),
# the mission's size, and features for the band, the mission phase and what an action does to the altitude.

ALT_CODES = [("LOW", 0, "altitude low (values of alt)"), ("MED", 1, "altitude medium"), ("HIGH", 2, "altitude high")]


def rule_vocabulary(d: Dict[str, Any]) -> Dict[str, Any]:
    """Constants and features for `rules.extended`: the visibility limits and band edges (the model's med_visib =
    (max_visib - min_visib)/3 and high_visib = 2(max_visib - min_visib)/3, compared with water_visib, as the first
    integer level of each band), the inspection length and deadline, the mission-state and altitude codes; state
    features for the visibility band and the mission phase, and per-action features for keeping the altitude and for
    the altitude the visibility allows."""
    span = d["max_visib"] - d["min_visib"]
    first_avg, first_good = -(-span // 3), -(-2 * span // 3)          # ceil(span/3), ceil(2 span/3)
    poor, good = "water_visib < FIRST_AVG", "water_visib >= FIRST_GOOD"
    average = f"!({poor}) & !({good})"
    constants = [
        {"name": "MIN_VISIB", "value": d["min_visib"], "description": "the worst water visibility"},
        {"name": "MAX_VISIB", "value": d["max_visib"], "description": "the best water visibility"},
        {"name": "FIRST_AVG", "value": first_avg,
         "description": "the lowest visibility level of the average band (medium altitude allowed)"},
        {"name": "FIRST_GOOD", "value": first_good,
         "description": "the lowest visibility level of the good band (high altitude allowed)"},
        {"name": "INSPECT", "value": d["inspect"], "description": "metres of pipeline to inspect"},
        {"name": "DEADLINE", "value": d["deadline"], "description": "steps within which the inspection should be done"},
    ] + [{"name": name.upper(), "value": code, "description": f"mission state {name} (values of s)"}
         for name, code in STATES.items()] + [{"name": n, "value": v, "description": t} for n, v, t in ALT_CODES]
    features = [
        {"name": "poor_visib", "description": "the visibility is in the poor band (only low altitude allowed)",
         "expr": poor},
        {"name": "average_visib", "description": "the visibility is in the average band (low or medium allowed)",
         "expr": average},
        {"name": "good_visib", "description": "the visibility is in the good band (every altitude allowed)",
         "expr": good},
        {"name": "starting", "description": "the mission or a search is starting (the first altitude is chosen)",
         "expr": "s = START_TASK | s = START_SEARCH"},
        {"name": "searching", "description": "searching for the pipeline at some altitude",
         "expr": "s = SEARCH_LOW | s = SEARCH_MED | s = SEARCH_HIGH"},
        {"name": "recovering", "description": "recovering from a thruster failure while searching",
         "expr": "s = RECOVER_LOW | s = RECOVER_MED | s = RECOVER_HIGH"},
        {"name": "keeps_altitude", "description": "true if this action leaves the active altitude unchanged (no "
                                                  "switching cost)",
         "per_action": {"low": "alt = LOW",
                        "med": f"(({poor}) & alt = LOW) | (!({poor}) & alt = MED)",
                        "high": f"(({poor}) & alt = LOW) | (({average}) & alt = MED) | (({good}) & alt = HIGH)"}},
        {"name": "allowed_now", "description": "true if the visibility allows this action's altitude (otherwise the "
                                               "highest allowed altitude is used)",
         "per_action": {"low": "true", "med": f"!({poor})", "high": good}},
    ]
    return {"rule_constants": constants, "rule_features": features}
