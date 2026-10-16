"""Randomised consensus: schedule the processes of a shared-coin consensus protocol so that they decide differently.

The MDP is the QVBS benchmark `consensus` (Aspnes and Herlihy's shared-coin protocol, the PRISM case study of
Kwiatkowska, Norman and Segala), with N processes and the random walk bound K. Each process flips a coin, moves a
shared counter one step down (tails) or up (heads) and checks it: at or below `left` it decides tails, at or above
`right` heads, otherwise it flips again. Every step of process i is the policy action p{i}, so the policy is the
scheduler. The requirement asks for a high probability that the processes finish without agreeing, the benchmark's
property `disagree` (Pmax). Instances vary K at a fixed N: the counter's range and the barriers move with K, the
variables and actions stay the same.

Dataset rows (CSV): name, n, k, optimum (Pmax from PRISM), disagree_threshold (the optimum minus 5% of it).
"""
from typing import Any, Dict, List

from core.domain import Domain, Instance


class Consensus(Domain):

    def load_instances(self, dataset: str) -> List[Instance]:
        df = self.read_dataset(dataset)
        return [Instance(id=str(idx), data={"name": str(row["name"]), "n": int(row["n"]), "k": int(row["k"]),
                                            "optimum": float(row["optimum"]),
                                            "disagree_threshold": float(row["disagree_threshold"])})
                for idx, row in df.iterrows()]

    def context(self, instance: Instance) -> Dict[str, Any]:
        d = dict(instance.data)
        n, k = d["n"], d["k"]
        d.update({"procs": list(range(1, n + 1)), "movers": movers(n), "left": n, "right": 2 * (k + 1) * n - n,
                  "range": 2 * (k + 1) * n, "start": (k + 1) * n, **rule_vocabulary(n, k)})
        return d


def movers(n: int) -> Dict[int, List[tuple]]:
    """Action p{i} -> the processes it may move, each with its guard prefix: process i itself, then each process j
    after i (cyclically) once every process from i up to j has finished."""
    out = {}
    for i in range(1, n + 1):
        order = [(i - 1 + s) % n + 1 for s in range(n)]
        out[i] = [(j, "".join(f"pc{b}=3 & " for b in order[:m])) for m, j in enumerate(order)]
    return out


# ---------------------------------------------------------------- extended rule vocabulary

# The model's codes, the same in every instance, named so that rules in the general language (`rules.general`: no
# number above 1) can use them
CODES = [
    ("FLIP", 0, "a process's pc: about to flip its coin"),
    ("WRITE", 1, "a process's pc: about to write its coin (move the counter)"),
    ("CHECK", 2, "a process's pc: about to check the counter"),
    ("DONE", 3, "a process's pc: finished (it has decided)"),
    ("TAILS", 0, "a coin or decision: tails (moves the counter down)"),
    ("HEADS", 1, "a coin or decision: heads (moves the counter up)"),
]


def rule_vocabulary(n: int, k: int) -> Dict[str, Any]:
    """The model's constants N, K, LEFT, RIGHT and START (counter_init) and the named codes (CODES); the state
    features to_heads, to_tails, heads_decided and tails_decided; the per-action features finished, stage and coin,
    about the process an action names."""
    procs = range(1, n + 1)
    decided = {v: " | ".join(f"(pc{i} = DONE & coin{i} = {v})" for i in procs) for v in ("HEADS", "TAILS")}
    features = [
        {"name": "to_heads", "description": "RIGHT - counter: how many more heads writes than tails writes the "
                                            "counter needs before a check decides heads (0 or less: a check now "
                                            "decides heads)",
         "expr": "RIGHT - counter"},
        {"name": "to_tails", "description": "counter - LEFT: how many more tails writes than heads writes the counter "
                                            "needs before a check decides tails (0 or less: a check now decides "
                                            "tails)",
         "expr": "counter - LEFT"},
        {"name": "heads_decided", "description": "some process has finished with the decision heads",
         "expr": decided["HEADS"]},
        {"name": "tails_decided", "description": "some process has finished with the decision tails",
         "expr": decided["TAILS"]},
        {"name": "finished", "description": "the process this action names has finished (the action then moves the "
                                            "next unfinished process)",
         "per_action": {f"p{i}": f"pc{i} = DONE" for i in procs}},
        {"name": "stage", "description": "the pc of the process this action names: FLIP, WRITE, CHECK, or DONE",
         "per_action": {f"p{i}": f"pc{i}" for i in procs}},
        {"name": "coin", "description": "the coin of the process this action names (TAILS or HEADS; once it has "
                                        "finished, its decision)",
         "per_action": {f"p{i}": f"coin{i}" for i in procs}},
    ]
    return {"rule_constants": [
                {"name": "N", "value": n, "description": "the number of processes"},
                {"name": "K", "value": k, "description": "the random walk bound (RIGHT and START grow with it)"},
                {"name": "LEFT", "value": n, "description": "a check at counter <= LEFT decides tails"},
                {"name": "RIGHT", "value": 2 * (k + 1) * n - n,
                 "description": "a check at counter >= RIGHT decides heads"},
                {"name": "START", "value": (k + 1) * n,
                 "description": "the counter's initial value, halfway between LEFT and RIGHT"}]
            + [{"name": name, "value": value, "description": text} for name, value, text in CODES],
            "rule_features": features}
