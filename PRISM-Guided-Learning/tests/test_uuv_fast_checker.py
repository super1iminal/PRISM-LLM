"""The UUV fast checker (UUV.fast_checker, baselines/explicit_mdp.RuleChecker; Asher's UUV plan, step 2): on every
member of the family's training and validation rows (0-29), its worst-case values must equal PRISM's interval
iteration within 1e-7 for the three stored item 8 lists, the hand-written list and the reference controllers. The
tightest certified margins are about 1e-5, so a looser match could reorder the pick."""
import json
from pathlib import Path

import pytest

from config import load_config
from core.domain import load_domain
from core.verifier import PolicyVerifier
from settings import PROJECT_ROOT

TOLERANCE = 1e-7
STORED = [PROJECT_ROOT / "out" / "results" / root / "port_uuv" for root in ("prepick", "ablations")]
HAND = [("alt = LOW & !starting & water_visib < FIRST_AVG + 1", "low"), ("alt = HIGH", "high"), ("true", "med")]


@pytest.fixture(scope="module")
def setting():
    domain = load_domain("uuv")
    general, numbers = load_config("port_uuv"), load_config("uuv_baseline")
    root = next(p for p in STORED if p.exists())
    lists = {f"item8_run{k}": ([tuple(r) for r in json.loads((root / f"seed_none_{k}" / "determinized.json")
                                                             .read_text())["0"]["rules"]], general) for k in (1, 2, 3)}
    lists["hand"] = (HAND, general)
    lists.update({name: ([tuple(r) for r in rules], numbers) for name, rules in domain.reference_policies().items()})
    return domain, {int(i.id): i for i in domain.load_instances("uuv_family_v0.csv")}, lists


@pytest.mark.parametrize("member", range(30))
def test_the_fast_checker_equals_interval_iteration(setting, member):
    domain, instances, lists = setting
    inst = instances[member]
    checkers = {}
    for name, (rules, cfg) in lists.items():
        if id(cfg) not in checkers:
            checkers[id(cfg)] = domain.fast_checker(inst, cfg, domain.spec(inst).vocabulary(cfg.rules))
        fast = checkers[id(cfg)].worst(rules)
        v = PolicyVerifier(domain, inst, cfg)
        exact, solver = v.verify_exact(v.spec.policy(rules, v.spec.vocabulary(cfg.rules)))
        assert solver == "interval iteration", (name, solver)
        for req, value in exact.worst.items():
            assert abs(fast[req] - value) <= TOLERANCE, (inst.data["name"], name, req, fast[req], value)


def test_almost_sure_reachability_is_exact(setting):
    """`done` is reached with probability 1 under every scheduler (the bare model), which value iteration only
    approaches: the graph algorithm must say so exactly, or expected energy would read as infinite."""
    domain, instances, _ = setting
    cfg = load_config("uuv_baseline")
    mdp = domain.fast_checker(instances[0], cfg, domain.spec(instances[0]).vocabulary(cfg.rules)).mdp
    done = mdp.labels["done"]
    assert mdp.almost_surely(done, every=True).all() and mdp.almost_surely(done, every=False).all()
    assert not mdp.almost_surely(mdp.labels["thruster_failure"], every=True)[mdp.initial]
