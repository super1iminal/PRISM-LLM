"""The fast checkers (Domain.fast_checker: gridworld's numpy checker, the explicit model's for Pac-Man and MER) give
PRISM's exact worst-case values for rule lists with `any` rules left open and resolved by an order."""
import shutil

import pytest

from config import load_config
from core.domain import Instance, load_domain
from core.rules import fixed_order
from core.verifier import PolicyVerifier

pytestmark = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")

GRID_RULES = [("closer & !wall & !obstacle & !early & !risky & !patrol", "any"),
              ("!wall & !obstacle & !early & !risky", "any"), ("!wall & !early", "any"), ("true", "up")]


def exact(domain, instance, cfg, rules):
    verifier = PolicyVerifier(domain, instance, cfg)
    spec = verifier.spec
    v, _ = verifier.verify_exact(spec.policy(rules, spec.vocabulary(cfg.rules)))
    return v.worst


@pytest.mark.parametrize("dataset, i", [("families_v0.csv", 0), ("families_v0.csv", 64), ("grid_20_large.csv", 3)])
def test_grid_checker_is_prisms_worst_case(dataset, i):
    cfg = load_config("haiku_fam_order", ["rules.any_order=none"])
    domain = load_domain("gridworld", cfg.domain.visible_extra)
    inst = domain.load_instances(dataset)[i]
    checker = domain.fast_checker(inst, cfg, domain.spec(inst).vocabulary(cfg.rules))
    for rules in (GRID_RULES, fixed_order(GRID_RULES, ["down", "right", "up", "left"])):
        theirs, ours = exact(domain, inst, cfg, rules), checker.worst(rules)
        assert ours == pytest.approx(theirs, abs=1e-6), (rules, ours, theirs)


def test_explicit_checker_is_prisms_worst_case():
    cfg = load_config("haiku_mi_pacman_long", ["rules.any_order=none"])
    pacman = load_domain("pacman")
    inst = Instance(id="8", data={"name": "steps_8", "max_steps": 8, "crash_threshold": 0.56})
    rules = [("!wall & !nearer_g0 & !nearer_g1", "any"), ("!wall", "left"), ("true", "up")]
    checker = pacman.fast_checker(inst, cfg, pacman.spec(inst).vocabulary(cfg.rules))
    assert checker.worst(rules) == pytest.approx(exact(pacman, inst, cfg, rules), abs=1e-6)

    cfg = load_config("haiku_mi_mer_n", ["rules.any_order=none"])
    mer = load_domain("mer")
    inst = mer.load_instances("mer_n_transfer.csv")[0]
    rules = [("enabled & s1 = READY & driveUser = USER2", "u1_request_comm"), ("enabled", "any"), ("true", "arb_reclaim")]
    checker = mer.fast_checker(inst, cfg, mer.spec(inst).vocabulary(cfg.rules))
    for r in (rules, fixed_order(rules, list(mer.spec(inst).actions))):
        assert checker.worst(r) == pytest.approx(exact(mer, inst, cfg, r), abs=1e-6)
