"""The fast checker (Domain.fast_checker, core/checker.py: the same for every domain) gives PRISM's exact worst-case
values for rule lists with `any` rules left open and resolved by an order, and its formula automata."""
import shutil

import numpy as np
import pytest

from config import load_config
from core.checker import Objective, parse, reach, supported
from core.domain import Instance, load_domain
from core.explicit import _build
from core.rules import fixed_order
from core.verifier import PolicyVerifier

needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")

GRID_RULES = [("closer & !wall & !obstacle & !early & !risky & !patrol", "any"),
              ("!wall & !obstacle & !early & !risky", "any"), ("!wall & !early", "any"), ("true", "up")]


def exact(domain, instance, cfg, rules):
    verifier = PolicyVerifier(domain, instance, cfg)
    spec = verifier.spec
    v, _ = verifier.verify_exact(spec.policy(rules, spec.vocabulary(cfg.rules)))
    return v.worst


@needs_prism
@pytest.mark.parametrize("dataset, i", [("families_v0.csv", 0), ("families_v0.csv", 64), ("grid_20_large.csv", 3)])
def test_gridworld_is_prisms_worst_case(dataset, i):
    """Every gridworld requirement: `F`, nested `U` and their conjunction (co-safe), `G` (safe)."""
    cfg = load_config("haiku_fam_order", ["rules.any_order=none"])
    domain = load_domain("gridworld", cfg.domain.visible_extra)
    inst = domain.load_instances(dataset)[i]
    checker = domain.fast_checker(inst, cfg, domain.spec(inst).vocabulary(cfg.rules))
    for rules in (GRID_RULES, fixed_order(GRID_RULES, ["down", "right", "up", "left"])):
        theirs, ours = exact(domain, inst, cfg, rules), checker.worst(rules)
        assert ours == pytest.approx(theirs, abs=1e-6), (rules, ours, theirs)


@needs_prism
def test_pacman_and_mer_are_prisms_worst_case():
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


@needs_prism
def test_consensus_is_prisms_worst_case():
    """A slowly mixing random walk, where value iteration stops early: policy iteration solves it exactly."""
    cfg = load_config()
    domain = load_domain("consensus")
    inst = Instance("0", {"name": "n2_k2", "n": 2, "k": 2, "optimum": 0.0, "disagree_threshold": 0.0})
    checker = domain.fast_checker(inst, cfg, domain.spec(inst).vocabulary(cfg.rules))
    for rules in ([("true", "p1")], [("counter > 6", "p2"), ("true", "p1")]):
        assert checker.worst(rules) == pytest.approx(exact(domain, inst, cfg, rules), abs=1e-6)


def test_uuv_has_no_fast_checker():
    """Step bounds and rewards are outside the checker: the loop uses PRISM there."""
    domain = load_domain("uuv")
    inst = domain.load_instances("uuv_paper.csv")[0]
    assert not supported(domain.spec(inst).requirements)


def test_parse_follows_prisms_precedence():
    a, b, c = ("label", "a"), ("label", "b"), ("label", "c")
    assert parse('F "a" & !"b"') == ("F", ("&", a, ("!", b)))
    assert parse('!"a" U "b" & "c"') == ("U", ("!", a), ("&", b, c))
    assert parse('G ("a" => !"b")') == ("G", ("=>", a, ("!", b)))
    assert parse('(F "a") & (G "b")') == ("&", ("F", a), ("G", b))
    for bad in ('F<=5 "a"', 'x = 3', 'F "a" U'):
        with pytest.raises(NotImplementedError):
            parse(bad)


def test_automata():
    """Letters are bit sets over the atoms (maximal state subformulas) in the order they first appear."""
    seq = Objective('!"c" U ("a" & (!"c" U "b"))')       # atoms: !c, a, b
    assert seq.cosafe and len(seq.atoms) == 3
    live, delta, first = seq.automaton(np.array([0b001, 0b011, 0b111, 0b000]))
    assert len(live) == 2
    assert first.tolist() == [0, 1, -1, -2]               # wait; a reached; a and b at once; c entered first
    assert delta[1].tolist() == [1, 1, -1, -2]            # after a: wait for b while avoiding c
    safe = Objective('G ("s" => !"o")')
    assert not safe.cosafe
    _, delta, first = safe.automaton(np.array([0, 1]))    # one atom: s => !o
    assert first.tolist() == [-2, 0] and delta.tolist() == [[-2, 0]]
    with pytest.raises(NotImplementedError):
        Objective('G (F "a")')
    with pytest.raises(NotImplementedError):
        Objective('!(F "a")')


def test_reach_is_exact_on_end_components():
    """State 0 chooses between a loop through state 1 (an end component, value 0 when minimizing) and a coin flip
    between the goal (2) and a trap (3); value iteration would only approach these values."""
    #            state 0: loop to 1 | flip;  state 1: back to 0 | 0.9 stay + 0.1 goal; 2, 3 absorbing
    m = _build(4, np.array([0, 0, 1, 1, 2, 3]), np.full(6, -1), np.array([0, 1, 1, 2, 3, 3, 4, 5]),
               np.array([1, 2, 3, 0, 1, 2, 2, 3]),
               np.array([1.0, 0.5, 0.5, 1.0, 0.9, 0.1, 1.0, 1.0]), {}, 0)
    target = np.array([False, False, True, False])
    assert reach(m, target, maximize=True).tolist() == pytest.approx([1.0, 1.0, 1.0, 0.0], abs=1e-12)
    assert reach(m, target, maximize=False).tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0], abs=1e-12)
