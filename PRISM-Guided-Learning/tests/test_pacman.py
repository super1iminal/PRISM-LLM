"""The Pac-Man domain reproduces the QVBS benchmark and its thresholds separate the reference policies."""
import json
import logging
import re
import shutil
from pathlib import Path

import pytest

from config import load_config
from core.domain import Instance, load_domain
from core.planner import SymbolicPlanner
from core.prism import PrismRunner
from core.rules import SymbolicPolicy
from core.verifier import PolicyVerifier
from fakes import ScriptedBackend, answer

ROOT = Path(__file__).resolve().parent.parent
CFG = load_config()
REFERENCE = json.loads((ROOT / "domains/pacman/data/reference_policies.json").read_text(encoding="utf-8"))

pytestmark = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")


@pytest.fixture(scope="module")
def domain():
    return load_domain("pacman")


def instance(max_steps, threshold=0.56):
    return Instance(id="0", data={"name": f"steps_{max_steps}", "max_steps": max_steps, "crash_threshold": threshold})


def test_bare_mdp_matches_qvbs(domain):
    """QVBS reports 498 states (Storm) and Pmin(F Crash) = 5511/10000 for MAXSTEPS = 5."""
    result = PrismRunner(CFG.prism).run(domain.model(instance(5)), ['Pmin=? [ F "Crash" ]'], export_transitions=True)
    assert len(result.states) == 498
    assert result.initial_values[0] == pytest.approx(0.5511, abs=1e-9)


def test_crash_range_over_controllers(domain):
    """Computed from the unchanged QVBS file with PRISM 4.10.1: the minimum stays 0.5511, the maximum grows."""
    verifier = PolicyVerifier(domain, instance(10), CFG)
    best, worst = verifier.runner.run(verifier.model, ['Pmin=? [ F "Crash" ]', 'Pmax=? [ F "Crash" ]']).initial_values
    assert (best, worst) == (pytest.approx(0.5511, abs=1e-9), pytest.approx(0.8844, abs=1e-9))


def test_decisions_only_at_crossings(domain):
    verifier = PolicyVerifier(domain, instance(10), CFG)
    v = verifier.verify(verifier.empty_policy())
    pos = v.result.positions
    cells = {(s[pos["xP"]], s[pos["yP"]]) for s in v.result.states}
    deciding = {(s[pos["xP"]], s[pos["yP"]]) for s, d in zip(v.result.states, v.decisions) if d}
    assert cells <= set(domain.walkable())
    assert deciding <= set(domain.crossings()) and len(deciding) > 1


def test_thresholds_separate_reference_policies(domain):
    def crash(max_steps, rules):
        verifier = PolicyVerifier(domain, instance(max_steps), CFG)
        spec = verifier.spec
        return verifier.verify(SymbolicPolicy.from_raw(spec.variables, list(spec.actions), rules),
                               analysis=False).worst["crash"]

    assert crash(8, REFERENCE["keep_heading"]) <= 0.56                        # short games are easy
    assert all(crash(12, rules) > 0.56 for rules in REFERENCE.values())       # no simple policy survives 12 moves


def test_horizon(domain):
    assert domain.horizon(instance(10)) == 33


def test_feature_space_feedback_names_no_position(domain):
    """Feature-space feedback on Pac-Man (horizon 8) names states by `moves_left` and the action features, never by
    a position, a heading or the step count; the integer feature `moves_left` reads as `name=value`."""
    cfg = load_config("haiku_mi_pacman_long", ["planner.max_rounds=2", "llm.seed=1", "prism.exact_check=false",
                                               "run.train_sets=[]", "feedback.space=features",
                                               "feedback.trajectory=true"])
    llm = ScriptedBackend([answer(("nearer_g0", "right"))])
    result = SymbolicPlanner(domain, llm, cfg).solve(instance(8), logging.getLogger("test_pacman"))
    assert [it["mode"] for it in result["iterations"]] == ["initial", "extend"]
    prompt = result["iterations"][1]["prompt"]
    feedback = prompt[prompt.index("## Uncovered states where probability is lost"):prompt.index("## Task")]
    assert "## A likely way to fail `crash`" in feedback and "moves_left=8 | right: " in feedback
    assert not re.search(r"\b(xP|yP|dP|xG0|yG0|dG0|xG1|yG1|dG1|steps)=", feedback)


def test_early_check_flags_the_dead_end(domain):
    """`feedback.early` on horizon 14: going up at (3,4), into the dead end, loses in the first half of the game; a
    list that goes down there instead loses less."""
    from core import early
    cfg = load_config("haiku_pac_early_refine")
    verifier = PolicyVerifier(domain, instance(14), cfg)
    check = early.EarlyCheck(verifier, "steps", "instance 1")
    vocab = verifier.spec.vocabulary(cfg.rules, ["xP", "yP"])
    up = early.check([(check, verifier.spec.policy([("!wall", "up"), ("true", "right")], vocab))], group=["xP", "yP"])
    pocket = next(r for r in up.rules if r.place == (("xP", 3), ("yP", 4)))
    assert pocket.move == "up" and pocket.rule == 0 and pocket.losing > 0 and "up" not in pocket.best
    down = early.check([(check, verifier.spec.policy([("xP = 3 & yP = 4", "down"), ("!wall", "up"),
                                                      ("true", "right")], vocab))], group=["xP", "yP"])
    assert down.mean < up.mean and not any(r.place == (("xP", 3), ("yP", 4)) and r.move == "up" for r in down.rules)


def test_narrowing_keeps_the_dead_end_move_away(domain):
    """`rules.early_narrow`: a list that passes horizons 14, 16, 18 but goes up into the dead end at (3,4) gets `up`
    kept away from (3,4) by the verifier, still passes, and the run stops there (`early_tolerance: 1`)."""
    cfg = load_config("haiku_pac_narrow", ["rules.any_order=rotations", "planner.max_rounds=2"])
    rules = [("!wall & !nearer_g0 & !nearer_g1", "any"), ("!nearer_g0 & !nearer_g1", "any"), ("!wall & !nearer_g0", "any"),
             ("!wall & !nearer_g1", "any"), ("!nearer_g0", "any"), ("!nearer_g1", "any"), ("true", "right")]
    from core.training import training_sets
    target = training_sets(domain.load_instances("pacman_transfer.csv"), [[3, 4, 5]])[0]
    result = SymbolicPlanner(domain, ScriptedBackend([answer(*rules)]), cfg).solve(target, logging.getLogger("test_pacman"))
    assert result["success"] and len(result["iterations"]) == 1
    assert result["iterations"][0]["narrowed"] == ["up kept away from xP = 3 & yP = 4"]
    up = [r["condition"] for r in result["final_rules"] if r["action"] == "up"]
    assert up and all(c.endswith("& !(xP = 3 & yP = 4)") for c in up)


def test_sealed_horizons_are_refused(domain, tmp_path):
    """Horizons 50 and 100 (pacman_transfer.csv ids 10, 11) are Pac-Man's sealed test set."""
    import yaml
    import run_baseline
    from transfer import guard_sealed
    horizons = [i.data["max_steps"] for i in domain.load_instances("pacman_transfer.csv")]
    assert horizons == [8, 10, 12, 14, 16, 18, 20, 25, 5, 30, 50, 100]
    guard_sealed("pacman_transfer.csv", list(range(10)))
    for ids in ([10], [11], [3, 11]):
        with pytest.raises(SystemExit, match="sealed"):
            guard_sealed("pacman_transfer.csv", ids)
    config = tmp_path / "sealed.yaml"
    config.write_text(yaml.safe_dump({
        "method": "rule_lists", "condition": "haiku_mi_pacman_long", "dataset": "pacman_transfer.csv",
        "target_dataset": None, "train": "", "targets": "9-10", "workers": 1, "exact_near": 0.0,
        "params": {"lists": {"right": [["true", "right"]]}}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="sealed"):
        run_baseline.run(config, out_root=tmp_path)
