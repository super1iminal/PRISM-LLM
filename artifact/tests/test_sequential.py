"""The adaptation manager of the sequential study (src/sequential.py): its synthesis fallback, which serves an episode no
rule list serves by a certified strategy synthesized for the instance alone (one requirement; with several, PRISM cannot
export one, so the episode is deferred), and the decisions it leads to."""
import shutil
from types import SimpleNamespace

import pandas as pd
import pytest

import sequential
from config import load_config
from core.domain import Instance, load_domain
from sequential import Entry, Sequence, Study, run_sequence, synthesize
from sequential_report import summary

NO_CFG = SimpleNamespace(prism=None)    # the manager tests replace everything that reads it
needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")


def pacman(threshold: float) -> Instance:
    return Instance(id="0", data={"name": "steps_8", "max_steps": 8, "crash_threshold": threshold})


@needs_prism
def test_synthesis_with_one_requirement_deploys_a_certified_optimal_strategy():
    """One requirement: an optimal action in every decision state, certified on the instance like any controller. At
    a threshold the optimum meets it is deployed; below the optimum no strategy is certified."""
    cfg = load_config("port_pacman_narrow_core")
    domain = load_domain("pacman")
    runner = sequential.PrismRunner(cfg.prism)
    row = synthesize(domain, cfg, runner, pacman(0.56))
    assert row["synthesis"] == "synthesized" and row["synthesis_method"] == "interval iteration"
    assert row["synthesis_value"] <= 0.56 and row["synthesis_states"] > 1

    row = synthesize(domain, cfg, runner, pacman(row["synthesis_value"] - 0.01))
    assert row["synthesis"] == "failed"


@needs_prism
def test_synthesis_with_several_requirements_only_decides_existence():
    """A grid world's requirements at once: the multi-objective query says a strategy exists, but PRISM cannot export
    it, so there is nothing to deploy."""
    cfg = load_config("port_fam_history")
    domain = load_domain("gridworld", cfg.domain.visible_extra)
    instance = domain.load_instances("families_v0.csv")[0]
    row = synthesize(domain, cfg, sequential.PrismRunner(cfg.prism), instance)
    assert row["synthesis"] == "exists" and row["synthesis_method"] == "joint query"


@pytest.fixture
def manager(monkeypatch):
    """The manager with every rule list failing certification and generation failing; synthesis succeeds on
    instance 0 and fails on instance 1. Returns the instances and the calls made."""
    calls = {"certify": [], "synthesize": []}
    monkeypatch.setattr(sequential, "PrismRunner", lambda *a, **k: None)
    monkeypatch.setattr(sequential, "seed_library", lambda study: [Entry("seed0", [("true", "up")], [5])])

    def certify(domain, cfg, runner, instance, entry):
        calls["certify"].append((int(instance.id), entry.name))
        return {"instance": int(instance.id), "rule_list": entry.name, "outcome": "failed", "margin": -0.1,
                "seconds": 1.0, "model_seconds": 0.5}

    def synthesize(domain, cfg, runner, instance):
        calls["synthesize"].append(int(instance.id))
        ok = instance.id == "0"
        return {"synthesis": "synthesized" if ok else "failed", "synthesis_method": "joint query",
                "synthesis_seconds": 2.0}

    monkeypatch.setattr(sequential, "certify", certify)
    monkeypatch.setattr(sequential, "generate", lambda *a: (None, {"generation_seconds": 3.0}))
    monkeypatch.setattr(sequential, "synthesize", synthesize)
    instances = {i: Instance(id=str(i), data={"family": "rooms", "n": 10}) for i in (0, 1, 5)}
    return instances, calls


def study(**kw) -> Study:
    return Study(name="t", condition="c", dataset="d.csv", seed_run="r", sequences=[], **kw)


@pytest.mark.parametrize("mode", ["growing", "frozen"])
def test_manager_synthesizes_instead_of_deferring(manager, tmp_path, mode):
    """Where no rule list serves the episode, synthesis does; the library and the current rule list stay as they were,
    and the episode is deferred only where synthesis fails too."""
    instances, calls = manager
    run_sequence(study(), Sequence("s", [0, 1, 0]), mode, NO_CFG, None, instances, tmp_path)
    episodes = pd.read_csv(tmp_path / "episodes.csv")
    assert episodes.decision.tolist() == ["synthesize", "defer", "synthesize"]
    assert episodes.deployed.tolist()[0] == "synthesis" and pd.isna(episodes.deployed[1])
    assert episodes.library_size.tolist() == [1, 1, 1]
    assert calls["synthesize"] == [0, 1, 0]
    assert all(rule_list == "seed0" for _, rule_list in calls["certify"])     # no rule list became current
    assert ("generation_seconds" in episodes) == (mode == "growing")


def test_existence_alone_does_not_serve_an_episode(manager, monkeypatch, tmp_path):
    """Where synthesis only proves that a strategy exists (several requirements), the episode is still deferred."""
    instances, calls = manager
    monkeypatch.setattr(sequential, "synthesize", lambda *a: {"synthesis": "exists", "synthesis_method": "joint query",
                                                              "synthesis_seconds": 2.0})
    run_sequence(study(), Sequence("s", [0, 1]), "frozen", NO_CFG, None, instances, tmp_path)
    episodes = pd.read_csv(tmp_path / "episodes.csv")
    assert episodes.decision.tolist() == ["defer", "defer"] and episodes.deployed.isna().all()
    assert episodes.synthesis.tolist() == ["exists", "exists"]


def test_manager_defers_without_the_fallback(manager, tmp_path):
    instances, calls = manager
    run_sequence(study(fallback="defer"), Sequence("s", [0, 1]), "growing", NO_CFG, None, instances, tmp_path)
    assert pd.read_csv(tmp_path / "episodes.csv").decision.tolist() == ["defer", "defer"]
    assert calls["synthesize"] == []


def test_library_seeds_from_several_runs():
    """pacman_v0 seeds from three runs: entries named seed<k>.<sample id>, as in its stored library.json files."""
    from settings import PROJECT_ROOT
    study = sequential.load_study(PROJECT_ROOT / "configs" / "sequential" / "pacman_v0.yaml")
    library = sequential.seed_library(study)
    assert [e.name for e in library] == ["seed1.0", "seed2.0", "seed3.0"]
    assert all(e.rules and e.training == [3, 4, 5] for e in library)
    single = sequential.load_study(PROJECT_ROOT / "configs" / "sequential" / "shift_v0.yaml")
    assert [e.name for e in sequential.seed_library(single)] == ["seed0", "seed1", "seed2"]


def test_unknown_fallback_is_refused():
    with pytest.raises(ValueError, match="fallback"):
        study(fallback="retry")


def test_report_counts_synthesized_episodes_as_served():
    df = pd.DataFrame({"mode": "growing", "sequence": "s", "episode": range(4), "library_size": 3,
                       "decision": ["retain", "synthesize", "defer", "synthesize"]})
    row = summary(df).iloc[0]
    assert (row.served, row.synthesize, row.defer, row.retain) == (3, 2, 1, 1)
