"""src/sequential.py: seed libraries from one or several runs, and the certification vocabulary of narrowed lists."""
from dataclasses import replace

from config import load_config
from core.domain import load_domain
from sequential import Sequence, Study, load_study, seed_library
from settings import PROJECT_ROOT
from transfer import final_rules

PAC = "out/results/core/port_pacman_narrow_core/seed_none_{}"


def test_one_seed_run_keeps_its_entry_names():
    lib = seed_library(load_study(PROJECT_ROOT / "configs" / "sequential" / "shift_v0.yaml"))
    assert [e.name for e in lib] == ["seed0", "seed1", "seed2"]
    assert [e.training for e in lib] == [[0, 4, 8], [60, 64, 68], [120, 124, 128]]


def test_several_seed_runs_seed_every_list_of_each():
    study = Study(name="t", condition="port_pacman_narrow_core", dataset="pacman_transfer.csv",
                  seed_run=[PAC.format(k) for k in (1, 2, 3)], sequences=[Sequence("s", [0])])
    lib = seed_library(study)
    assert [e.name for e in lib] == ["seed1.0", "seed2.0", "seed3.0"]
    assert all(e.training == [3, 4, 5] for e in lib)
    for k, e in zip((1, 2, 3), lib):                     # the narrowed lists as kept (determinized = final here)
        assert e.rules == final_rules(PROJECT_ROOT / PAC.format(k))[0]


def test_narrowed_lists_parse_only_with_the_certification_vocabulary():
    """sequential.main turns on rules.shared_literals for conditions with rules.early_narrow, as transfer.py and
    recertify.py do: the core Pac-Man lists name places with numbers (e.g. `!(xP = 3 & yP = 4)`)."""
    cfg = load_config("port_pacman_narrow_core", ["domain.dataset=pacman_transfer.csv"])
    assert cfg.rules.early_narrow and not cfg.rules.shared_literals
    domain = load_domain(cfg.domain.name, cfg.domain.visible_extra)
    spec = domain.spec(domain.load_instances("pacman_transfer.csv")[0])
    rules = final_rules(PROJECT_ROOT / PAC.format(1))[0]
    certify = replace(cfg, rules=replace(cfg.rules, shared_literals=True))
    spec.policy(rules, spec.vocabulary(certify.rules))
    try:
        spec.policy(rules, spec.vocabulary(cfg.rules))
    except Exception as e:
        assert "not allowed" in str(e)
    else:
        raise AssertionError("the narrowed list parsed without shared literals")
