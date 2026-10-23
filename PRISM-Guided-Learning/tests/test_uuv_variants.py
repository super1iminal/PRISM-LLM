"""The members_first variants of the UUV plan's conditions (configs/run/conditions/*_mf.yaml) differ from their base
condition in `planner.repair_score` only, so the two variants' rows compare."""
from dataclasses import asdict, replace

import pytest

from config import load_config


@pytest.mark.parametrize("base", ["port_uuv_sets", "port_uuv_sets_words", "port_uuv_resample"])
def test_the_mf_condition_changes_only_the_repair_score(base):
    plain, mf = load_config(base), load_config(base + "_mf")
    assert plain.planner.repair_score == "capped_sum" and mf.planner.repair_score == "members_first"
    same = asdict(replace(mf, planner=replace(mf.planner, repair_score="capped_sum")))
    assert same.pop("condition") == base + "_mf"         # the name the run records
    assert same == {k: v for k, v in asdict(plain).items() if k != "condition"}
