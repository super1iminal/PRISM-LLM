"""src/budget_transfer.py: the list keep-best holds after each round, its cost, and its certification on targets."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

import budget_transfer as bt
from config import load_config
from core.domain import Requirement

needs_prism = pytest.mark.skipif(not shutil.which("prism"), reason="PRISM not on PATH")

RIGHT = [{"condition": "true", "action": "right"}]
LEFT = [{"condition": "true", "action": "left"}]


def iteration(k, rules, improved, prompt=100, output=10, solved=False, worst=0.5):
    return {"iteration": k, "rules": rules, "improved": improved, "llm_prompt_tokens": prompt,
            "llm_output_tokens": output, "worst_case_success": solved, "worst": {"goal": worst}}


def fake_run(tmp_path, tiny_grid, records):
    """A finished run on the tiny grid with the given per-sample records."""
    run = tmp_path / "run"
    (run / "outputs").mkdir(parents=True)
    cfg = load_config(overrides=[f"domain.dataset={tiny_grid.as_posix()}"])
    (run / "config.json").write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    for sid, record in enumerate(records):
        (run / "outputs" / f"sample_{sid:03d}.json").write_text(json.dumps(record), encoding="utf-8")
    return run


def test_kept_lists_hold_the_last_improved_round_and_add_up_the_cost():
    record = {"instance": "0", "iterations": [iteration(1, LEFT, True), iteration(2, RIGHT, False, output=30),
                                              iteration(3, RIGHT, True, prompt=50, solved=True, worst=0.9)]}
    kept = bt.kept_lists(record, lambda iid: [Requirement("goal", "F goal", 0.8)])
    assert [k["kept_round"] for k in kept] == [1, 1, 3]
    assert [k["rules"] for k in kept] == [[("true", "left")], [("true", "left")], [("true", "right")]]
    assert [(k["prompt_tokens"], k["output_tokens"]) for k in kept] == [(100, 10), (200, 40), (250, 50)]
    assert [k["solved"] for k in kept] == [False, False, True]
    assert [(k["train_met"], k["train_requirements"]) for k in kept] == [(0, 1), (0, 1), (1, 1)]


def test_the_training_score_sums_the_shown_members():
    it = {"worst": {"goal": 0.1}, "members": [{"instance": "4", "worst": {"goal": 0.9}},
                                              {"instance": "5", "worst": {"goal": 0.85}},
                                              {"instance": "6", "worst": {"goal": 0.95}, "hidden": True}]}
    thresholds = {4: 0.8, 5: 0.9, 6: 0.0}
    met = bt.training_met(it, "set0", lambda iid: [Requirement("goal", "F goal", thresholds[iid])])
    assert met == (1, 2)


def _table():
    rows = []
    for source, rounds in {0: [(1, 0.1, 1, False), (2, 0.2, 3, False), (3, 0.3, 3, True)],
                           1: [(1, 0.25, 2, False)]}.items():
        for k, dollars, certified, solved in rounds:
            for t in range(4):     # target 3 is a training instance: left out
                rows.append({"source": source, "round": k, "kept_round": k, "first_answer": k == 1,
                             "prompt_tokens": 0, "output_tokens": 0, "dollars": dollars, "solved": solved,
                             "train_met": k, "train_requirements": 9, "target": t, "trained": t == 3,
                             "certified": t < certified})
    return bt.rounds_table(pd.DataFrame(rows))


def test_rounds_table_counts_held_out_targets_only():
    table = _table()
    assert table[table.source == 0].certified.tolist() == [1, 3, 3]
    assert table.held_out.unique().tolist() == [3]


def test_at_budget_reads_each_set_at_its_last_affordable_round():
    table = _table()
    early = bt.at_budget(table, 0.15).set_index("source")
    assert early.loc[0, "certified"] == 1 and early.loc[1, "certified"] == 0 and np.isnan(early.loc[1, "round"])
    late = bt.at_budget(table, 1.0).set_index("source")
    assert late.loc[0, "round"] == 3 and late.loc[0, "solved"] and late.loc[1, "certified"] == 2
    assert late.loc[0, "train_met"] == 3 and early.loc[1, "train_met"] == 0
    assert bt.run_value(table, 0.2, "certified") == pytest.approx((3 + 0) / 2)
    assert bt.run_value(table, 1.0, "solved") == 0.5
    assert bt.at_round(table, 2).set_index("source")["round"].to_dict() == {0: 2, 1: 1}


def test_bootstrap_test_and_holm():
    diff, lo, hi, p = bt.bootstrap_test([5, 6, 7], [1, 2, 3])
    assert diff == 4 and lo > 0 and p < 0.01
    _, lo, hi, p = bt.bootstrap_test([1, 2, 3], [1, 2, 3])
    assert lo < 0 < hi and p > 0.5
    assert bt.holm([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])


@needs_prism
def test_budget_table_certifies_each_kept_list_once_and_reuses_the_transfer_csv(tmp_path, tiny_grid, monkeypatch):
    from core.domain import load_domain
    domain = load_domain("gridworld")
    reqs = domain.spec(domain.load_instances(str(tiny_grid))[0]).requirements
    fail = {r.name: (0.0 if r.maximize else 1.0) for r in reqs}
    passing = {r.name: r.threshold for r in reqs}
    rounds = [iteration(1, LEFT, True), iteration(2, LEFT, False), iteration(3, RIGHT, True, solved=True)]
    for it, worst in zip(rounds, [fail, fail, passing]):
        it["worst"] = worst
    run = fake_run(tmp_path, tiny_grid, [{"instance": "0", "final_rules": RIGHT, "iterations": rounds}])
    calls = []
    real = bt.certify
    monkeypatch.setattr(bt, "certify", lambda *a, **k: calls.append(sorted(a[4])) or real(*a, **k))
    # the final list's row comes from the transfer CSV: only the round-1 list needs PRISM
    pd.DataFrame([{"source": 0, "target": 0, "parsed": True, "certified": True, "met": 3, "requirements": 3,
                   "trained": True}]).to_csv(run / f"transfer_{tiny_grid.stem}.csv", index=False)
    df = bt.budget_table(run, price=[1e6, 0])
    assert calls == [[bt.rules_hash([("true", "left")])]]
    assert df["round"].tolist() == [1, 2, 3] and df.first_answer.tolist() == [True, False, False]
    assert df.certified.tolist() == [False, False, True] and df.trained.all()
    assert df.dollars.tolist() == [100, 200, 300]
    assert df.train_met.tolist() == [0, 0, len(reqs)] and (df.train_requirements == len(reqs)).all()
    calls.clear()
    assert bt.budget_table(run).certified.tolist() == [False, False, True] and calls == []   # cached in the CSV
    assert (run / bt.BUDGET_CSV).exists()


def test_sealed_targets_are_refused(tmp_path):
    run = tmp_path / "run"
    (run / "outputs").mkdir(parents=True)
    cfg = load_config("haiku_fam_order")
    (run / "config.json").write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    with pytest.raises(SystemExit, match="sealed"):
        bt.budget_table(run, targets="30-40")
