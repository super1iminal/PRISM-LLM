"""Asher's UUV plan, step 3: the pick with repair re-run on a stored item 8 run (src/equal_spend.py --as-run, no LLM
calls). The NoRepair list must be item 8's (run 2 has no `any` rules), and the new run must read as a finished run:
every round kept, a results parquet with item 8's rows and the LLM task log."""
import json
import shutil

import pandas as pd
import pytest

from equal_spend import main as equal_spend
from settings import PROJECT_ROOT

RUN = PROJECT_ROOT / "out" / "results" / "prepick" / "port_uuv" / "seed_none_2"
needs = pytest.mark.skipif(not shutil.which("prism") or not RUN.exists(), reason="PRISM or item 8's run missing")


@needs
def test_the_repick_reproduces_item_8s_norepair_list_and_reads_as_a_run(tmp_path):
    out = tmp_path / "seed_none_2"
    equal_spend([str(RUN), "--budget", "1000", "--out", str(out), "--as-run"])
    stored = json.loads((RUN / "determinized.json").read_text())["0"]["rules"]
    assert json.loads((out / "norepair.json").read_text())["0"]["determinized"] == stored
    record = json.loads((out / "outputs" / "sample_000.json").read_text())
    original = json.loads((RUN / "outputs" / "sample_000.json").read_text())
    assert record["iterations"] == original["iterations"] and record["equal_spend"]["rounds"] == 7
    assert "pick" in record and record["chains"] == original["chains"]
    new, old = pd.read_parquet(out / "SYMBOLIC_results.parquet"), pd.read_parquet(RUN / "SYMBOLIC_results.parquet")
    assert new.shape == old.shape and (new.llm_calls == old.llm_calls).all()
    assert (out / "llm_tasks.jsonl").read_bytes() == (RUN / "llm_tasks.jsonl").read_bytes()
