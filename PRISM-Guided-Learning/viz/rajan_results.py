"""Rajan's SEAMS checklist tables, from the runs and checks the items produced (no LLM or PRISM calls).

Coverage, per-list statistics, training pass rates and LLM calls reuse viz/paper_results.py's own functions, so the
numbers are computed exactly as Asher's tables compute them. Writes into out/paper/tables/ under the names the checklist
gives; what each item ran is in out/paper/RESULTS-NOTES-rajan.md (written by hand).

Usage (from PRISM-Guided-Learning/): python viz/rajan_results.py [--items 1 2 3 4 5 6 7]
Items whose outputs do not exist yet are skipped with a note.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_results as pr  # noqa: E402
from experiments.base import experiment  # noqa: E402
from transfer import csv_path  # noqa: E402

RESULTS, TABLES = pr.RESULTS, pr.OUT / "tables"
SEQ = RESULTS / "sequential"
# Item 3's LLMs: label -> (condition suffix, OpenRouter id, $ per M prompt / output tokens on the pinned endpoint)
LLMS = {"GPT-6 Luna": ("_luna", "openai/gpt-6-luna", (0.10, 0.50)),
        "Gemini 3.8 Flash": ("_gemini", "google/gemini-3.8-flash", (0.75, 3.75))}
HAIKU = (0.10, 0.50)
DECISIONS = ["retain", "switch", "generate", "defer"]


def note(msg: str) -> None:
    print(f"  {msg}")


# ---------------------------------------------------------------- item 1: synthesis time

def item1() -> None:
    parts = sorted(TABLES.glob("tab_rq2-synthesis-time.*.csv"))
    final = TABLES / "tab_rq2-synthesis-time.csv"
    if parts:
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        df.to_csv(final, index=False)
        print(f"wrote {final.relative_to(pr.ROOT)} from {len(parts)} parts")
    if not final.exists():
        return note("item 1: no synthesis-time CSV yet")
    df = pd.read_csv(final)
    df["kind"] = np.where(df.method == "synthesis", "synthesis", "certification")
    summary = df.groupby(["domain", "size", "kind"]).agg(
        checks=("instance", "count"), states_median=("states", "median"),
        construction_s_median=("construction_s", "median"), check_s_median=("check_s", "median"),
        check_s_max=("check_s", "max"), not_done=("outcome", lambda s: int((s != "done").sum()))).reset_index()
    summary.to_csv(TABLES / "tab_rq2-synthesis-time-summary.csv", index=False)
    print("wrote tables/tab_rq2-synthesis-time-summary.csv")
    if "asher_outcome" in df:
        diff = df[df.asher_outcome.notna() & (df.asher_outcome != df.verdict)]
        note(f"item 1: {len(diff)} verdicts differ from Asher's (write them under Blocked)")


# ---------------------------------------------------------------- sequences (items 2, 4, 5, 6)

def episodes_of(study: str, mode: str) -> pd.DataFrame:
    frames = []
    for seq in sorted((SEQ / study / mode).glob("*/episodes.csv")):
        frames.append(pd.read_csv(seq).assign(study=study, mode=mode, sequence=seq.parent.name))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def per_sequence(ep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (study, mode, seq), g in ep.groupby(["study", "mode", "sequence"]):
        counts = {d: int((g.decision == d).sum()) for d in DECISIONS}
        rows.append({"study": study, "mode": mode, "sequence": seq, "episodes": len(g), **counts,
                     "served": counts["retain"] + counts["switch"] + counts["generate"],
                     "generations_attempted": int(g.get("generation_seconds", pd.Series(dtype=float)).notna().sum()),
                     "final_library": int(g.library_size.iloc[-1]),
                     "llm_calls": g.get("llm_calls", pd.Series(dtype=float)).sum(),
                     "prompt_tokens": g.get("prompt_tokens", pd.Series(dtype=float)).sum(),
                     "output_tokens": g.get("output_tokens", pd.Series(dtype=float)).sum()})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["llm_dollars"] = (df.prompt_tokens * HAIKU[0] + df.output_tokens * HAIKU[1]) / 1e6
    return df


def item2() -> None:
    rows = []
    for k, study in enumerate(["random_v0", "random_v0_lib2", "random_v0_lib3"], 1):
        ep = episodes_of(study, "frozen")
        if ep.empty:
            note(f"item 2: no frozen episodes for {study}")
            continue
        for seq, g in ep.groupby("sequence"):
            c = {d: int((g.decision == d).sum()) for d in ("retain", "switch", "defer")}
            rows.append({"sequence": seq, "library": f"seed_none_{k}", **c, "served": c["retain"] + c["switch"],
                         "episodes": len(g), "split": "validation"})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(TABLES / "tab_rq2-random-sequences.csv", index=False)
        print("wrote tables/tab_rq2-random-sequences.csv")
        print(df.groupby("library")[["retain", "switch", "defer", "served"]].agg(["mean", "min", "max"]).to_string())


def served_by_earlier(ep: pd.DataFrame) -> pd.Series:
    """Per episode of a growing run: served by a list generated earlier in the same sequence."""
    gen_at = ep.deployed.astype(str).str.extract(r"^gen_t(\d+)$")[0].astype(float)
    return ep.decision.isin(["retain", "switch"]) & gen_at.notna() & (gen_at < ep.episode)


def item4() -> None:
    frozen, growing = episodes_of("long_v0", "frozen"), episodes_of("long_v0", "growing")
    if growing.empty and frozen.empty:
        return note("item 4: no long_v0 episodes yet")
    out = []
    for mode, ep in (("frozen", frozen), ("growing", growing)):
        if ep.empty:
            continue
        ep = ep.copy()
        ep["served_by"] = ep.deployed
        ep["generated_earlier_in_sequence"] = served_by_earlier(ep) if mode == "growing" else False
        out.append(ep)
    per_ep = pd.concat(out, ignore_index=True)
    per_ep.to_csv(TABLES / "tab_rq2-long-episodes.rajan.csv", index=False)
    summary = per_sequence(per_ep)
    if not frozen.empty and not growing.empty:
        m = frozen[["sequence", "episode", "decision"]].merge(
            growing.assign(earlier=served_by_earlier(growing))[["sequence", "episode", "decision", "earlier"]],
            on=["sequence", "episode"], suffixes=("_frozen", "_growing"))
        spared = m[(m.decision_frozen == "defer") & m.earlier].groupby("sequence").size()
        summary["frozen_defer_growing_served_by_earlier"] = summary.apply(
            lambda r: int(spared.get(r.sequence, 0)) if r["mode"] == "growing" else np.nan, axis=1)
    summary.to_csv(TABLES / "tab_rq2-long-sequences.rajan.csv", index=False)
    print("wrote tables/tab_rq2-long-episodes.rajan.csv, tab_rq2-long-sequences.rajan.csv")
    print(summary.to_string(index=False))


def item5() -> None:
    frames = [episodes_of(s, "growing") for s in ("shift_v0_lib2", "shift_v0_lib3")]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return note("item 5: no growing episodes for shift_v0_lib2 / lib3 yet")
    ep = pd.concat(frames, ignore_index=True)
    ep.to_csv(TABLES / "tab_rq2-growing-libs-episodes.rajan.csv", index=False)
    summary = per_sequence(ep)
    summary.to_csv(TABLES / "tab_rq2-growing-libs.rajan.csv", index=False)
    print("wrote tables/tab_rq2-growing-libs.rajan.csv, tab_rq2-growing-libs-episodes.rajan.csv")
    print(summary.to_string(index=False))


def item6() -> None:
    frames = [episodes_of("pacman_v0", m) for m in ("frozen", "growing")]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return note("item 6: no pacman_v0 episodes (blocked; see the notes)")
    ep = pd.concat(frames, ignore_index=True)
    summary = per_sequence(ep)
    summary.to_csv(TABLES / "tab_rq2-pacman-sequences.rajan.csv", index=False)
    # every generation attempt at lengths 25 and 30 (ids 7 and 9): its verdict and smallest margin
    attempts = []
    for seq in sorted((SEQ / "pacman_v0" / "growing").glob("*")):
        e, c = pd.read_csv(seq / "episodes.csv"), pd.read_csv(seq / "checks.csv")
        for _, r in e[e.instance.isin([7, 9]) & e.get("generation_seconds", pd.Series(dtype=float)).notna()].iterrows():
            new = c[(c.episode == r.episode) & (c.rule_list == f"gen_t{r.episode}")]
            attempts.append({"sequence": seq.name, "episode": r.episode, "instance": r.instance,
                             "length": {7: 25, 9: 30}[int(r.instance)], "verdict": r.get("new_outcome"),
                             "margin": new.margin.iloc[0] if len(new) else np.nan,
                             "llm_calls": r.get("llm_calls"), "output_tokens": r.get("output_tokens")})
    pd.DataFrame(attempts).to_csv(TABLES / "tab_rq2-pacman-generations.rajan.csv", index=False)
    print("wrote tables/tab_rq2-pacman-sequences.rajan.csv, tab_rq2-pacman-generations.rajan.csv")


# ---------------------------------------------------------------- item 3 / 7: other LLMs

def spend(runs: List[Path], price) -> dict:
    """Tokens and dollars over every LLM call of `runs`: OpenRouter's reported cost where the calls logged it
    (llm_tasks.jsonl result.meta.usage.cost), and tokens x the listed price for all calls."""
    prompt = output = calls = 0
    reported, n_reported = 0.0, 0
    for run in runs:
        path = run / "llm_tasks.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
            res = json.loads(line).get("result") or {}
            prompt += res.get("prompt_tokens") or 0
            output += res.get("output_tokens") or 0
            calls += 1
            cost = ((res.get("meta") or {}).get("usage") or {}).get("cost")
            if cost is not None:
                reported += cost
                n_reported += 1
    return {"calls": calls, "prompt_tokens": prompt, "output_tokens": output,
            "dollars_at_list_price": (prompt * price[0] + output * price[1]) / 1e6,
            "dollars_reported": reported if n_reported else np.nan, "calls_with_reported_cost": n_reported}


def llm_data(exp, rules_of) -> Dict[str, pd.DataFrame]:
    frozen, permissive = [], []
    for name, bench in exp.spec["benchmarks"].items():
        if name not in pr.RQ1_BENCHES:
            continue
        cfg = pr.load_config(bench.base)
        dataset = bench.dataset or cfg.domain.dataset
        for label, (suffix, _, _) in LLMS.items():
            for run in pr.runs_of(bench.base + suffix, "ablations"):
                fz, perm = rules_of(name)
                frozen.append(pr.outcomes(csv_path(run, dataset, fz), bench, label, run.name))
                if perm:
                    permissive.append(pr.outcomes(csv_path(run, dataset, perm), bench, label, run.name))
    pick = lambda xs: pd.concat([x for x in xs if not x.empty], ignore_index=True) if any(
        not x.empty for x in xs) else pd.DataFrame()
    return {k: v for k, v in (("frozen", pick(frozen)), ("permissive", pick(permissive))) if not v.empty}


def coverage_tex(table: pd.DataFrame, title_col: str = "approach") -> List[str]:
    tex = [r"\toprule", r"LLM & Permissive (\%) & Frozen (\%) & Inconclusive (\%) & Timeout (\%) \\"]
    for bench, title in pr.RQ1_BENCHES.items():
        b = table[table.bench == bench]
        if b.empty:
            continue
        tex += [r"\midrule", rf"\multicolumn{{5}}{{l}}{{\emph{{{title}}}}} \\"]
        for _, r in b.iterrows():
            tex.append(f"{r[title_col]} & {pr.pct(r.get('permissive_mean'), r.get('permissive_sd'), pr.count(r.get('permissive_n')))}"
                       f" & {pr.pct(r.get('frozen_mean'), r.get('frozen_sd'), pr.count(r.get('frozen_n')))}"
                       f" & {pr.share(r.get('inconclusive_share'))} & {pr.share(r.get('timeout_share'))} \\\\")
    return tex + [r"\bottomrule"]


def item3(exp) -> None:
    data = llm_data(exp, lambda b: ("final", None) if b == "pacman" else ("determinized", "final"))
    if not data:
        return note("item 3: no certified runs of the other LLMs yet")
    table = pr.coverage_table(data, list(pr.RQ1_BENCHES), list(LLMS))
    table["llm"] = table.approach.map({k: v[1] for k, v in LLMS.items()})
    long = table.melt(id_vars=["bench", "approach", "llm", "split", "checks"], var_name="metric").dropna(subset=["value"])
    pr.write_table("tab_rq1-coverage-llm.rajan", coverage_tex(table), long.to_dict("records"))
    costs = []
    for label, (suffix, model, price) in LLMS.items():
        for bench in pr.RQ1_BENCHES:
            base = exp.spec["benchmarks"][bench].base
            runs = pr.runs_of(base + suffix, "ablations")
            if runs:
                costs.append({"llm": label, "model": model, "bench": bench, "runs": len(runs), **spend(runs, price)})
    if costs:
        df = pd.DataFrame(costs)
        df.to_csv(TABLES / "tab_rq1-llm-spend.rajan.csv", index=False)
        print("wrote tables/tab_rq1-llm-spend.rajan.csv")
        print(df.groupby("llm")[["calls", "dollars_at_list_price", "dollars_reported"]].sum().to_string())


def item3_ablation(exp) -> None:
    """NoRepair (item 3, if time remains) and Restart on the families (item 7) for the other LLMs, as T4."""
    rows, tex = [], [r"\toprule",
                     r"LLM / configuration & Certified coverage (\%) & Training pass rate (\%) & LLM calls per set \\"]
    for bench, title in pr.RQ3_BENCHES.items():
        b = exp.spec["benchmarks"][bench]
        cfg = pr.load_config(b.base)
        dataset = b.dataset or cfg.domain.dataset
        frozen = "final" if bench == "pacman" else "determinized"
        block = []
        for label, (suffix, _, _) in LLMS.items():
            full = pr.runs_of(b.base + suffix, "ablations")
            configs = {"Full": (full, frozen, False), "NoRepair": (full, "norepair_determinized", True),
                       "Restart": (pr.runs_of(b.base + suffix + "_resample", "ablations") if bench == "families"
                                   else [], frozen, False)}
            for name, (runs, rules, norepair) in configs.items():
                if not runs or not all(pr.checked(csv_path(r, dataset, rules)).exists() for r in runs):
                    continue
                cov = pr.per_list(pd.concat([pr.outcomes(csv_path(r, dataset, rules), b, name, r.name) for r in runs]))
                passed = [x for r in runs for x in pr.members_passed(r, norepair)]
                c, p = pr.stats(cov.coverage), pr.stats(passed)
                calls = [pr.calls_per_set(r) for r in runs]
                rows += [{"bench": bench, "llm": label, "configuration": name, "metric": m, "value": v,
                          "split": "validation", "runs": len(runs)}
                         for m, v in (("coverage_mean", c["mean"]), ("coverage_sd", c["sd"]), ("coverage_n", c["n"]),
                                      ("certified_per_list", cov.certified.mean()),
                                      ("training_pass_mean", p["mean"]), ("training_pass_n", p["n"]),
                                      ("llm_calls_per_set", np.mean(calls)),
                                      ("inconclusive_share", cov.inconclusive.mean()),
                                      ("timeout_share", cov.timeout.mean()))]
                block.append(f"{label}, {name} & {pr.pct(c['mean'], c['sd'], c['n'])} & "
                             f"{pr.pct(p['mean'], p['sd'], p['n'])} & ${np.mean(calls):.1f}$ \\\\")
        if block:
            tex += [r"\midrule", rf"\multicolumn{{4}}{{l}}{{\emph{{{title}}}}} \\"] + block
    if rows:
        pr.write_table("tab_rq3-ablation-llm.rajan", tex + [r"\bottomrule"], rows)
    else:
        note("items 3/7: no NoRepair or Restart checks of the other LLMs yet")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--items", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7])
    a = p.parse_args(argv)
    TABLES.mkdir(parents=True, exist_ok=True)
    exp = None
    if {3, 7} & set(a.items):
        exp = experiment(pr.RQ1_SPEC)
        pr.EXP = exp
    for item in a.items:
        print(f"== item {item}")
        {1: item1, 2: item2, 4: item4, 5: item5, 6: item6}.get(item, lambda: None)()
        if item == 3:
            item3(exp)
        if item in (3, 7) and item == max(i for i in a.items if i in (3, 7)):
            item3_ablation(exp)


if __name__ == "__main__":
    main()
