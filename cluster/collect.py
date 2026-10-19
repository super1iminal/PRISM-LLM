"""Tidy tables from the raw data pulled back from Fir (cluster/pull.sh), for plots and the timing estimates.

    python cluster/collect.py [--since 2026-10-19]     # from the repo root, with the repo's venv

Writes PRISM-Guided-Learning/out/cluster/tables/:
  jobs.csv       one row per Slurm job: its record (meta.json), Slurm's accounting (sacct.csv) and the peaks of its
                 resource samples (resources.csv)
  resources.csv  every resource sample of every job, with its job id (memory and CPU over time)
  runs.csv       one row per LLM run under out/results/ (core, ablations, smoke): model, condition settings, sets,
                 tokens, LLM and PRISM time, wall time, and the transfer CSVs it has
  rounds.csv     one row per run x training set x chain x round: tokens, calls, invalid answers, LLM and PRISM time
Nothing is changed in the runs themselves; re-run after every pull.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

GL = Path(__file__).resolve().parents[1] / "PRISM-Guided-Learning"
OUT = GL / "out"
CLUSTER = OUT / "cluster"


def jobs_tables():
    rows, samples = [], []
    for meta_path in sorted((CLUSTER / "jobs").glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        job = meta_path.parent.name
        row = {"job": job, "job_id": meta.get("slurm", {}).get("SLURM_JOB_ID"), "state": meta.get("state"),
               "exit_code": meta.get("exit_code"), "start": meta.get("start"), "elapsed_s": meta.get("elapsed_s"),
               "cpus": meta.get("slurm", {}).get("SLURM_CPUS_PER_TASK"),
               "mem_mb": meta.get("slurm", {}).get("SLURM_MEM_PER_NODE"), "time_limit": meta.get("time_limit"),
               "git_commit": meta.get("git_commit"), "command": meta.get("command")}
        res = meta_path.parent / "resources.csv"
        if res.exists():
            r = pd.read_csv(res)
            if not r.empty:
                row.update(peak_mem_mb=r.cgroup_mem_mb.max(), peak_java=r.java_n.max(),
                           peak_java_rss_mb=r.java_rss_mb.max(), mean_load1=r.load1.mean())
                samples.append(r.assign(job=job))
        rows.append(row)
    jobs = pd.DataFrame(rows)
    sacct = CLUSTER / "sacct.csv"
    if sacct.exists() and not jobs.empty:
        acc = pd.read_csv(sacct, sep="|", dtype=str)
        acc = acc[~acc.JobID.str.contains(r"\.")]          # the job line, not its steps
        jobs = jobs.merge(acc.add_prefix("sacct_"), left_on="job_id", right_on="sacct_JobID", how="left")
    return jobs, (pd.concat(samples, ignore_index=True) if samples else pd.DataFrame())


def run_tables(since):
    runs, rounds = [], []
    for cfg_path in sorted((OUT / "results").glob("**/seed_*/config.json")):
        run_dir = cfg_path.parent
        if since and pd.Timestamp(cfg_path.stat().st_mtime, unit="s") < since:
            continue
        cfg = json.loads(cfg_path.read_text())
        llm, planner = cfg.get("llm", {}), cfg.get("planner", {})
        row = {"run": str(run_dir.relative_to(OUT / "results")), "root": run_dir.relative_to(OUT / "results").parts[0],
               "condition": cfg.get("condition"), "model": llm.get("model"), "chains": planner.get("chains"),
               "max_rounds": planner.get("max_rounds"), "retry": planner.get("retry"),
               "workers": cfg.get("run", {}).get("workers"), "finished": any(run_dir.glob("*_results.parquet")),
               "transfer_csvs": ";".join(sorted(p.name for p in run_dir.glob("transfer_*.csv")))}
        totals = {"sets": 0, "llm_calls": 0, "prompt_tokens": 0, "output_tokens": 0, "llm_time_s": 0.0,
                  "prism_time_s": 0.0, "max_set_time_s": 0.0, "sum_set_time_s": 0.0}
        for sample in sorted((run_dir / "outputs").glob("sample_*.json")):
            rec = json.loads(sample.read_text())
            totals["sets"] += 1
            totals["max_set_time_s"] = max(totals["max_set_time_s"], rec.get("total_time") or 0)
            totals["sum_set_time_s"] += rec.get("total_time") or 0
            for it in rec.get("iterations", []):
                r = {"run": row["run"], "set": sample.stem, "chain": it.get("chain"), "round": it.get("iteration"),
                     "mode": it.get("mode"), "llm_calls": it.get("llm_calls"),
                     "invalid_answers": it.get("invalid_answers"), "prompt_tokens": it.get("llm_prompt_tokens"),
                     "output_tokens": it.get("llm_output_tokens"), "llm_time_s": it.get("llm_time"),
                     "llm_wall_time_s": it.get("llm_wall_time"), "prism_time_s": it.get("prism_time"),
                     "iteration_time_s": it.get("iteration_time"), "best_case_success": it.get("best_case_success"),
                     "worst_case_success": it.get("worst_case_success")}
                rounds.append(r)
                totals["llm_calls"] += r["llm_calls"] or 0
                totals["prompt_tokens"] += r["prompt_tokens"] or 0
                totals["output_tokens"] += r["output_tokens"] or 0
                totals["llm_time_s"] += r["llm_time_s"] or 0
                totals["prism_time_s"] += r["prism_time_s"] or 0
        runs.append({**row, **totals})
    return pd.DataFrame(runs), pd.DataFrame(rounds)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--since", default=None, help="only runs whose config.json is newer than this date")
    a = p.parse_args()
    since = pd.Timestamp(a.since) if a.since else None
    out = CLUSTER / "tables"
    out.mkdir(parents=True, exist_ok=True)
    jobs, samples = jobs_tables()
    runs, rounds = run_tables(since)
    for name, df in [("jobs", jobs), ("resources", samples), ("runs", runs), ("rounds", rounds)]:
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"{out / name}.csv: {len(df)} rows")


if __name__ == "__main__":
    main()
