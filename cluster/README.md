# Running the SEAMS experiments on Fir

Scripts to run the experiments (E1–E10) on the Alliance cluster **Fir**. Code goes up with `rsync`, results
come back with `rsync`, and you submit every job by hand from a Fir login node. Nothing here holds a secret:
the OpenRouter key stays in your shell as `OPENROUTER_API_KEY`, and Slurm passes it to the jobs.

| File | Where it runs | What it does |
|---|---|---|
| `config.sh` | everywhere | **Edit first**: username, Slurm account, install root on Fir, smoke-test model |
| `push.sh` / `pull.sh` | your machine | rsync code up / all of `out/` back |
| `setup.sh` | Fir login node, once | modules, venv, PRISM 4.10.1 |
| `env.sh` | sourced by every job | modules, venv, PRISM on PATH, `TMPDIR=$SLURM_TMPDIR`, one Java GC thread per JVM |
| `smoke/01_login.sh` | login node | versions, PRISM on a tiny MDP, the repo's PRISM tests, the key, one free LLM request |
| `smoke/02_compute.sbatch` | compute node, 1h | test suite, internet, one free request, a whole toy run plus its certification |
| `smoke/03_timing.sbatch` | compute node, 3h | certification time per benchmark on copies of core runs (no LLM) |
| `submit.sh` | login node | sizes and submits `smoke`, `timing`, `exp` (run_experiment.py) and `cmd` jobs |
| `checklist.sh` | login node | Rajan's SEAMS checklist, one subcommand per item (`item1` ... `item7`, `report`) |
| `checks/` | anywhere | `request_dump.py` (rule 1a: Haiku request byte-identical), `smoke_report.py` (item 3 smoke tests) |
| `jobs/certify_llm.sh` | inside a job | rule-2 certification of the other LLMs' runs (items 3, 7) |
| `sacct.sh` | login node | Slurm accounting for every submitted job, to `out/cluster/sacct.csv` |
| `collect.py` | your machine | tidy CSVs from everything pulled back (`out/cluster/tables/`) |

## 0. Once

`config.sh` has the username (`maghera`), the account (`def-mmehride`), the install root (default
`/project/def-mmehride/maghera`) and the ssh host, `fir`, an alias from your `~/.ssh/config`. Every
`ssh`/`rsync` call goes to `fir`, so the user, keys and any ControlMaster set there apply. Fir requires MFA, so a
ControlMaster in that entry (`ControlMaster auto`, `ControlPath ~/.ssh/cm-%r@%h:%p`, `ControlPersist 8h`) saves
re-authenticating for every rsync.

Then:

```bash
bash cluster/push.sh --with-results       # local: code plus the existing runs (E1 and E4 reuse the core runs)
ssh fir
cd /project/def-mmehride/maghera/PRISM-LLM
bash cluster/setup.sh                     # ~10 min: modules, venv, PRISM; the log is in $FIR_ROOT/setup.log
```

If a module doesn't load, `module spider java` / `module spider python` lists what Fir has. Set
`JAVA_MODULES` / `PYTHON_MODULES` in `config.sh` and re-run `setup.sh`; it skips what already exists.

## 1. Smoke tests (free model, no credits needed)

The free model is `nvidia/nemotron-3-ultra-550b-a55b:free`. It has no structured outputs, so the smoke condition
sets `llm.openrouter.structured: false`: the schema goes in the prompt, and the JSON is cut out of the reply.
A key with no credits gets 50 free requests a day (20 a minute). The three smoke tests together use about 6.

```bash
export OPENROUTER_API_KEY=sk-or-...       # in the shell you submit from; never in a file in the repo
bash cluster/smoke/01_login.sh            # ~2 min, prints SMOKE 1 PASSED / FAILED
bash cluster/submit.sh smoke              # smoke 2, ~30 min (the test suite is most of it)
bash cluster/submit.sh timing             # smoke 3, 1-3 h; BENCHES="families grids" for a quicker one
squeue -u $USER                           # state; logs in PRISM-Guided-Learning/out/cluster/slurm/
```

What passing means:
- `01`: PRISM gives 1.0 and 0.5 on the tiny MDP, the PRISM, config and OpenRouter tests pass, and `llm_ping.json`
  shows `parsed: true` and the key's limits.
- `02`: `out/cluster/jobs/<id>_smoke/summary.json` has `"passed": true`. It records each step's seconds and
  the LLM calls and tokens of the toy run (at most 4 calls). The run itself is in `out/results/smoke/fir/`.
  A free model may answer badly; what's being tested is the plumbing, not the rules.
- `03`: `out/cluster/jobs/<id>_timing/timing.csv`, seconds per benchmark and rules variant. This is the
  certification cost the repo doesn't record, so use it to correct the `--time` limits below.

Before the paid runs, once the key has credits, check the real route with one request (about $0.01):

```bash
python cluster/smoke/llm_ping.py --model anthropic/claude-haiku-5.5 --providers anthropic --structured --think \
    --out PRISM-Guided-Learning/out/cluster/smoke/ping_haiku.json
```

## 2. Rajan's checklist

The items, their rules and what each produces are in `PRISM-Guided-Learning/out/paper/RESULTS-NOTES-rajan.md`. Asher's
list (E1/E2 Haiku restarts, re-certification, the sealed set, Sonnet, ...) is not run from here (checklist rule 6),
so the earlier `campaign.sh` waves are gone. Order, as the checklist sets it:

| Step | Command | Size | Needs |
|---|---|---|---|
| 1 | `bash cluster/checklist.sh item1` (or `EXCLUSIVE=1 ...` for a node of our own) | 3 jobs × 4 cores | PRISM only; first, and alone among our jobs |
| 2 | `bash cluster/checklist.sh item3-smoke luna` and `... gemini` | 8 cores each | paid key; after item 1 (or 3 h into it) |
| 3 | `bash cluster/checklist.sh item3-full luna` / `gemini` (each model that passed) | 24 cores, 120 GB | paid key |
| 3 | `bash cluster/checklist.sh item2` | 12 cores | PRISM only; alongside item 3 |
| 3 | `bash cluster/checklist.sh item4 frozen`, then `item4 growing` | 10 cores | growing: paid key (Haiku) |
| 3 | `bash cluster/checklist.sh item5` | 16 cores | paid key (Haiku) |
| 4 | `bash cluster/checklist.sh item3-certify luna` / `gemini` (after its full runs) | 12 cores | PRISM only |
| 5 | `bash cluster/checklist.sh item7 <model>` | 8 cores | only once item 7's two conditions hold |
| - | `bash cluster/checklist.sh item6 ...` | | blocked at this commit (see the notes) |
| any | `bash cluster/checklist.sh report` | 4 cores | builds Rajan's tables (`viz/rajan_results.py`) |

Add `--dry` to see the sbatch line. Every job writes its record (commit, hardware, our other running jobs) to
`out/cluster/jobs/`. A job's PRISM cap is `prism.max_concurrent` (8; item 3's full runs: 20) slots in its own
`TMPDIR`, shared by everything in the job, generation runs' `run.workers` included.

Freeze: **Wednesday 21 October, 23:59 Toronto time.** A run that is not complete and clean by then stays out of the
tables; don't extend a job to rescue it. **If a run dies midway:** stop it, delete its directory (handover rule) and
resubmit. Finished runs are skipped:

```bash
cd PRISM-Guided-Learning
for d in out/results/ablations/*_{luna,gemini}*/seed_*; do [ -f "$d/SYMBOLIC_results.parquet" ] || echo "$d"; done
```

## 3. Getting the data back

```bash
ssh fir 'cd /project/def-mmehride/maghera/PRISM-LLM && bash cluster/sacct.sh'
bash cluster/pull.sh                     # everything under out/ (add "cluster" for the job records only)
python cluster/collect.py                # out/cluster/tables/{jobs,resources,runs,rounds}.csv
```

Everything raw comes back:
- **Each run:** `config.json`, `outputs/sample_*.json` (per round: rules, values, tokens, LLM and PRISM time),
  `llm_tasks.jsonl` (every prompt and reply), the results parquet, the transfer CSVs and the run's log.
- **Each job:** `out/cluster/jobs/<id>_<name>/` holds `meta.json` (node, sizes, git commit, modules, exit code),
  `resources.csv` (memory, PRISM processes and load every 30 s) and `stdout.log`. Slurm logs are in
  `out/cluster/slurm/`, and the submission history is in `out/cluster/submissions.csv`.
- **Reports and figures:** `out/results/experiments/` holds what `run_experiment.py` rebuilt on Fir. Commit
  these, the parquet files and the `outputs/` JSON from your machine (logs are gitignored).
