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
| `campaign.sh` | login node | the campaign in waves: `plan`, `wave1`, `wave2`, `baselines`, `report` |
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

## 2. The campaign

| Job | Experiments | Size | Estimated wall | `--time` |
|---|---|---|---|---|
| `campaign.sh wave1` | E4 (`rq1_comparison`, 14 runs) + E1 (`rq2_new_loop_feedback`, ~30 runs), one launcher | 12 runs in flight: 96 cores, 488 GB | 4–6 h | 12h |
| `campaign.sh wave2` | E2 (`rq2_new_loop`, 30 runs) + the RQ1 cells added later (E6 Sonnet, E10 UUV) | 96 cores, 488 GB | 4–6 h | 12h |
| `campaign.sh baselines` | RQ1 baselines without a certification (E10) | 16 cores | 1–3 h | 6h |
| `submit.sh cmd ...` | E3 (`adapt.py`), E5, E7, E8: no LLM | 8–16 cores each | 1–4 h | 3–12h |
| `campaign.sh report` | `--check` everything, rebuild reports and the index | 8 cores | <1 h | 3h |

Per-run figures (training only, from the existing core runs): families 12–13 min, grids and Pac-Man 19–22 min,
MER trained on one instance 14–33 min, MER 39–69 min. Certification adds an unmeasured amount; `03` measures
it. Sonnet runs (one chain: 33 min) are probably 1–2 h with the full chain count.

```bash
bash cluster/campaign.sh plan            # login node: per cell, the runs found and what --run would launch
bash cluster/campaign.sh wave1 --dry     # print the sbatch command
bash cluster/campaign.sh wave1           # Asher has approved E4 and E1? Then submit
bash cluster/campaign.sh wave2           # queued behind wave1 automatically (Slurm dependency)
```

The rules the scripts enforce, and why:
- **Only one `--run` launcher at a time.** Two launchers that both see a repeat as missing would both start it
  in the same directory. Experiments that share cells (E1's core arm is E4's core cell) go in one job, and
  `run_experiment.py` runs each configuration once. `submit.sh` refuses a second launcher unless `--force`.
- **8 cores and 40 GB per run in flight** (`--parallel N`). That's `prism.max_concurrent: 8` PRISM processes with
  4 GB heaps each. These settings are not part of a run's identity, so changing them never splits a cell.
- **`TMPDIR=$SLURM_TMPDIR`.** PRISM's process slots are lock files in the temp dir. Per job, they cap that job
  alone, not every job on the node.
- **`--time` with 2–3× headroom.** A paid run that Slurm kills midway has to be deleted and rerun (handover rules).
  Shorter limits do start sooner: 3h and 12h jobs backfill well, 7-day jobs wait.

**If a job dies:** the runs that finished are kept. Find the unfinished run directories, delete them (handover
rule), then submit the same wave again; finished runs are skipped:

```bash
cd PRISM-Guided-Learning
for d in out/results/{core,ablations}/*/seed_*; do [ -f "$d/SYMBOLIC_results.parquet" ] || echo "$d"; done
```

Other jobs: anything else runs through `cmd`, from `PRISM-Guided-Learning/`, e.g.

```bash
bash cluster/submit.sh cmd --cpus 16 --mem 64 --time 3:00:00 --name e10-gate -- python src/calibrate.py --domain uuv
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
