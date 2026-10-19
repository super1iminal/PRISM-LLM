#!/bin/bash
# One launcher job: src/run_experiment.py over one or more experiment files, in one shared job pool.
# Submit through cluster/submit.sh (it sizes cores and memory from --parallel); the settings arrive as variables:
#   EXPS              experiment files, relative to PRISM-Guided-Learning/ (space-separated)
#   MAX_PARALLEL      runs/checks at once (run_experiment.py --max-parallel)
#   TRANSFER_WORKERS  targets certified at once per check (--transfer-workers)
#   FLAGS             e.g. "--run --check" or "--baselines"
# Interrupted? Submit the same thing again: finished runs are skipped. A run that died midway must be deleted
# first (out/results/<root>/<run>/seed_none_<k>/ without SYMBOLIC_results.parquet; see cluster/README.md).
#SBATCH --job-name=exp
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh || exit 1
source cluster/lib/job.sh
: "${EXPS:?EXPS not set}" "${MAX_PARALLEL:=6}" "${TRANSFER_WORKERS:=2}" "${FLAGS:=--run --check}"
cmd=(python src/run_experiment.py $EXPS $FLAGS --max-parallel "$MAX_PARALLEL" --transfer-workers "$TRANSFER_WORKERS")
job_start "${cmd[*]}"
if [[ " $FLAGS " == *" --run "* && -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "OPENROUTER_API_KEY is not set: submit from a shell where it is exported"; job_end 2; exit 2
fi
cd "$GL_DIR"
"${cmd[@]}" 2>&1 | tee "$JOB_DIR/stdout.log"
code=${PIPESTATUS[0]}
# Snapshot what the store now holds (run counts per cell), for the record pulled back.
python src/run_experiment.py $EXPS > "$JOB_DIR/plan_after.txt" 2>&1
job_end "$code"
exit "$code"
