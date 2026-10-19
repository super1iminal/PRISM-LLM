#!/bin/bash
# Any command in the run environment, from PRISM-Guided-Learning/ (baselines, E5/E7/E8 scripts, report rebuilds).
#   bash cluster/submit.sh cmd --cpus 8 --mem 32 --time 3:00:00 --name e7 -- python src/some_script.py ...
#SBATCH --job-name=cmd
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh || exit 1
source cluster/lib/job.sh
job_start "$*"
cd "$GL_DIR"
"$@" 2>&1 | tee "$JOB_DIR/stdout.log"
code=${PIPESTATUS[0]}
job_end "$code"
exit "$code"
