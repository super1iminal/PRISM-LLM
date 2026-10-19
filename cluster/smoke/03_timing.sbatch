#!/bin/bash
# Smoke test 3, no LLM: how long certification (run_experiment.py --check) takes on Fir, the one cost the repo
# does not record. Copies one existing core run per benchmark into out/results/smoke/fir/timing_<job>/ (the core
# runs are never touched), drops its transfer CSVs, and certifies it on the benchmark's held-out targets as
# written and with ties open, with 2 targets at a time (run_experiment.py's --transfer-workers default).
# Pac-Man includes horizons 25 and 30 (3.7M and 8.3M states), the heaviest PRISM work in the plan.
# Submit:   bash cluster/submit.sh timing          (BENCHES="families grids" to run fewer)
# Output: out/cluster/jobs/<job>_timing/timing.csv (benchmark, rules, seconds, exit code) plus the usual record.
#SBATCH --job-name=timing
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=03:00:00
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh || exit 1
source cluster/lib/job.sh
job_start "timing"
cd "$GL_DIR"

# benchmark -> core run, targets, dataset (from configs/experiments/rq1_comparison.yaml)
declare -A RUN=([families]=port_fam_history [grids]=port_ts_history [pacman]=port_pacman_narrow_core
                [mer]=port_mer [mer_one]=port_mer_one)
declare -A TARGETS=([families]="16-35,76-95,136-155" [grids]="12-19" [pacman]="0-7,9" [mer]="0-8" [mer_one]="0-8")
declare -A DATA=([families]="" [grids]="" [pacman]="pacman_transfer.csv" [mer]="mer_transfer.csv"
                 [mer_one]="mer_transfer.csv")
BENCHES=${BENCHES:-"families grids pacman mer mer_one"}
dest="out/results/smoke/fir/timing_${SLURM_JOB_ID:-local}"
csv="$JOB_DIR/timing.csv"
echo "benchmark,run,rules,targets,workers,seconds,exit_code" > "$csv"

for b in $BENCHES; do
    src="out/results/core/${RUN[$b]}/seed_none_1"
    [ -d "$src" ] || { echo "missing $src (push with --with-results?)"; continue; }
    copy="$dest/${RUN[$b]}/seed_none_1"
    mkdir -p "$(dirname "$copy")" && rm -rf "$copy" && cp -r "$src" "$copy"
    rm -f "$copy"/transfer_*.csv
    for rules in final open; do
        t0=$(date +%s)
        python src/transfer.py "$copy" --targets "${TARGETS[$b]}" --rules "$rules" --workers 2 \
            ${DATA[$b]:+--data "${DATA[$b]}"} > "$JOB_DIR/transfer_${b}_${rules}.log" 2>&1
        code=$?
        echo "$b,${RUN[$b]},$rules,\"${TARGETS[$b]}\",2,$(( $(date +%s) - t0 )),$code" | tee -a "$csv"
    done
done
job_end 0
