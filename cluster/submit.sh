#!/bin/bash
# Submit jobs on Fir, sized from the number of runs in flight. Run from anywhere on the login node.
#
#   bash cluster/submit.sh smoke                                   # smoke test 2 (compute node, free model)
#   bash cluster/submit.sh timing                                  # smoke test 3 (certification timing, no LLM)
#   bash cluster/submit.sh exp <file.yaml>... --parallel N --time H:MM:SS [--flags "--run --check"] [--name X]
#   bash cluster/submit.sh cmd --cpus C --mem G --time H:MM:SS [--name X] [--exclusive] -- <command...>  (--mem 0: all)
# --after takes job ids (afterany) or a full Slurm dependency such as afterok:123:456.
# Every job mails its state changes to $NOTIFY_EMAIL (config.sh; an ntfy.sh topic). Common options: --dry (print the sbatch command only), --force (allow a second LLM launcher), --after JOBID.
#
# exp: experiment files are relative to PRISM-Guided-Learning/ (e.g. configs/experiments/rq1_comparison.yaml).
# Cores = 8 x N and memory = 40 GB x N + 8 GB (one run keeps up to 8 PRISM processes with 4 GB heaps each).
# Only one job with --run may be queued or running at a time (--force overrides): two launchers that both see
# the same repeat as missing would both start it in the same directory. Put experiments that share cells
# (e.g. E1's core arm and E4) in ONE exp job; run_experiment.py then runs each configuration once.
# Every submission is appended to out/cluster/submissions.csv.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/config.sh"
GL_DIR="$REPO_DIR/PRISM-Guided-Learning"
CLUSTER_OUT="$GL_DIR/out/cluster"
mkdir -p "$CLUSTER_OUT/slurm"
cd "$REPO_DIR"

kind=${1:?usage: submit.sh smoke|timing|exp|cmd ...}; shift
parallel=1 time="" cpus="" mem="" name="" flags="--run --check" dry=0 force=0 after="" exclusive=0 exps=() cmd=()
while [ $# -gt 0 ]; do
    case $1 in
        --parallel) parallel=$2; shift 2 ;;
        --time) time=$2; shift 2 ;;
        --cpus) cpus=$2; shift 2 ;;
        --mem) mem=$2; shift 2 ;;
        --name) name=$2; shift 2 ;;
        --flags) flags=$2; shift 2 ;;
        --after) after=$2; shift 2 ;;
        --dry) dry=1; shift ;;
        --force) force=1; shift ;;
        --exclusive) exclusive=1; shift ;;
        --) shift; cmd=("$@"); break ;;
        *) exps+=("$1"); shift ;;
    esac
done

args=(--account="$FIR_ACCOUNT" --output="$CLUSTER_OUT/slurm/%x-%j.out" --chdir="$REPO_DIR")
# --after: a Slurm dependency ("afterok:123:456", "afterany:123") or bare job ids ("123" or "123:456": afterany)
if [ -n "$after" ]; then
    [[ $after == *after* ]] && args+=(--dependency="$after") || args+=(--dependency="afterany:$after")
fi
[ $exclusive = 1 ] && args+=(--exclusive)
[ -n "$NOTIFY_EMAIL" ] && args+=(--mail-user="$NOTIFY_EMAIL" --mail-type="$NOTIFY_TYPES")
case $kind in
    smoke)  script=cluster/smoke/02_compute.sbatch; name=${name:-smoke} ;;
    timing) script=cluster/smoke/03_timing.sbatch; name=${name:-timing}
            [ -n "${BENCHES:-}" ] && args+=(--export="ALL,BENCHES=$BENCHES") ;;
    exp)
        [ ${#exps[@]} -gt 0 ] || { echo "exp: name at least one experiment file"; exit 1; }
        [ -n "$time" ] || { echo "exp: --time is required (see cluster/README.md for estimates)"; exit 1; }
        for e in "${exps[@]}"; do
            [ -f "$GL_DIR/$e" ] || { echo "no such experiment file: $GL_DIR/$e"; exit 1; }
        done
        script=cluster/jobs/experiment.sbatch
        base=$(basename "${exps[0]}" .yaml)
        if [[ " $flags " == *" --run "* ]]; then
            name=${name:-launch-$base}
            [ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY is not set in this shell"; exit 1; }
            busy=$(squeue -h -u "$USER" -o "%j %i %T" 2>/dev/null | awk '$1 ~ /^launch-/' || true)
            if [ -n "$busy" ] && [ $force = 0 ]; then
                echo "an LLM launcher is already queued or running:"; echo "$busy"
                echo "wait for it, chain with --after <jobid>, or add --force if the cells cannot overlap"; exit 1
            fi
        else
            name=${name:-exp-$base}
        fi
        args+=(--cpus-per-task=$(( CORES_PER_RUN * parallel )) --mem="$(( MEM_GB_PER_RUN * parallel + MEM_GB_BASE ))G"
               --time="$time"
               --export="ALL,EXPS=${exps[*]},MAX_PARALLEL=$parallel,TRANSFER_WORKERS=2,FLAGS=$flags") ;;
    cmd)
        [ ${#cmd[@]} -gt 0 ] || { echo "cmd: give the command after --"; exit 1; }
        [ -n "$time" ] && [ -n "$cpus" ] && [ -n "$mem" ] || { echo "cmd: --cpus, --mem (GB) and --time are required"; exit 1; }
        script=cluster/jobs/command.sbatch; name=${name:-cmd}
        args+=(--cpus-per-task="$cpus" --mem="${mem}G" --time="$time") ;;
    *) echo "unknown kind: $kind"; exit 1 ;;
esac
args+=(--job-name="$name")

echo "sbatch ${args[*]} $script ${cmd[*]:-}"
[ $dry = 1 ] && exit 0
out=$(sbatch --parsable "${args[@]}" "$script" "${cmd[@]}")
jobid=${out%%;*}
echo "submitted $name as job $jobid  (log: $CLUSTER_OUT/slurm/$name-$jobid.out)"
[ -f "$CLUSTER_OUT/submissions.csv" ] || echo "time,job_id,name,kind,git_commit,args" > "$CLUSTER_OUT/submissions.csv"
echo "$(date -Is),$jobid,$name,$kind,$(git rev-parse --short HEAD 2>/dev/null),\"${args[*]} ${exps[*]:-} ${cmd[*]:-}\"" \
    >> "$CLUSTER_OUT/submissions.csv"
