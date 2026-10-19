#!/bin/bash
# The SEAMS campaign as a few Slurm jobs (cluster/README.md has the schedule and the estimates).
#
#   bash cluster/campaign.sh plan          # login node: what each experiment file has, and what --run would launch
#   bash cluster/campaign.sh wave1 [--dry] # E4 + E1: one launcher, so E1's core arm and E4 share their runs
#   bash cluster/campaign.sh wave2 [--dry] # E2 + the RQ1 cells added later (E6 Sonnet, E10); queued after wave1
#   bash cluster/campaign.sh baselines     # RQ1 baselines missing a certification (E10's new benchmark), no LLM
#   bash cluster/campaign.sh report        # certify anything unchecked, rebuild every report and the index
#
# Files a wave names but that do not exist yet are skipped with a warning, so a wave can be submitted as soon as
# its first experiment file is written. Override sizes with PARALLEL=... TIME=... in the environment.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/config.sh"
GL_DIR="$REPO_DIR/PRISM-Guided-Learning"
X=configs/experiments

E4=$X/rq1_comparison.yaml             # core cell at 5 repeats; later also E6 (Sonnet cells) and E10 (UUV benchmark)
E1=$X/rq2_new_loop_feedback.yaml      # feedback vs restarts at equal cost (new loop)
E2=$X/rq2_new_loop.yaml               # attribution of the new loop's parts

existing() {                          # the files that exist, with a warning for the others
    local f
    for f in "$@"; do
        if [ -f "$GL_DIR/$f" ]; then echo "$f"; else echo "skipping $f (not written yet)" >&2; fi
    done
}
launcher_job() {                      # id of a queued or running LLM launcher, if any
    squeue -h -u "$USER" -o "%j %i" 2>/dev/null | awk '$1 ~ /^launch-/ {print $2; exit}'
}

what=${1:?usage: campaign.sh plan|wave1|wave2|baselines|report [--dry]}; shift
case $what in
    plan)
        source "$here/env.sh"
        cd "$GL_DIR"
        mapfile -t files < <(existing "$E4" "$E1" "$E2")
        python src/run_experiment.py "${files[@]}" ;;
    wave1)
        mapfile -t files < <(existing "$E4" "$E1")
        bash "$here/submit.sh" exp "${files[@]}" --parallel "${PARALLEL:-12}" --time "${TIME:-12:00:00}" \
            --name launch-wave1 "$@" ;;
    wave2)
        mapfile -t files < <(existing "$E2" "$E4")
        prev=$(launcher_job || true)
        bash "$here/submit.sh" exp "${files[@]}" --parallel "${PARALLEL:-12}" --time "${TIME:-12:00:00}" \
            --name launch-wave2 ${prev:+--after "$prev" --force} "$@" ;;
    baselines)
        bash "$here/submit.sh" exp "$E4" --flags "--baselines" --parallel "${PARALLEL:-2}" \
            --time "${TIME:-6:00:00}" --name exp-baselines "$@" ;;
    report)
        mapfile -t files < <(existing "$E4" "$E1" "$E2")
        prev=$(launcher_job || true)
        bash "$here/submit.sh" cmd --cpus 8 --mem 40 --time "${TIME:-3:00:00}" --name report \
            ${prev:+--after "$prev"} "$@" -- \
            bash -c "python src/run_experiment.py ${files[*]} --check && python src/run_experiment.py --index" ;;
    *) echo "unknown: $what"; exit 1 ;;
esac
