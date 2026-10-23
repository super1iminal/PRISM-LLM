#!/bin/bash
# Items 3 and 7 (Rajan's checklist): certify every finished run of the other LLMs' core conditions under rule 2
# (src/recertify.py: interval iteration at 1e-9, LP where it stalls, inconclusive only within the band), on each
# benchmark's validation held-out targets as in configs/experiments/rq1_comparison.yaml, writing the per-check CSVs in
# Asher's format (<run>/transfer_exact_<dataset>[_suffix].csv) that viz/paper_results.py and viz/rajan_results.py read.
#   bash cluster/jobs/certify_llm.sh <suffix> [rules...]        e.g. _luna   (default rules: frozen and permissive)
#   bash cluster/jobs/certify_llm.sh _luna norepair_determinized             (NoRepair, if time remains)
#   CONDITIONS="port_fam_history" bash cluster/jobs/certify_llm.sh _luna_resample   (item 7's restart arm)
# Run from PRISM-Guided-Learning/ inside a job (cluster/submit.sh cmd ...). Finished checks are kept (recertify merges),
# runs still in progress (no results parquet) are skipped. WORKERS = targets certified at once per run (default 2).
set -uo pipefail
suffix=${1?usage: certify_llm.sh <suffix> [rules...] (suffix "" for a condition of its own, e.g. port_uuv)}; shift
rules_arg=("$@")
declare -A TARGETS=([port_ts_history]="12-19" [port_fam_history]="16-35,76-95,136-155"
                    [port_pacman_narrow_core]="0-7,9" [port_mer]="0-8" [port_mer_n]="0-8"
                    [port_uuv]="0-29"           # item 8: the UUV family's training and validation members
                    [port_uuv_sets]="0-29" [port_uuv_sets_words]="0-29")   # Asher's UUV plan, steps 4 and 5
declare -A DATA=([port_pacman_narrow_core]=pacman_transfer.csv [port_mer]=mer_transfer.csv
                 [port_mer_n]=mer_n_transfer.csv)
status=0
for base in ${CONDITIONS:-port_ts_history port_fam_history port_pacman_narrow_core port_mer port_mer_n}; do
    if [ ${#rules_arg[@]} -gt 0 ]; then rules=("${rules_arg[@]}")
    elif [ "$base" = port_pacman_narrow_core ]; then rules=(final)          # the narrowed lists, already resolved
    else rules=(determinized final); fi                                     # frozen, permissive
    for run in out/results/ablations/"$base$suffix"/seed_*; do
        [ -f "$run/SYMBOLIC_results.parquet" ] || { echo "skip $run (not finished)"; continue; }
        for r in "${rules[@]}"; do
            echo "== $run --rules $r"
            python src/recertify.py run "$run" --rules "$r" --targets "${TARGETS[$base]}" \
                ${DATA[$base]:+--data "${DATA[$base]}"} --workers "${WORKERS:-2}" || status=1
        done
    done
done
exit $status
