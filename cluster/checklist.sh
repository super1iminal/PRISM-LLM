#!/bin/bash
# Rajan's SEAMS checklist (out/paper/RESULTS-NOTES-rajan.md) as Slurm jobs on Fir. Run from the repo root on the login
# node; every job goes through cluster/submit.sh (job record, ntfy mail). Add --dry to any command to print the sbatch
# line only. Order and overlap, as the checklist sets them:
#   item1 first and alone among our jobs (its times are the point); then item3 smoke tests; item3 full runs as soon
#   as item 1 is done, alongside items 2 and 4-6.
#
#   bash cluster/checklist.sh item1                  # synthesis vs certification time: 3 jobs (families, grids, pacman)
#   EXCLUSIVE=1 bash cluster/checklist.sh item1      #   ... or one whole node to ourselves, the 3 parts side by side
#   bash cluster/checklist.sh item2                  # 10 random sequences x 3 libraries, frozen (PRISM only)
#   bash cluster/checklist.sh item3-smoke luna|gemini    # one training set of port_fam_history_<model>, then its report
#   bash cluster/checklist.sh item3-full luna|gemini     # the 5 core conditions x 3 runs
#   bash cluster/checklist.sh item3-certify luna|gemini [norepair_determinized]   # rule 2 on the validation targets
#   bash cluster/checklist.sh item4 frozen|growing   # long_v0: 2 sequences x 36 (growing: Haiku 5.5)
#   bash cluster/checklist.sh item5                  # shift_v0 growing with seed libraries 2 and 3 (Haiku 5.5)
#   bash cluster/checklist.sh item6 frozen|growing   # pacman_v0: 3 sequences x 12 (growing: Haiku 5.5)
#   bash cluster/checklist.sh item7 luna|gemini      # restart arm on the families (only once item 7's conditions hold)
#   bash cluster/checklist.sh item8-baselines        # UUV family: trees (raw, features), PPO, hand-written list (no LLM)
#   bash cluster/checklist.sh item8-full             # UUV family: 3 Haiku runs of port_uuv, then their certification
#   Asher's UUV plan (item 8 follow-up), steps 3-7 (out/paper/RESULTS-NOTES-rajan.md, item 8), in this order:
#   bash cluster/checklist.sh item8-repick           # step 3: the pick with repair on item 8's runs (PRISM only)
#   bash cluster/checklist.sh item8-sets             # step 4: 3 runs of port_uuv_sets (Haiku 5.5), then certified
#   bash cluster/checklist.sh item8-words            # step 5: 3 runs of port_uuv_sets_words (Haiku 5.5), certified
#   bash cluster/checklist.sh item8-resample         # step 6: the Restart arm, 3 runs of port_uuv_resample (Haiku)
#   bash cluster/checklist.sh item8-baselines2       # step 6: PPO raw, random policies, one instance, PPO refits
#   bash cluster/checklist.sh item8-curves           # step 6: OneShot and budget curves (after repick and resample)
#   SEALED=yes bash cluster/checklist.sh item8-sealed   # step 7, last: every controller on the sealed members 30-41
#   REPAIR_SCORE=members_first (environment, for item8-repick/-sets/-words/-resample): the pick's repair objective
#   (planner.repair_score) instead of the conditions' capped_sum; same for every step, or the rows do not compare.
#   bash cluster/checklist.sh report                 # viz/rajan_results.py (Rajan's tables; Asher rebuilds his own)
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/config.sh"
cd "$REPO_DIR"
what=${1:?usage: checklist.sh item1|item2|item3-smoke|item3-full|item3-certify|item4|item5|item6|item7|item8-...|report}; shift
dry=(); args=()
for a in "$@"; do if [ "$a" = --dry ]; then dry=(--dry); else args+=("$a"); fi; done
# AFTER (environment): a Slurm dependency for every job this call submits (cluster/queue_all.sh sets it)
submit() { bash "$here/submit.sh" cmd "${dry[@]}" ${AFTER:+--after "$AFTER"} "$@"; }
need_key() { [ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY is not set in this shell"; exit 1; }; }
model() {
    case ${1:-} in luna|gemini) echo "_$1" ;; *) echo "name the model: luna or gemini" >&2; exit 1 ;; esac
}
item1_running() { squeue -h -u "$USER" -o "%j" 2>/dev/null | grep -q '^item1' && echo yes || true; }
warn_item1() {
    if [ "$(item1_running)" = yes ]; then
        echo "note: item 1 is still queued or running; the checklist holds LLM work until it is done (or 3 h past)."
        echo "      Fir gives each job its own cores, so this only matters for memory bandwidth; record the overlap."
    fi
}
# Pac-Man work uses an 8 GB Java heap (4 GB cannot build the bare model at length 30; the run store ignores the heap).
# Every job runs from PRISM-Guided-Learning/ (cluster/jobs/command.sbatch). PRISM's cap per job is
# prism.max_concurrent's lock slots in the job's own TMPDIR (8 unless a command sets it), shared by all its processes.

case $what in
item1)
    if [ "${EXCLUSIVE:-0}" = 1 ]; then
        submit --name item1-exclusive --exclusive --cpus 192 --mem 0 --time 10:00:00 -- bash -c '
            for part in families grids pacman; do
                python src/synthesis_time.py --only $part --out out/paper/tables/tab_rq2-synthesis-time.$part.csv \
                    > "$JOB_DIR/item1_$part.log" 2>&1 &
            done; wait'
    else
        submit --name item1-families --cpus 4 --mem 16 --time 10:00:00 -- python src/synthesis_time.py --only families \
            --out out/paper/tables/tab_rq2-synthesis-time.families.csv
        submit --name item1-grids --cpus 4 --mem 16 --time 6:00:00 -- python src/synthesis_time.py --only grids \
            --out out/paper/tables/tab_rq2-synthesis-time.grids.csv
        submit --name item1-pacman --cpus 4 --mem 32 --time 6:00:00 -- python src/synthesis_time.py --only pacman \
            --java-max-mem 8g --out out/paper/tables/tab_rq2-synthesis-time.pacman.csv
    fi ;;
item2)
    submit --name item2-random --cpus 12 --mem 64 --time 6:00:00 -- bash -c '
        for s in random_v0 random_v0_lib2 random_v0_lib3; do
            python src/sequential.py configs/sequential/$s.yaml --mode frozen > "$JOB_DIR/$s.log" 2>&1 &
        done; wait' ;;
item3-smoke)
    m=$(model "${args[0]:-}"); need_key; warn_item1
    submit --name "item3-smoke$m" --cpus 8 --mem 40 --time 3:00:00 -- bash -c "
        python src/run_ablation.py port_fam_history$m --unseeded 1 --name smoke_port_fam_history$m \
            --out-root out/results/smoke/item3 --set 'run.train_sets=[[0, 4, 8]]' &&
        python ../cluster/checks/smoke_report.py out/results/smoke/item3/smoke_port_fam_history$m/seed_none_1" ;;
item3-full)
    m=$(model "${args[0]:-}"); need_key; warn_item1
    submit --name "item3-full$m" --cpus 24 --mem 120 --time 16:00:00 -- bash -c "
        for c in port_ts_history port_fam_history port_pacman_narrow_core port_mer port_mer_n; do
            heap=; [ \$c = port_pacman_narrow_core ] && heap=\"--set prism.java_max_mem=8g\"   # length 30 needs 8 GB
            python src/run_ablation.py \${c}$m --unseeded 3 --set prism.max_concurrent=20 \$heap \
                > \"\$JOB_DIR/\${c}$m.log\" 2>&1 &
        done; wait" ;;
item3-certify)
    m=$(model "${args[0]:-}")
    submit --name "item3-certify$m" --cpus 12 --mem 96 --time 10:00:00 -- bash -c "
        for c in port_ts_history port_fam_history port_pacman_narrow_core port_mer port_mer_n; do
            CONDITIONS=\$c bash ../cluster/jobs/certify_llm.sh $m ${args[*]:1} > \"\$JOB_DIR/certify_\$c.log\" 2>&1 &
        done; wait" ;;
item4)
    mode=${args[0]:?item4 frozen|growing}
    [ "$mode" = growing ] && need_key
    submit --name "item4-$mode" --cpus 10 --mem 48 --time 20:00:00 -- bash -c "
        for s in r01 r02; do
            python src/sequential.py configs/sequential/long_v0.yaml --mode $mode --only \$s > \"\$JOB_DIR/\$s.log\" 2>&1 &
        done; wait" ;;
item5)
    need_key
    submit --name item5-growing --cpus 16 --mem 80 --time 20:00:00 -- bash -c '
        for lib in shift_v0_lib2 shift_v0_lib3; do for s in s1 s2 s3; do
            python src/sequential.py configs/sequential/$lib.yaml --mode growing --only $s \
                > "$JOB_DIR/${lib}_$s.log" 2>&1 &
        done; done; wait' ;;
item6)
    mode=${args[0]:?item6 frozen|growing}
    [ "$mode" = growing ] && need_key
    submit --name "item6-$mode" --cpus 12 --mem 128 --time 20:00:00 -- bash -c "
        for s in r01 r02 r03; do
            python src/sequential.py configs/sequential/pacman_v0.yaml --mode $mode --only \$s > \"\$JOB_DIR/\$s.log\" 2>&1 &
        done; wait" ;;
item7)
    m=$(model "${args[0]:-}"); need_key
    echo "Item 7 starts only when Asher confirms E1 shows feedback and restarts differ AND this model's Full runs on"
    echo "the families are certified. Submitting the runs, then their certification (chained)."
    submit --name "item7$m" --cpus 8 --mem 40 --time 12:00:00 -- bash -c "
        python src/run_ablation.py port_fam_history${m}_resample --unseeded 3 &&
        CONDITIONS=port_fam_history bash ../cluster/jobs/certify_llm.sh ${m}_resample" ;;
item8-baselines)
    submit --name item8-baselines --cpus 8 --mem 32 --time 8:00:00 -- bash -c '
        for b in uuv_hand uuv_tree_raw uuv_tree_features uuv_ppo_features; do
            python src/recertify.py baseline configs/baselines/$b.yaml --workers 4 > "$JOB_DIR/$b.log" 2>&1 || exit 1
        done' ;;
item8-full)
    need_key
    submit --name item8-full --cpus 8 --mem 40 --time 16:00:00 -- bash -c "
        python src/run_ablation.py port_uuv --unseeded 3 &&
        CONDITIONS=port_uuv bash ../cluster/jobs/certify_llm.sh ''" ;;
item8-repick)       # step 3: no LLM calls; the item 8 originals stay in out/results/prepick/port_uuv
    submit --name item8-repick --cpus 8 --mem 40 --time 8:00:00 -- bash -c '
        for k in 1 2 3; do
            out=out/results/core/port_uuv/seed_none_$k
            [ -e $out ] && { echo "$out exists: delete it first (a partial re-pick)"; exit 1; }
            python src/equal_spend.py out/results/prepick/port_uuv/seed_none_$k --budget 1000 --out $out --as-run \
                ${REPAIR_SCORE:+--repair-score $REPAIR_SCORE} > "$JOB_DIR/repick_$k.log" 2>&1 || exit 1
        done
        pids=()
        for k in 1 2 3; do (
            for r in determinized norepair_determinized final; do
                python src/recertify.py run out/results/core/port_uuv/seed_none_$k --rules $r --targets 0-29 \
                    --workers 4 || exit 1
            done ) > "$JOB_DIR/certify_$k.log" 2>&1 &
            pids+=($!)
        done
        for p in "${pids[@]}"; do wait $p || exit 1; done
        python ../cluster/checks/uuv_repick_check.py' ;;
item8-sets|item8-words)
    need_key
    c=port_uuv_sets; [ "$what" = item8-words ] && c=port_uuv_sets_words
    submit --name "$what" --cpus 8 --mem 40 --time 16:00:00 -- bash -c "
        python src/run_ablation.py $c --unseeded 3 \${REPAIR_SCORE:+--set planner.repair_score=\$REPAIR_SCORE} &&
        CONDITIONS=$c WORKERS=4 bash ../cluster/jobs/certify_llm.sh '' determinized final norepair_determinized" ;;
item8-resample)
    need_key
    submit --name item8-resample --cpus 8 --mem 40 --time 16:00:00 -- bash -c "
        python src/run_ablation.py port_uuv_resample --unseeded 3 \${REPAIR_SCORE:+--set planner.repair_score=\$REPAIR_SCORE} &&
        CONDITIONS=port_uuv WORKERS=4 bash ../cluster/jobs/certify_llm.sh _resample" ;;
item8-baselines2)   # step 6; PPO refits save their networks for the sealed pass and replace item 8's PPO checks
    submit --name item8-baselines2 --cpus 8 --mem 32 --time 12:00:00 -- bash -c '
        for b in uuv_random_policies uuv_one_instance uuv_ppo_features uuv_ppo_raw; do
            nets=; case $b in uuv_ppo_*) nets="--nets out/results/baselines/$b/nets.pt" ;; esac
            python src/recertify.py baseline configs/baselines/$b.yaml $nets --workers 4 > "$JOB_DIR/$b.log" 2>&1 || exit 1
        done' ;;
item8-curves)       # step 6: needs item8-repick and item8-resample finished
    submit --name item8-curves --cpus 8 --mem 40 --time 12:00:00 -- bash -c '
        for k in 1 2 3; do
            [ -f out/results/core/port_uuv/seed_none_$k/determinized.json ] || { echo "run item8-repick first"; exit 1; }
            [ -f out/results/ablations/port_uuv_resample/seed_none_$k/SYMBOLIC_results.parquet ] ||
                { echo "run item8-resample first"; exit 1; }
        done
        for k in 1 2 3; do
            python src/oneshot.py out/results/core/port_uuv/seed_none_$k --targets 0-29 --workers 4 \
                > "$JOB_DIR/oneshot_$k.log" 2>&1 || exit 1
        done
        python src/budget_transfer.py out/results/core/port_uuv/seed_none_* \
            out/results/ablations/port_uuv_resample/seed_none_* --targets 0-29 --price 0.10 0.50 --workers 4 \
            > "$JOB_DIR/budget.log" 2>&1' ;;
item8-sealed)       # step 7, once: after every validation result of steps 3-6 is final and committed
    [ "${SEALED:-}" = yes ] || { echo "step 7 checks the sealed test members: run only once every validation";
                                 echo "result is final and committed, as SEALED=yes bash cluster/checklist.sh item8-sealed"; exit 1; }
    submit --name item8-sealed --cpus 8 --mem 40 --time 16:00:00 -- bash -c '
        T=30-41
        for run in out/results/core/port_uuv/seed_none_* \
                   out/results/ablations/port_uuv_{sets,sets_words,resample}/seed_none_*; do
            for r in determinized norepair_determinized final; do
                python src/recertify.py run $run --rules $r --targets $T --prefix transfer_sealed --unseal --workers 4 \
                    || exit 1
            done
        done > "$JOB_DIR/runs.log" 2>&1
        for k in 1 2 3; do
            python src/oneshot.py out/results/core/port_uuv/seed_none_$k --targets $T --prefix transfer_sealed \
                --unseal --workers 4 || exit 1
        done > "$JOB_DIR/oneshot.log" 2>&1
        for b in uuv_hand uuv_tree_raw uuv_tree_features uuv_ppo_features uuv_ppo_raw uuv_random_policies \
                 uuv_one_instance; do
            nets=; case $b in uuv_ppo_*)        # the networks of step 6, never a new fit
                nets="--nets out/results/baselines/$b/nets.pt"
                [ -f out/results/baselines/$b/nets.pt ] || { echo "no $b networks: run item8-baselines2"; exit 1; } ;; esac
            python src/recertify.py baseline configs/baselines/$b.yaml --set targets=$T $nets \
                --prefix transfer_sealed --unseal --workers 4 > "$JOB_DIR/$b.log" 2>&1 || exit 1
        done
        python src/uuv_gate.py --targets $T --unseal --out out/paper/rajan/item8/gate_sealed.csv \
            > "$JOB_DIR/gate.log" 2>&1' ;;
report)
    submit --name report --cpus 4 --mem 16 --time 1:00:00 -- python viz/rajan_results.py ;;   # Asher's tables are his
*) echo "unknown: $what"; exit 1 ;;
esac
