#!/bin/bash
# Haiku 5.5 smoke tests of Asher's UUV plan's three new conditions (steps 4-6), before their Fir jobs: each cut to one
# training set, 1 chain x 2 rounds (about 2-3 calls, ~$0.005 each), then cluster/checks/smoke_report.py.
#   port_uuv_sets        on the new training set 1 (ids 42-44)
#   port_uuv_sets_words  on the same set, failure feedback in words
#   port_uuv_resample    on item 8's set (0, 6, 10), every round a fresh start
# Uses the paid key in OPENROUTER_API_KEY (set -a && . .env && set +a). Run from the repo root, locally or on Fir's
# login node. Outputs: PRISM-Guided-Learning/out/results/smoke/uuv2/<name>/seed_none_1 (not counted by the run store).
# REPAIR_SCORE=members_first (environment): the pick's repair objective, as the Fir jobs take it.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$here/../../PRISM-Guided-Learning"
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY is not set in this shell"; exit 1; }
cut=(--set planner.chains=1 --set planner.max_rounds=2 --set planner.max_fixups=1
     ${REPAIR_SCORE:+--set planner.repair_score=$REPAIR_SCORE})
root=out/results/smoke/uuv2
status=0
for spec in "port_uuv_sets:[[42, 43, 44]]" "port_uuv_sets_words:[[42, 43, 44]]" "port_uuv_resample:"; do
    c=${spec%%:*} sets=${spec#*:}
    model=$(cd src && python -c "from config import load_config; print(load_config('$c').llm.model)")
    [ "$model" = anthropic/claude-haiku-5.5 ] || { echo "$c: model $model, not Haiku 5.5; stopping"; exit 1; }
    echo "== $c"
    python src/run_ablation.py "$c" --unseeded 1 --name "smoke_$c" --out-root "$root" "${cut[@]}" \
        ${sets:+--set "run.train_sets=$sets"} || { status=1; continue; }
    python ../cluster/checks/smoke_report.py "$root/smoke_$c/seed_none_1" || status=1
done
exit $status
