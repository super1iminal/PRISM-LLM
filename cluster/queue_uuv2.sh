#!/bin/bash
# Queue Asher's UUV plan, steps 3-6, on Fir for both repair objectives at once (run on the login node, repo root, with
# the paid OpenRouter key exported):
#
#   bash cluster/queue_uuv2.sh --dry     # print every sbatch line (dependencies shown as placeholders)
#   bash cluster/queue_uuv2.sh           # submit
#   VARIANTS=mf bash cluster/queue_uuv2.sh   # one variant only (capped = capped_sum, mf = members_first)
#
#   per variant (capped_sum: the conditions as written; mf: planner.repair_score members_first, the *_mf conditions):
#     item8-repick       step 3, PRISM only            starts now
#     item8-sets         step 4, Haiku (~$1.30)        starts now
#     item8-words        step 5, Haiku (~$1.30)        starts now
#     item8-resample     step 6 Restart, Haiku (~$0.40) starts now
#     item8-curves       step 6 OneShot (capped_sum only) and budget curves, after repick and resample (afterok)
#   once: item8-baselines2 (step 6 baselines, no LLM; repair does not apply)  starts now
#   report (viz/rajan_results.py)                     after everything above ends
# Haiku in all: about $6 for both variants. Step 7 (SEALED=yes bash cluster/checklist.sh item8-sealed) is not queued:
# it runs once, after the validation results are pulled and committed; it checks every variant that ran.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/config.sh"
dry=""
[ "${1:-}" = --dry ] && dry=--dry
variants=${VARIANTS:-capped mf}      # default: both

if [ -z "$dry" ]; then
    [ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY is not set in this shell"; exit 1; }
    tier=$(curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" |
           python3 -c "import json,sys; d=json.load(sys.stdin).get('data',{}); print(d.get('is_free_tier'), d.get('limit_remaining'))")
    echo "OpenRouter key: is_free_tier, limit_remaining = $tier"
    if [[ $tier == True* ]] && [ "${FORCE:-0}" != 1 ]; then
        echo "this key has never bought credits (free tier): the Haiku jobs would fail. Use the paid key (FORCE=1 overrides)."
        exit 1
    fi
    [ -d "$REPO_DIR/PRISM-Guided-Learning/out/results/prepick/port_uuv" ] ||
        { echo "out/results/prepick/port_uuv is missing on Fir: move item 8's runs there first"; exit 1; }
fi

q() {   # q <dependency or ""> <checklist args...>: submit, echo the new job ids joined by ':'
    local after=$1; shift
    local out
    out=$(AFTER="$after" bash "$here/checklist.sh" "$@" $dry 2>&1) || { echo "$out" >&2; exit 1; }
    echo "$out" | sed 's/^/  /' >&2
    if [ -n "$dry" ]; then local IFS=-; echo "<$*>"; else
        echo "$out" | grep -o 'as job [0-9]*' | awk '{print $3}' | paste -sd: -; fi
}

all=""
echo "== baselines (once)" >&2;  B=$(q "" item8-baselines2); all=$B
for name in $variants; do
    case $name in capped) v= ;; mf) v=mf ;; *) echo "VARIANTS: capped and/or mf" >&2; exit 1 ;; esac
    echo "== $name" >&2
    P=$(q "" item8-repick $v)
    S=$(q "" item8-sets $v)
    W=$(q "" item8-words $v)
    R=$(q "" item8-resample $v)
    C=$(q "afterok:$P:$R" item8-curves $v)
    all="$all:$P:$S:$W:$R:$C"
done
echo "== report" >&2;            q "afterany:$all" report >/dev/null
echo
echo "queued. squeue -u \$USER shows them. Step 7 waits: SEALED=yes bash cluster/checklist.sh item8-sealed"
