#!/bin/bash
# Queue Rajan's whole checklist on Fir at once, ordered by Slurm dependencies (run on the login node, repo root):
#
#   bash cluster/queue_all.sh --dry     # print every sbatch line (dependencies shown as placeholders)
#   bash cluster/queue_all.sh           # submit
#
#   item 1 (3 jobs: families, grids, pacman)            starts now, alone among our jobs
#   ├─ item 2 (frozen, PRISM only)                      after item 1 ends
#   ├─ item 4 frozen ─> item 4 growing (Haiku)          after item 1 ends
#   ├─ item 5 growing (Haiku)                           after item 1 ends
#   ├─ item 6 frozen ─> item 6 growing (Haiku)          after item 1 ends
#   └─ item 3 smoke (luna, gemini)                      after item 1 ends
#        └─ item 3 full ─> certify ─> certify NoRepair  per model, only if its smoke test passed (afterok)
#   report (viz/rajan_results.py)                       after everything above ends
# Item 7 is not queued: it waits on Asher's E1 verdict (bash cluster/checklist.sh item7 <model> when it holds).
#
# A model whose smoke test fails leaves its full / certify jobs pending with "DependencyNeverSatisfied": cancel
# them (scancel), fix the config if it is a config problem, and resubmit its smoke test once (rule: dropped after
# two failures). The OpenRouter key must have credits: free-tier keys are refused here.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/config.sh"
dry=""
[ "${1:-}" = --dry ] && dry=--dry

if [ -z "$dry" ]; then
    [ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY is not set in this shell"; exit 1; }
    tier=$(curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" |
           python3 -c "import json,sys; d=json.load(sys.stdin).get('data',{}); print(d.get('is_free_tier'), d.get('limit_remaining'))")
    echo "OpenRouter key: is_free_tier, limit_remaining = $tier"
    if [[ $tier == True* ]] && [ "${FORCE:-0}" != 1 ]; then
        echo "this key has never bought credits (free tier): the LLM items would fail. Use the paid key (FORCE=1 overrides)."
        exit 1
    fi
fi

q() {   # q <dependency or ""> <checklist args...>: submit, echo the new job ids joined by ':'
    local after=$1; shift
    local out
    out=$(AFTER="$after" bash "$here/checklist.sh" "$@" $dry 2>&1) || { echo "$out" >&2; exit 1; }
    echo "$out" | sed 's/^/  /' >&2
    if [ -n "$dry" ]; then local IFS=-; echo "<$*>"; else
        echo "$out" | grep -o 'as job [0-9]*' | awk '{print $3}' | paste -sd: -; fi
}

echo "== item 1" >&2;            I1=$(q "" item1)
echo "== item 2" >&2;            I2=$(q "afterany:$I1" item2)
echo "== item 4" >&2;            I4F=$(q "afterany:$I1" item4 frozen);  I4G=$(q "afterany:$I4F" item4 growing)
echo "== item 5" >&2;            I5=$(q "afterany:$I1" item5)
echo "== item 6" >&2;            I6F=$(q "afterany:$I1" item6 frozen);  I6G=$(q "afterany:$I6F" item6 growing)
all="$I1:$I2:$I4F:$I4G:$I5:$I6F:$I6G"
for m in luna gemini; do
    echo "== item 3, $m" >&2
    S=$(q "afterany:$I1" item3-smoke "$m")
    F=$(q "afterok:$S" item3-full "$m")
    C=$(q "afterok:$F" item3-certify "$m")
    N=$(q "afterany:$C" item3-certify "$m" norepair_determinized)
    all="$all:$S:$F:$C:$N"
done
echo "== report" >&2;            R=$(q "afterany:$all" report)
echo
echo "queued. squeue -u \$USER shows them; item 7 waits for Asher (bash cluster/checklist.sh item7 <model>)."
