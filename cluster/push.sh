#!/bin/bash
# Run on your own machine: copy the repo to Fir.
#   bash cluster/push.sh                  # code, configs, docs, .git (never out/)
#   bash cluster/push.sh --with-results   # also out/: the existing runs, checks and sequences the items read.
#                                         # A file is sent only if the local copy is newer (rsync --update), so
#                                         # Asher's re-certified files replace stale ones and results made on Fir
#                                         # are never overwritten.
# Add --dry to see what would be sent. With Fir's MFA, an ssh ControlMaster saves re-authenticating (README).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/config.sh"
local_root=$(cd "$here/.." && pwd)
remote="$FIR_HOST:$REPO_DIR"
with_results=0 dry=()
for a in "$@"; do
    case $a in --with-results) with_results=1 ;; --dry) dry=(--dry-run) ;; *) echo "unknown: $a"; exit 1 ;; esac
done

ssh "$FIR_HOST" "mkdir -p '$REPO_DIR'"
rsync -az --info=stats1 "${dry[@]}" \
    --exclude 'PRISM-Guided-Learning/out/' --exclude '.venv/' --exclude '__pycache__/' --exclude '.claude/' \
    --exclude '.pytest_cache/' --exclude 'third_party/*/' --exclude '.env' \
    "$local_root/" "$remote/"
if [ $with_results = 1 ]; then
    rsync -az --info=stats1 --update "${dry[@]}" --exclude 'smoke/' --exclude 'cluster/' \
        "$local_root/PRISM-Guided-Learning/out/" "$remote/PRISM-Guided-Learning/out/"
fi
echo "pushed to $remote"
