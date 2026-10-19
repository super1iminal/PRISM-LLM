#!/bin/bash
# Run on your own machine: bring every result and record back from Fir into this checkout's out/.
#   bash cluster/pull.sh               # all of out/: runs (parquet, outputs/*.json, llm_tasks.jsonl, config.json,
#                                      # transfer CSVs, per-run logs), reports, figures, out/cluster/ job records
#   bash cluster/pull.sh cluster       # only out/cluster/ (job records, Slurm logs, smoke tests): small and quick
# Nothing local is deleted; files that differ are replaced by Fir's. Add --dry to list what would change.
# Run cluster/sacct.sh on Fir first to include Slurm's accounting (out/cluster/sacct.csv).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/config.sh"
local_out=$(cd "$here/.." && pwd)/PRISM-Guided-Learning/out
remote="$FIR_HOST:$REPO_DIR/PRISM-Guided-Learning/out"
sub="" dry=()
for a in "$@"; do
    case $a in cluster) sub=cluster/ ;; --dry) dry=(--dry-run -i) ;; *) echo "unknown: $a"; exit 1 ;; esac
done
mkdir -p "$local_out/$sub"
rsync -az --info=stats1 "${dry[@]}" "$remote/$sub" "$local_out/$sub"
echo "pulled $remote/$sub -> $local_out/$sub"
