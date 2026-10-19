#!/bin/bash
# On Fir: Slurm's accounting for every job in out/cluster/submissions.csv, to out/cluster/sacct.csv
# (pipe-separated: elapsed, CPU time, MaxRSS, requested resources, state, exit code, queue wait).
#   bash cluster/sacct.sh     # then cluster/pull.sh cluster on your machine
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/config.sh"
out="$REPO_DIR/PRISM-Guided-Learning/out/cluster"
ids=$(tail -n +2 "$out/submissions.csv" | cut -d, -f2 | paste -sd, -)
[ -n "$ids" ] || { echo "no submissions recorded"; exit 0; }
sacct -j "$ids" -P --units=G \
    --format=JobID,JobName,State,ExitCode,Submit,Start,End,Elapsed,Timelimit,AllocCPUS,ReqMem,MaxRSS,TotalCPU,NodeList \
    > "$out/sacct.csv"
echo "wrote $out/sacct.csv ($(($(wc -l < "$out/sacct.csv") - 1)) rows)"
