#!/bin/bash
# On Fir: Slurm's accounting for every job in out/cluster/submissions.csv, to out/cluster/sacct.csv
# (pipe-separated: elapsed, CPU time, MaxRSS, requested resources, state, exit code, queue wait).
#   bash cluster/sacct.sh     # then cluster/pull.sh cluster on your machine
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/config.sh"
out="$REPO_DIR/PRISM-Guided-Learning/out/cluster"
# Job ids from submissions.csv, read as CSV (older rows can span lines: a multi-line command in their args)
ids=$(python3 -c "import csv,sys; print(','.join(r['job_id'] for r in csv.DictReader(open(sys.argv[1])) if r['job_id'].isdigit()))" \
      "$out/submissions.csv")
[ -n "$ids" ] || { echo "no submissions recorded"; exit 0; }
sacct -j "$ids" -P --units=G \
    --format=JobID,JobName,State,ExitCode,Submit,Start,End,Elapsed,Timelimit,AllocCPUS,ReqMem,MaxRSS,TotalCPU,NodeList \
    > "$out/sacct.csv"
echo "wrote $out/sacct.csv ($(($(wc -l < "$out/sacct.csv") - 1)) rows)"
