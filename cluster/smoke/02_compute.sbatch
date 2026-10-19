#!/bin/bash
# Smoke test 2, on a compute node: everything a real job does, at toy size, with the free model.
#   1. environment and internet from the compute node
#   2. the test suite without the saved-run replays (needs PRISM, no LLM)
#   3. one free LLM request
#   4. a whole run through the same entry point as the experiments (run_ablation.py), 1 set x 1 chain x 2 rounds
#      (at most 4 free requests), then its certification on the 20 held-out rooms members (transfer.py),
#      as written and with ties open, as run_experiment.py --check does
# Submit from the repo root on Fir:   bash cluster/submit.sh smoke
# The model is $SMOKE_MODEL (cluster/config.sh); SMOKE_SET="section.key=value ..." adds run-config overrides.
# Output: out/cluster/jobs/<job>_smoke/ (record, resources, summary.json) and out/results/smoke/fir/ (the run).
#SBATCH --job-name=smoke
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:30:00
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source cluster/env.sh || exit 1
source cluster/lib/job.sh
job_start "smoke"
summary="$JOB_DIR/summary.json"
declare -A result
step() { echo; echo "== $(date +%T) $1"; }
run_step() {             # run_step <name> <command...>: records pass/fail and seconds
    local name=$1; shift
    local t0=$(date +%s)
    step "$name"
    if "$@"; then result[$name]="pass $(( $(date +%s) - t0 ))"; else result[$name]="FAIL $(( $(date +%s) - t0 ))"; fi
    echo "-- $name: ${result[$name]}"
}

run_step env bash -c 'hostname; nproc; free -g | head -2; echo "TMPDIR=$TMPDIR"; prism -version | head -1;
    curl -s -o /dev/null -w "openrouter.ai HTTP %{http_code} in %{time_total}s\n" https://openrouter.ai/api/v1/models'

run_step pytest bash -c "cd '$GL_DIR' && python -m pytest -q -p no:cacheprovider tests -k 'not replay' \
    --junitxml='$JOB_DIR/pytest.xml' --durations=15"

sets=(--set "llm.model=$SMOKE_MODEL")
for kv in ${SMOKE_SET:-}; do sets+=(--set "$kv"); done
run_step llm_ping python cluster/smoke/llm_ping.py --model "$SMOKE_MODEL" "${sets[@]}" --out "$JOB_DIR/llm_ping.json"

RUN_NAME="smoke_free_${SLURM_JOB_ID:-local}"
RUN_DIR="out/results/smoke/fir/$RUN_NAME/seed_none_1"
run_step run bash -c "cd '$GL_DIR' && python src/run_ablation.py smoke_free --unseeded 1 \
    --name '$RUN_NAME' --out-root out/results/smoke/fir ${sets[*]@Q}"
run_step transfer_final bash -c "cd '$GL_DIR' && python src/transfer.py '$RUN_DIR' --targets 16-35 --rules final --workers 2"
run_step transfer_open bash -c "cd '$GL_DIR' && python src/transfer.py '$RUN_DIR' --targets 16-35 --rules open --workers 2"

step "summary"
python - "$summary" "$GL_DIR/$RUN_DIR" "$(declare -p result)" <<'EOF'
import glob, json, os, re, sys
path, run_dir, decl = sys.argv[1:4]
steps = {k: {"ok": v.split()[0] == "pass", "seconds": int(v.split()[1])}
         for k, v in re.findall(r'\[(\w+)\]="([^"]+)"', decl)}
calls = tokens = 0
for f in glob.glob(os.path.join(run_dir, "outputs", "sample_*.json")):
    for it in json.load(open(f)).get("iterations", []):
        calls += it.get("llm_calls") or 0
        tokens += it.get("llm_output_tokens") or 0
out = {"steps": steps, "passed": all(s["ok"] for s in steps.values()),
       "run_dir": run_dir, "llm_calls": calls, "llm_output_tokens": tokens}
json.dump(out, open(path, "w"), indent=2)
print(json.dumps(out, indent=2))
EOF
ok=$(python -c "import json;print(int(json.load(open('$summary'))['passed']))")
job_end $(( 1 - ok ))
[ "$ok" = 1 ] && echo "SMOKE 2 PASSED" || echo "SMOKE 2 FAILED (see $summary)"
exit $(( 1 - ok ))
