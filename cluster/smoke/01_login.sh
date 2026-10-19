#!/bin/bash
# Smoke test 1, on the login node (a minute or two; well inside the login nodes' ~10 CPU-minute limit):
# environment, PRISM on a tiny model, the repo's PRISM tests, the OpenRouter key and one free LLM request.
#   bash cluster/smoke/01_login.sh
# Output: out/cluster/smoke/login_<time>/ (pulled back by cluster/pull.sh).
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$here/env.sh" || exit 1
out="$CLUSTER_OUT/smoke/login_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
exec > >(tee "$out/stdout.log") 2>&1
status=0
step() { echo; echo "== $1"; }
fail() { echo "FAIL: $1"; status=1; }

step "versions"
echo "host $(hostname)  modules: ${LOADEDMODULES:-none}"
python --version; java -version 2>&1 | head -1; prism -version | head -1 || fail "prism not runnable"

step "PRISM on a two-state MDP (expect Pmax=1 and Pmin=0.5)"
tmp=$(mktemp -d)
cat > "$tmp/m.prism" <<'EOF'
mdp
module m
  s : [0..2] init 0;
  [a] s=0 -> 0.5:(s'=1) + 0.5:(s'=2);
  [b] s=0 -> 1:(s'=1);
  [] s>0 -> true;
endmodule
EOF
echo 'Pmax=? [ F s=1 ];
Pmin=? [ F s=1 ];' > "$tmp/m.props"
prism "$tmp/m.prism" "$tmp/m.props" -javamaxmem 1g > "$out/prism_tiny.txt" 2>&1
grep -E "^Result" "$out/prism_tiny.txt" || fail "PRISM produced no result (see prism_tiny.txt)"
grep -q "^Result: 1.0" "$out/prism_tiny.txt" && grep -q "^Result: 0.5" "$out/prism_tiny.txt" || fail "unexpected PRISM values"
rm -rf "$tmp"

step "the repo's PRISM runner tests"
(cd "$GL_DIR" && python -m pytest -q -p no:cacheprovider tests/test_prism.py tests/test_config.py tests/test_openrouter.py \
    --junitxml="$out/pytest_login.xml") || fail "pytest"

step "OpenRouter key and one free request ($SMOKE_MODEL)"
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    fail "OPENROUTER_API_KEY is not set in this shell"
else
    python "$here/smoke/llm_ping.py" --out "$out/llm_ping.json" || fail "LLM ping (see llm_ping.json)"
fi

echo; [ $status -eq 0 ] && echo "SMOKE 1 PASSED -> $out" || echo "SMOKE 1 FAILED -> $out"
exit $status
