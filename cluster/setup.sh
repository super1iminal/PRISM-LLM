#!/bin/bash
# One-time setup on a Fir login node: modules, Python venv, PRISM. Safe to re-run (skips what exists).
#   bash cluster/setup.sh            # after cluster/push.sh has copied the repo to $REPO_DIR
# Writes $FIR_ROOT/modules.txt (the modules that loaded) and $FIR_ROOT/setup.log.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$here/config.sh"
mkdir -p "$FIR_ROOT"
exec > >(tee -a "$FIR_ROOT/setup.log") 2>&1
echo "== setup $(date -Is) on $(hostname)"

load_first() {                # load_first <var> <modules...>: loads the first module that loads, in THIS shell
    local var=$1 m; shift     # (not in $(...): a subshell's module load is lost when it exits)
    for m in "$@"; do
        if module load "$m" >/dev/null 2>&1; then printf -v "$var" '%s' "$m"; return 0; fi
    done
    return 1
}

module --force purge >/dev/null 2>&1
modules=()
for m in $MODULES_BASE; do module load "$m" && modules+=("$m"); done
load_first java_mod $JAVA_MODULES || { echo "no java module among: $JAVA_MODULES (module spider java)"; exit 1; }
load_first py_mod $PYTHON_MODULES || { echo "no python module among: $PYTHON_MODULES (module spider python)"; exit 1; }
modules+=("$java_mod" "$py_mod")
for m in $EXTRA_MODULES; do
    if module load "$m" >/dev/null 2>&1; then modules+=("$m"); else echo "note: module $m did not load (skipped)"; fi
done
echo "modules: ${modules[*]}"
java -version 2>&1 | head -1
python --version
# Stop here rather than build a venv on the system Python or run PRISM on the system Java 8.
java_major=$(java -version 2>&1 | awk -F'"' '/version/ {split($2, v, "."); print (v[1] == 1 ? v[2] : v[1])}')
[ "${java_major:-0}" -ge 17 ] || { echo "java is $java_major after loading $java_mod; need >= 17"; exit 1; }
[ "$(command -v python)" != /usr/bin/python ] && [ "$(command -v python)" != /usr/bin/python3 ] \
    || { echo "python is the system one ($(command -v python)) after loading $py_mod"; exit 1; }
printf '%s\n' "${modules[@]}" > "$FIR_ROOT/modules.txt"

# --- Python venv ------------------------------------------------------------------------------------------
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    virtualenv --no-download "$VENV_DIR" || python -m venv "$VENV_DIR" || { echo "venv creation failed"; exit 1; }
fi
source "$VENV_DIR/bin/activate" || exit 1
pip install --no-index --upgrade pip >/dev/null 2>&1 || true
# Prefer the Alliance wheelhouse (--no-index); fall back to PyPI (Fir has internet) for anything it lacks.
grep -vE '^\s*(#|$)' "$REPO_DIR/requirements.txt" | while read -r pkg; do
    if pip install --no-index "$pkg" >/dev/null 2>&1; then
        echo "pip: $pkg (wheelhouse)"
    elif pip install "$pkg" >/dev/null 2>&1; then
        echo "pip: $pkg (PyPI)"
    else
        echo "pip: FAILED $pkg"
    fi
done
python - <<'EOF'
import importlib
for m in ["pandas", "pyarrow", "pydantic", "openai", "yaml", "numpy", "scipy", "sklearn", "matplotlib", "pytest", "torch"]:
    try:
        mod = importlib.import_module(m)
        print(f"  import {m:12s} ok {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  import {m:12s} FAILED: {e}")
EOF

# --- PRISM (built from source; ~1-2 min) -------------------------------------------------------------------
if [ ! -f "$PRISM_HOME/lib/libprism.so" ]; then
    tmp=$(mktemp -d)
    curl -fsSL -o "$tmp/prism-src.tgz" "$PRISM_URL" || { echo "PRISM download failed: $PRISM_URL"; exit 1; }
    tar -xzf "$tmp/prism-src.tgz" -C "$FIR_ROOT"
    rm -rf "$tmp"
    echo "building PRISM in $PRISM_HOME (log: $FIR_ROOT/prism-build.log)"
    (cd "$PRISM_HOME" && make -j4 > "$FIR_ROOT/prism-build.log" 2>&1) \
        || { echo "PRISM build failed; tail of $FIR_ROOT/prism-build.log:"; tail -20 "$FIR_ROOT/prism-build.log"; exit 1; }
fi
unset JAVA_TOOL_OPTIONS
"$PRISM_HOME/bin/prism" -version | head -1
# The native engines must load: a tiny MDP on the default (hybrid) engine and on the sparse engine.
tmp=$(mktemp -d)
printf 'mdp\nmodule m\n s : [0..2] init 0;\n [a] s=0 -> 0.5:(s\x27=1) + 0.5:(s\x27=2);\n [b] s=0 -> 1:(s\x27=1);\n [] s>0 -> true;\nendmodule\n' > "$tmp/m.prism"
echo 'Pmin=? [ F s=1 ]' > "$tmp/m.props"
for engine in hybrid sparse; do
    if "$PRISM_HOME/bin/prism" "$tmp/m.prism" "$tmp/m.props" -$engine -javamaxmem 1g 2>&1 | grep -q "^Result: 0.5"; then
        echo "PRISM $engine engine: ok"
    else
        echo "PRISM $engine engine: FAILED"; "$PRISM_HOME/bin/prism" "$tmp/m.prism" "$tmp/m.props" -$engine -javamaxmem 1g | tail -15
    fi
done
rm -rf "$tmp"

echo "== setup done. Next: bash cluster/smoke/01_login.sh"
