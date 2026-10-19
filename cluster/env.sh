# The run environment on Fir: modules, venv, PRISM, temp dirs. Source it (login node or inside a job):
#   source cluster/env.sh
# It never exits the calling shell; it returns 1 and says what is missing.

_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$_here/config.sh"

if [ -f "$FIR_ROOT/modules.txt" ]; then           # the modules cluster/setup.sh found, one per line
    module --force purge >/dev/null 2>&1
    while read -r m; do [ -n "$m" ] && module load "$m" >/dev/null 2>&1; done < "$FIR_ROOT/modules.txt"
else
    echo "env.sh: $FIR_ROOT/modules.txt missing; run cluster/setup.sh first" >&2
fi

if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "env.sh: no venv at $VENV_DIR; run cluster/setup.sh first" >&2
    return 1 2>/dev/null
fi

export PATH="$PRISM_HOME/bin:$PATH"
export PRISM_PATH="$PRISM_HOME/bin/prism"
# Fir's java module sets JAVA_TOOL_OPTIONS=-Xmx2g; PRISM sets its own heap (prism.java_max_mem), and the JVM's
# "Picked up JAVA_TOOL_OPTIONS" notice would land in every PRISM output the repo parses.
unset JAVA_TOOL_OPTIONS
# One GC thread per JVM, not one per node core (192 on Fir): 8 PRISM processes per run would start ~1500.
export PRISM_JAVA_PARAMS="${PRISM_JAVA_PARAMS:--XX:+UseSerialGC}"

# PRISM's machine-wide process slots (prism.max_concurrent) are lock files in the system temp dir. In a job,
# point it at the job's own node-local disk, so jobs sharing a node do not share (and throttle) the slots.
if [ -n "${SLURM_TMPDIR:-}" ]; then
    export TMPDIR="$SLURM_TMPDIR"
fi

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}

GL_DIR="$REPO_DIR/PRISM-Guided-Learning"          # every Python entry point runs from here
CLUSTER_OUT="$GL_DIR/out/cluster"                 # job records, Slurm logs, resource samples (pulled back)
export GL_DIR CLUSTER_OUT
