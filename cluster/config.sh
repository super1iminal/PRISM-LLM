# Site settings for the Fir cluster (Alliance Canada). Sourced by every script in cluster/.
# Edit the four values marked EDIT, or export them before calling a script; nothing here is secret
# (the OpenRouter key stays in your shell's environment as OPENROUTER_API_KEY, never in a file).

# --- EDIT: who and where -------------------------------------------------------------------------------
FIR_USER=${FIR_USER:-maghera}                                 # your Alliance username
FIR_HOST=${FIR_HOST:-fir}                                     # ssh host alias from ~/.ssh/config (user, MFA, keys)
FIR_ACCOUNT=${FIR_ACCOUNT:-def-mmehride}                      # Slurm account, e.g. def-<pi>
FIR_ROOT=${FIR_ROOT:-/project/$FIR_ACCOUNT/$FIR_USER}         # where the repo, venv and PRISM live on Fir

# Slurm mails every job's state changes (start, end, failure, requeue, 80% of its time limit) here: an ntfy.sh
# topic's email address, so they arrive as ntfy notifications. Empty: no mail.
NOTIFY_EMAIL=${NOTIFY_EMAIL-ntfy-OfDNOyH4RvAsiaHr@ntfy.sh}
NOTIFY_TYPES=${NOTIFY_TYPES:-ALL,TIME_LIMIT_80}

# --- Layout on Fir (derived) ----------------------------------------------------------------------------
REPO_DIR=${REPO_DIR:-$FIR_ROOT/PRISM-LLM}                     # the rsync'd repo (cluster/push.sh)
VENV_DIR=${VENV_DIR:-$FIR_ROOT/venv-prism-llm}                # Python venv (cluster/setup.sh)
PRISM_VERSION=${PRISM_VERSION:-4.10.1}
# Built from source on Fir: the prebuilt Linux binary's liblpsolve55.so needs GLIBC 2.38, newer than Fir's
# software stack, so its native engines (hybrid, sparse, multi-objective LP) fail to load.
PRISM_HOME=${PRISM_HOME:-$FIR_ROOT/prism-$PRISM_VERSION-src/prism}
PRISM_URL=${PRISM_URL:-https://github.com/prismmodelchecker/prism/releases/download/v$PRISM_VERSION/prism-$PRISM_VERSION-src.tar.gz}

# --- Modules (StdEnv/2023); cluster/setup.sh tries these in order and records the ones that loaded -------
MODULES_BASE=${MODULES_BASE:-"StdEnv/2023"}
JAVA_MODULES=${JAVA_MODULES:-"java/21.0.1 java/17.0.6 java"}
PYTHON_MODULES=${PYTHON_MODULES:-"python/3.12 python/3.13 python/3.11"}
EXTRA_MODULES=${EXTRA_MODULES:-"arrow"}                       # Alliance ships pyarrow through the arrow module

# --- Per-run sizing (docs: cluster/README.md) ------------------------------------------------------------
# One LLM run keeps up to prism.max_concurrent (8) PRISM processes busy, each with a 4 GB heap.
CORES_PER_RUN=${CORES_PER_RUN:-8}
MEM_GB_PER_RUN=${MEM_GB_PER_RUN:-40}
MEM_GB_BASE=${MEM_GB_BASE:-8}                                 # the launcher, reports, page cache

# --- Smoke-test model (free on OpenRouter; no structured outputs, so llm.openrouter.structured=false) ---
SMOKE_MODEL=${SMOKE_MODEL:-nvidia/nemotron-3-ultra-550b-a55b:free}
