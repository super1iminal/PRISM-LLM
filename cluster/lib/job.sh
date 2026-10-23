# Helpers for Slurm job scripts: a record per job and a resource sampler. Source after cluster/env.sh.
#
# Every job writes to $CLUSTER_OUT/jobs/<job id>_<job name>/:
#   meta.json        what ran: job id, node, cores, memory, time limit, git commit, modules, command, start/end, exit code
#   resources.csv    every $SAMPLE_S seconds: job memory (cgroup), RSS and CPU% of PRISM (java) and Python processes
#   stdout.log       the job's command output (also in the Slurm log)
# cluster/sacct.sh adds Slurm's own accounting (elapsed, MaxRSS, CPU time) for every recorded job.

SAMPLE_S=${SAMPLE_S:-30}

job_start() {             # job_start <command words...>: sets JOB_DIR, writes meta.json, starts the sampler
    export JOB_DIR="$CLUSTER_OUT/jobs/${SLURM_JOB_ID:-local$$}_${SLURM_JOB_NAME:-interactive}"
    mkdir -p "$JOB_DIR"
    JOB_T0=$(date +%s)
    JOB_CMD="$*"
    _job_meta running ""
    _job_sampler &
    JOB_SAMPLER=$!
    echo "job record: $JOB_DIR"
}

job_end() {               # job_end <exit code>
    local code=$1
    [ -n "${JOB_SAMPLER:-}" ] && kill "$JOB_SAMPLER" 2>/dev/null
    _job_meta finished "$code"
}

_job_meta() {
    local state=$1 code=$2
    python - "$JOB_DIR/meta.json" "$state" "$code" "$JOB_T0" "$JOB_CMD" <<'EOF'
import json, os, subprocess, sys, time
path, state, code, t0, cmd = sys.argv[1:6]
def sh(*a, stderr=False, timeout=120):
    """A command's output (stderr instead with `stderr`), or None if it fails or times out."""
    try:
        p = subprocess.run(a, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    return (p.stderr if stderr else p.stdout).strip() if p.returncode == 0 else None
env = {k: v for k, v in os.environ.items() if k.startswith("SLURM_") and k in (
    "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
    "SLURM_JOB_ACCOUNT", "SLURM_SUBMIT_DIR", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")}
root = os.environ.get("REPO_DIR", ".")
def git_diff():
    """`git diff --stat HEAD` outside PRISM-Guided-Learning/out (pathspecs from the top of the repo, whatever the
    working directory), or None. cluster/push.sh copies .git, so the index's file stats are stale on Fir and git
    re-reads every file once: refresh the index first, with time for Lustre (item 8's jobs recorded null here)."""
    sh("git", "-C", root, "update-index", "-q", "--refresh", timeout=900)
    return sh("git", "-C", root, "diff", "--stat", "HEAD", "--", ":(top)", ":(top,exclude)PRISM-Guided-Learning/out",
              timeout=900)
def hardware():
    out = {"os": " ".join(open("/etc/os-release").read().split("PRETTY_NAME=")[1].splitlines()[0].strip('"').split())
           if os.path.exists("/etc/os-release") else None, "kernel": os.uname().release,
           "cores_allocated": os.environ.get("SLURM_CPUS_PER_TASK"), "cores_visible": len(os.sched_getaffinity(0))}
    try:
        cpu = [l for l in open("/proc/cpuinfo") if l.startswith("model name")]
        out["cpu"] = cpu[0].split(":", 1)[1].strip() if cpu else None
        out["node_cores"] = len(cpu)
        mem = [l for l in open("/proc/meminfo") if l.startswith("MemTotal")]
        out["node_ram_gb"] = round(int(mem[0].split()[1]) / 2**20, 1)
    except OSError:
        pass
    out["job_mem_mb"] = os.environ.get("SLURM_MEM_PER_NODE")
    return out
meta = {"state": state, "exit_code": int(code) if code else None, "command": cmd,
        "start": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(int(t0))),
        "end": time.strftime("%Y-%m-%dT%H:%M:%S%z") if state == "finished" else None,
        "elapsed_s": int(time.time()) - int(t0) if state == "finished" else None,
        "host": os.uname().nodename, "slurm": env,
        "time_limit": sh("squeue", "-h", "-j", os.environ.get("SLURM_JOB_ID", "0"), "-o", "%l"),
        "git_commit": sh("git", "-C", root, "rev-parse", "HEAD"),
        "git_dirty": None if (diff := git_diff()) is None else bool(diff),   # tracked files outside out/ changed?
        "git_diff": diff,
        "modules": os.environ.get("LOADEDMODULES", "").split(":"),
        "python": sys.version.split()[0], "java": next(iter((sh("java", "-version", stderr=True) or "").splitlines()), None),
        "prism": os.environ.get("PRISM_PATH"), "tmpdir": os.environ.get("TMPDIR"),
        "hardware": hardware(),
        "own_jobs_running": sh("squeue", "-h", "-u", os.environ.get("USER", ""), "-t", "RUNNING",
                               "-o", "%i %j %C %N"),        # rule 5: what else of ours was running
        "node_jobs_running": sh("squeue", "-h", "-w", os.uname().nodename, "-t", "RUNNING", "-o", "%C") and
            len(sh("squeue", "-h", "-w", os.uname().nodename, "-t", "RUNNING", "-o", "%C").splitlines()),
        "openrouter_key_set": bool(os.environ.get("OPENROUTER_API_KEY"))}
json.dump(meta, open(path, "w"), indent=2)
EOF
}

_job_sampler() {          # resources.csv: time, cgroup memory, java and python RSS (MB) and CPU%, process counts
    local cg mem
    cg=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null)
    echo "t_s,cgroup_mem_mb,java_n,java_rss_mb,java_cpu,python_n,python_rss_mb,python_cpu,load1" > "$JOB_DIR/resources.csv"
    while true; do
        mem=""
        for f in "/sys/fs/cgroup$cg/memory.current" "/sys/fs/cgroup${cg%/*}/memory.current"; do
            [ -r "$f" ] && { mem=$(( $(cat "$f") / 1048576 )); break; }
        done
        ps -u "$USER" -o comm=,rss=,pcpu= | awk -v t=$(( $(date +%s) - JOB_T0 )) -v mem="$mem" \
            -v ld="$(cut -d' ' -f1 /proc/loadavg)" '
            $1=="java"{jn++; jr+=$2; jc+=$3} $1~/^python/{pn++; pr+=$2; pc+=$3}
            END{printf "%d,%s,%d,%d,%.0f,%d,%d,%.0f,%s\n", t, mem, jn, jr/1024, jc, pn, pr/1024, pc, ld}' \
            >> "$JOB_DIR/resources.csv"
        sleep "$SAMPLE_S"
    done
}
