"""One numpy-heavy step at a time per process (evaluations, feedback, repair): numpy releases the GIL, so a process
with many chains would otherwise use several cores. FRESH_CPU raises the limit."""
import os
import threading

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # effective for libraries loaded after this import

LOCK = threading.BoundedSemaphore(int(os.environ.get("FRESH_CPU", "1")))
