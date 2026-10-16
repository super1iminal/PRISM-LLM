"""One numpy-heavy step at a time per process (evaluations, feedback, repair): numpy releases the GIL, so a process
with many chains would otherwise use several cores. FRESH_CPU raises the limit."""
import os
import threading

LOCK = threading.BoundedSemaphore(int(os.environ.get("FRESH_CPU", "1")))
