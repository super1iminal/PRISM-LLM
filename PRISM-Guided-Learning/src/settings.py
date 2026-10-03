import os
import shutil
from pathlib import Path

# Project layout (all paths absolute, so scripts can be run from any cwd)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOMAINS_PATH = PROJECT_ROOT / "domains"
OUT_PATH = PROJECT_ROOT / "out"
LOGGING_PATH = OUT_PATH / "logs"
RESULTS_PATH = OUT_PATH / "results"


def get_prism_path() -> str:
    """PRISM executable: $PRISM_PATH if set, otherwise `prism` on PATH."""
    path = os.environ.get("PRISM_PATH") or shutil.which("prism")
    if not path:
        raise FileNotFoundError("PRISM executable not found. Set PRISM_PATH or add PRISM's bin/ to PATH.")
    return path


# Run settings (LLM, planner, PRISM limits, ...) live in configs/default.yaml; see src/config.py.
