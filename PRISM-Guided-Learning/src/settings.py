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


# LLM (local Ollama)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b-q4_K_M")
OLLAMA_NUM_CTX = 16384
OLLAMA_NUM_PREDICT = 8192
OLLAMA_THINK = False
