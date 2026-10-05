import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # test helpers (fakes.py)


@pytest.fixture
def tiny_grid(tmp_path):
    """Path of a one-instance gridworld dataset small enough for fast PRISM checks."""
    from fakes import write_tiny_grid
    return write_tiny_grid(tmp_path)
