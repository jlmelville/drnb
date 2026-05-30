"""Expose the local SDK source tree on `sys.path` for tests."""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))
