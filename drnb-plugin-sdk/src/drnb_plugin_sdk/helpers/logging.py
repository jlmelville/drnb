from __future__ import annotations

import sys
from typing import Any


def log(*parts: Any) -> None:
    """Print a message to stderr immediately."""
    print(*parts, file=sys.stderr, flush=True)
