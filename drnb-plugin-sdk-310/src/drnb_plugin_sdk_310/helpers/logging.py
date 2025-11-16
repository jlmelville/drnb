from __future__ import annotations

import sys
from typing import Any, Mapping

import numpy as np


def log(*parts: Any) -> None:
    """Print a message to stderr immediately."""
    print(*parts, file=sys.stderr, flush=True)


def summarize_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return a log-friendly view of params, eliding large/complex values."""

    def _summarize(value: Any) -> Any:
        if isinstance(value, (int, float, bool, str)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            if all(isinstance(x, (int, float, bool, str)) or x is None for x in value):
                return list(value) if isinstance(value, tuple) else value
            return f"<{type(value).__name__} len={len(value)}>"
        if isinstance(value, np.ndarray):
            return f"<ndarray shape={value.shape} dtype={value.dtype}>"
        return f"<{type(value).__name__}>"

    return {str(key): _summarize(val) for key, val in params.items()}
