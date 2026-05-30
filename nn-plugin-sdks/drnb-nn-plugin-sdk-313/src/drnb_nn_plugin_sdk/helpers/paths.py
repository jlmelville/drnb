from __future__ import annotations

from pathlib import Path

from drnb_nn_plugin_sdk.protocol import NNPluginRequest


def resolve_x_path(req: NNPluginRequest) -> str:
    """Return the required path for the feature matrix; raises if missing."""
    x_path = getattr(req.input, "x_path", None) if req.input else None
    if x_path is None:
        raise FileNotFoundError("Request missing input.x_path")
    if not Path(x_path).exists():
        raise FileNotFoundError(f"Input path not found: {x_path}")
    return x_path
