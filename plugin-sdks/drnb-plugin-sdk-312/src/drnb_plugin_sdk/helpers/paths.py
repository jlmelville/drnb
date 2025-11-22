from __future__ import annotations

from pathlib import Path

from drnb_plugin_sdk.protocol import PluginNeighbors, PluginRequest


def _pick_path(req: PluginRequest, attr: str, *, allow_none: bool) -> str | None:
    path = getattr(req.input, attr, None)
    if path is None:
        if allow_none:
            return None
        raise FileNotFoundError(f"Request missing input.{attr}")
    if not Path(path).exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    return path


def resolve_x_path(req: PluginRequest) -> str:
    """Return the required path for the feature matrix; raises if missing."""
    return _pick_path(req, "x_path", allow_none=False)


def resolve_init_path(req: PluginRequest) -> str | None:
    """Return the initialization path if provided; raises if a provided path is missing."""
    return _pick_path(req, "init_path", allow_none=True)


def resolve_neighbors(req: PluginRequest) -> PluginNeighbors:
    """Return neighbor paths; if a path is supplied but missing, raise an error."""
    neighbors = req.input.neighbors
    if neighbors is None:
        return PluginNeighbors()

    idx_path = neighbors.idx_path
    dist_path = neighbors.dist_path
    if idx_path and not Path(idx_path).exists():
        raise FileNotFoundError(f"Neighbor index path not found: {idx_path}")
    if dist_path and not Path(dist_path).exists():
        raise FileNotFoundError(f"Neighbor distance path not found: {dist_path}")

    return neighbors
