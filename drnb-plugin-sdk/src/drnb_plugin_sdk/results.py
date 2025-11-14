from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def save_result_npz(
    output_path: str | Path,
    coords: np.ndarray,
    snapshots: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Write coords (and optional snapshots) to an .npz file."""
    out_path = Path(output_path).resolve()
    payload = {"coords": np.asarray(coords, dtype=np.float32, order="C")}
    if snapshots:
        for key, value in snapshots.items():
            payload[key] = np.asarray(value, dtype=np.float32, order="C")
    np.savez_compressed(out_path, **payload)
    return {"ok": True, "result_npz": str(out_path)}
