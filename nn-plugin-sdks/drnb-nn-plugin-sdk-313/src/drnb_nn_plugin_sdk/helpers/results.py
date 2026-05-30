from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def save_neighbors_npz(
    output_path: str | Path,
    idx: np.ndarray,
    dist: np.ndarray | None = None,
) -> dict[str, Any]:
    """Write neighbor idx/dist arrays to an .npz file."""
    out_path = Path(output_path).resolve()
    payload: dict[str, np.ndarray] = {
        "idx": np.asarray(idx, dtype=np.int32, order="C"),
    }
    if dist is not None:
        payload["dist"] = np.asarray(dist, dtype=np.float32, order="C")
    np.savez_compressed(out_path, **payload)
    return {"ok": True, "result_npz": str(out_path)}


def write_response_json(response_path: str | Path, payload: Mapping[str, Any]) -> str:
    """Persist the final plugin response JSON to disk."""
    out_path = Path(response_path).resolve()
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(out_path)
