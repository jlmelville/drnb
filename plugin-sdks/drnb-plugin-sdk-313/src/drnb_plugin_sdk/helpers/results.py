from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def save_result_npz(
    output_path: str | Path,
    coords: np.ndarray,
    snapshots: Mapping[str, np.ndarray] | None = None,
    version: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Write coords (and optional snapshots) to an .npz file."""
    out_path = Path(output_path).resolve()
    payload = {"coords": np.asarray(coords, dtype=np.float32, order="C")}
    if snapshots:
        for key, value in snapshots.items():
            payload[key] = np.asarray(value, dtype=np.float32, order="C")
    np.savez_compressed(out_path, **payload)
    response: dict[str, Any] = {"ok": True, "result_npz": str(out_path)}
    if version is not None:
        if isinstance(version, Mapping):
            response["version"] = dict(version)
        else:
            response["version"] = str(version)
    return response


def write_response_json(response_path: str | Path, payload: Mapping[str, Any]) -> str:
    """Persist the final plugin response JSON to disk."""
    out_path = Path(response_path).resolve()
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(out_path)
