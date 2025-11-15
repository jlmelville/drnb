#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import ncvis
import numpy as np
import protocol_compat as sdk_protocol


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _load_request(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    proto = data.get("protocol") or data.get("protocol_version")
    if proto != sdk_protocol.PROTOCOL_VERSION:
        raise RuntimeError(
            f"protocol mismatch: expected {sdk_protocol.PROTOCOL_VERSION}, got {proto}"
        )
    return data


def run_method(req: dict[str, Any], method: str) -> dict[str, Any]:
    if method != "ncvis-plugin":
        raise RuntimeError(f"unknown method {method}")

    sdk_protocol.context_from_payload(req.get("context"))

    x = np.load(req["input"]["x_path"], allow_pickle=False)
    params = dict(req.get("params") or {})

    _log(f"Running NCVis with params={params}")
    embedder = ncvis.NCVis(**params)
    coords = embedder.fit_transform(x).astype(np.float32, copy=False)

    result_path = Path(req["output"]["result_path"]).resolve()
    np.savez_compressed(result_path, coords=coords)
    return {"ok": True, "result_npz": str(result_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    req = _load_request(Path(args.request))
    try:
        resp = run_method(req, args.method)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        _log(tb)
        resp = {"ok": False, "message": tb}

    print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    main()
