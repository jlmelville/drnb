#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import umato

from drnb.plugins.protocol import PROTOCOL_VERSION, context_from_payload

DEFAULT_HUB_NUM = 300


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _load_request(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    proto = data.get("protocol") or data.get("protocol_version")
    if proto != PROTOCOL_VERSION:
        raise RuntimeError(
            f"protocol mismatch: expected {PROTOCOL_VERSION}, got {proto}"
        )
    return data


def _adjust_hub_num(x: np.ndarray, params: dict[str, Any]) -> None:
    n_samples = x.shape[0]
    hub_num = int(params.get("hub_num", DEFAULT_HUB_NUM))
    if n_samples < DEFAULT_HUB_NUM and hub_num > n_samples:
        adjusted = max(1, n_samples // 3)
        _log(
            f"Reducing hub_num from {hub_num} to {adjusted} for dataset size {n_samples}"
        )
        params["hub_num"] = adjusted


def run_method(req: dict[str, Any], method: str) -> dict[str, Any]:
    if method != "umato-plugin":
        raise RuntimeError(f"unknown method {method}")

    context_from_payload(req.get("context"))

    x = np.load(req["input"]["x_path"], allow_pickle=False)
    params = dict(req.get("params") or {})

    _adjust_hub_num(x, params)

    _log(f"Running UMATO with params={params}")
    embedder = umato.UMATO(**params)
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
