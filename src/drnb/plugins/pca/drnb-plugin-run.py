#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from drnb.plugins.protocol import PROTOCOL_VERSION, context_from_payload


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _load_request(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    proto = data.get("protocol") or data.get("protocol_version")
    if proto != PROTOCOL_VERSION:
        raise RuntimeError(
            f"protocol mismatch: expected {PROTOCOL_VERSION}, got {proto}"
        )
    return data


def run_pca(req: dict) -> dict:
    context_from_payload(req.get("context"))  # ensure it deserializes even if unused
    x = np.load(req["input"]["x_path"], allow_pickle=False)
    params = dict(req.get("params") or {})
    n_components = int(params.pop("n_components", 2))

    _log(f"Running PCA with params={params} n_components={n_components}")
    pca = PCA(n_components=n_components, **params)
    coords = pca.fit_transform(x).astype(np.float32, copy=False)

    result_path = Path(req["output"]["result_path"]).resolve()
    np.savez_compressed(result_path, coords=coords)
    return {"ok": True, "result_npz": str(result_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    req_path = Path(args.request)
    try:
        req = _load_request(req_path)
        if args.method != "pca-plugin":
            raise RuntimeError(f"unknown method {args.method}")
        resp = run_pca(req)
    except Exception as exc:  # noqa: BLE001
        resp = {"ok": False, "message": str(exc)}

    print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    main()
