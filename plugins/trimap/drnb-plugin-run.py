#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimap
from drnb_plugin_sdk import protocol as sdk_protocol

from drnb.embed.context import get_neighbors_with_ctx


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


def _neighbor_tuple(
    req: dict[str, Any], x: np.ndarray, ctx
) -> tuple[np.ndarray, np.ndarray | None] | None:
    neigh = (req.get("input") or {}).get("neighbors") or {}
    idx_path = neigh.get("idx_path")
    if idx_path:
        idx = np.load(idx_path, allow_pickle=False)
        dist_path = neigh.get("dist_path")
        dist = np.load(dist_path, allow_pickle=False) if dist_path else None
        return idx, dist
    options = req.get("options") or {}
    if ctx is None or not options.get("use_precomputed_knn", True):
        return None
    params = req.get("params") or {}
    metric = params.get("metric", "euclidean")
    n_neighbors = int(params.get("n_neighbors", 15))
    pre = get_neighbors_with_ctx(x, metric, n_neighbors, ctx=ctx)
    if pre is None:
        return None
    return pre.idx, pre.dist


def run_trimap(req: dict[str, Any]) -> dict[str, Any]:
    ctx = sdk_protocol.context_from_payload(req.get("context"))
    x = np.load(req["input"]["x_path"], allow_pickle=False)
    params = dict(req.get("params") or {})

    init = params.pop("init", None)
    return_every = params.pop("return_every", None)
    knn_tuple = _neighbor_tuple(req, x, ctx)
    if knn_tuple is not None:
        params["knn_tuple"] = knn_tuple
        _log("Using precomputed KNN data for TriMap")

    return_seq = bool(params.get("return_seq", False))
    orig_return_every = trimap.trimap_._RETURN_EVERY
    if return_every is None:
        return_every = orig_return_every

    try:
        if return_every != orig_return_every:
            trimap.trimap_._RETURN_EVERY = return_every
        _log(f"Running TriMap with params={params}")
        embedder = trimap.TRIMAP(n_dims=2, **params)
        result = embedder.fit_transform(x, init=init)
    finally:
        if trimap.trimap_._RETURN_EVERY != orig_return_every:
            trimap.trimap_._RETURN_EVERY = orig_return_every

    save: dict[str, np.ndarray] = {}
    if return_seq:
        coords = result[:, :, -1].astype(np.float32, copy=False)
        save["coords"] = coords
        for i in range(result.shape[-1]):
            iteration = i * return_every
            save[f"snap_{iteration}"] = result[:, :, i].astype(np.float32, copy=False)
    else:
        save["coords"] = result.astype(np.float32, copy=False)

    out_path = Path(req["output"]["result_path"]).resolve()
    np.savez_compressed(out_path, **save)
    return {"ok": True, "result_npz": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    req_path = Path(args.request)
    try:
        req = _load_request(req_path)
        if args.method != "trimap-plugin":
            raise RuntimeError(f"unknown method {args.method}")
        resp = run_trimap(req)
    except Exception as exc:  # noqa: BLE001
        resp = {"ok": False, "message": str(exc)}

    print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    main()
