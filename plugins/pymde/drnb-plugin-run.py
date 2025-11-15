#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from drnb_plugin_sdk import protocol as sdk_protocol

from drnb.embed.context import get_neighbors_with_ctx
from drnb.embed.deprecated.pymde import (
    embed_pymde_nbrs,
    nn_to_graph,
    pymde_n_neighbors,
)


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


def _load_init(req: dict[str, Any], params: dict[str, Any]) -> None:
    init_path = (req.get("input") or {}).get("init_path")
    if init_path:
        params["init"] = np.load(init_path, allow_pickle=False)


def _build_graph(
    req: dict[str, Any],
    params: dict[str, Any],
    x: np.ndarray,
    ctx,
) -> Any | None:
    options = req.get("options") or {}
    use_knn = options.get("use_precomputed_knn")
    if use_knn is None:
        use_knn = True
    if not use_knn:
        return None

    if ctx is None:
        _log("No EmbedContext supplied; PyMDE plugin cannot reuse neighbors")
        return None

    metric = params.get("distance", "euclidean")
    default_neighbors = pymde_n_neighbors(x.shape[0])
    n_neighbors = int(params.get("n_neighbors", default_neighbors)) + 1

    pre = get_neighbors_with_ctx(x, metric, n_neighbors, ctx=ctx)
    if pre is None:
        _log("No precomputed knn available; PyMDE will use its internal graph")
        return None
    return nn_to_graph(pre)


def run_method(req: dict[str, Any], method: str) -> dict[str, Any]:
    if method != "pymde-plugin":
        raise RuntimeError(f"unknown method {method}")

    ctx = sdk_protocol.context_from_payload(req.get("context"))
    x = np.load(req["input"]["x_path"], allow_pickle=False)
    params = dict(req.get("params") or {})

    _load_init(req, params)
    graph = _build_graph(req, params, x, ctx)

    seed = params.pop("seed", None)

    _log(f"Running PyMDE with params={params}")
    coords = embed_pymde_nbrs(x, seed=seed, params=params, graph=graph).astype(
        np.float32, copy=False
    )

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
