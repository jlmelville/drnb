#!/usr/bin/env python
from __future__ import annotations

from typing import Any

import numpy as np
import trimap
from drnb_plugin_sdk import protocol as sdk_protocol
from drnb_plugin_sdk.helpers.logging import log, summarize_params
from drnb_plugin_sdk.helpers.results import save_result_npz
from drnb_plugin_sdk.helpers.runner import run_plugin


def _neighbor_tuple(
    req: sdk_protocol.PluginRequest,
) -> tuple[np.ndarray, np.ndarray | None] | None:
    neighbors = req.input.neighbors
    if neighbors is None or not neighbors.idx_path:
        return None

    idx = np.load(neighbors.idx_path, allow_pickle=False)
    dist = None
    if neighbors.dist_path:
        dist = np.load(neighbors.dist_path, allow_pickle=False)
    return idx, dist


def run_trimap(req: sdk_protocol.PluginRequest) -> dict[str, Any]:
    x = np.load(req.input.x_path, allow_pickle=False)
    params = dict(req.params or {})

    init = params.pop("init", None)
    return_every = params.pop("return_every", None)
    knn_tuple = _neighbor_tuple(req)
    if knn_tuple is not None:
        params["knn_tuple"] = knn_tuple
        log("Using precomputed KNN data for TriMap")

    return_seq = bool(params.get("return_seq", False))
    orig_return_every = trimap.trimap_._RETURN_EVERY
    if return_every is None:
        return_every = orig_return_every

    try:
        if return_every != orig_return_every:
            trimap.trimap_._RETURN_EVERY = return_every
        log(f"Running TriMap with params={summarize_params(params)}")
        embedder = trimap.TRIMAP(n_dims=2, **params)
        result = embedder.fit_transform(x, init=init)
    finally:
        if trimap.trimap_._RETURN_EVERY != orig_return_every:
            trimap.trimap_._RETURN_EVERY = orig_return_every

    snapshots: dict[str, np.ndarray] = {}
    if return_seq:
        coords = result[:, :, -1].astype(np.float32, copy=False)
        for i in range(result.shape[-1]):
            iteration = i * return_every
            snapshots[f"snap_{iteration}"] = result[:, :, i].astype(
                np.float32, copy=False
            )
    else:
        coords = result.astype(np.float32, copy=False)

    return save_result_npz(req.output.result_path, coords, snapshots=snapshots)


if __name__ == "__main__":
    run_plugin({"trimap-plugin": run_trimap})
