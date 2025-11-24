#!/usr/bin/env python
from __future__ import annotations

import numpy as np
from drnb_nn_plugin_sdk.helpers.logging import log, summarize_params
from drnb_nn_plugin_sdk.helpers.paths import resolve_x_path
from drnb_nn_plugin_sdk.helpers.results import save_neighbors_npz
from drnb_nn_plugin_sdk.helpers.runner import run_nn_plugin
from drnb_nn_plugin_sdk.protocol import NNPluginRequest


def _load_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "faiss not installed in plugin env; install faiss-cpu or faiss-gpu"
        ) from exc
    return faiss


def _handler(req: NNPluginRequest) -> dict:
    log(f"[faiss] request metric={req.metric} n_neighbors={req.n_neighbors}")
    params = req.params or {}
    log(f"[faiss] params={summarize_params(params)}")

    faiss = _load_faiss()

    x_path = resolve_x_path(req)
    X = np.load(x_path, allow_pickle=False)
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)

    metric = req.metric
    n_neighbors = int(req.n_neighbors)
    use_gpu = bool(params.get("use_gpu", True))

    if metric == "cosine":
        faiss_space = faiss.IndexFlatIP
        faiss.normalize_L2(X)
    elif metric == "euclidean":
        faiss_space = faiss.IndexFlatL2
    else:
        raise ValueError(f"Unsupported metric for faiss: '{metric}'")

    index_flat = faiss_space(X.shape[1])

    try:
        if use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index_flat)
        else:
            index = index_flat
    except Exception:  # noqa: BLE001
        log("[faiss] falling back to CPU index")
        index = index_flat

    index.add(X)
    distances, indices = index.search(X, n_neighbors)

    if metric == "euclidean":
        distances = np.sqrt(distances)
    elif metric == "cosine":
        distances = 1.0 - distances

    result = save_neighbors_npz(req.output.result_path, indices, distances)  # type: ignore[arg-type]
    return result


if __name__ == "__main__":
    run_nn_plugin({"faiss": _handler}, description="drnb FAISS NN plugin")
