#!/usr/bin/env python
from __future__ import annotations

import numpy as np
from drnb_nn_plugin_sdk.helpers.logging import log, summarize_params
from drnb_nn_plugin_sdk.helpers.paths import resolve_x_path
from drnb_nn_plugin_sdk.helpers.results import save_neighbors_npz
from drnb_nn_plugin_sdk.helpers.runner import run_nn_plugin
from drnb_nn_plugin_sdk.protocol import NNPluginRequest
from hnswlib import Index as HnswIndex
from sklearn.utils import check_random_state

HNSW_METRICS = {
    "cosine": "cosine",
    "dot": "ip",
    "euclidean": "l2",
    "l2": "l2",
}


def _handler(req: NNPluginRequest) -> dict:
    log(f"[hnsw] request metric={req.metric} n_neighbors={req.n_neighbors}")
    params = req.params or {}
    log(f"[hnsw] params={summarize_params(params)}")

    x_path = resolve_x_path(req)
    X = np.load(x_path, allow_pickle=False)

    metric = req.metric
    n_neighbors = int(req.n_neighbors)
    M = int(params.get("M", 16))
    ef_construction = int(params.get("ef_construction", 200))
    random_state = int(params.get("random_state", 42))
    n_jobs = int(params.get("n_jobs", -1))

    hnsw_space = HNSW_METRICS[metric]

    rng = check_random_state(random_state)
    random_seed = int(rng.randint(np.iinfo(np.int32).max))

    index = HnswIndex(space=hnsw_space, dim=X.shape[1])
    index.init_index(
        max_elements=X.shape[0],
        ef_construction=ef_construction,
        M=M,
        random_seed=random_seed,
    )
    index.add_items(X, num_threads=n_jobs)
    index.set_ef(min(2 * n_neighbors, index.get_current_count()))

    # Match legacy behavior: request n_neighbors+1 to include self, caller may slice.
    k = n_neighbors + 1
    indices, distances = index.knn_query(X, k=k, num_threads=n_jobs)
    if metric == "euclidean":
        distances = np.sqrt(distances)

    result = save_neighbors_npz(req.output.result_path, indices, distances)  # type: ignore[arg-type]
    return result


if __name__ == "__main__":
    run_nn_plugin({"hnsw": _handler}, description="drnb HNSW NN plugin")
