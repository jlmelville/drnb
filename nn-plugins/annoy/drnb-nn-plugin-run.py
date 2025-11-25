#!/usr/bin/env python
from __future__ import annotations

import numpy as np
from annoy import AnnoyIndex
from drnb_nn_plugin_sdk.helpers.logging import log, summarize_params
from drnb_nn_plugin_sdk.helpers.paths import resolve_x_path
from drnb_nn_plugin_sdk.helpers.results import save_neighbors_npz
from drnb_nn_plugin_sdk.helpers.runner import run_nn_plugin
from drnb_nn_plugin_sdk.protocol import NNPluginRequest
from sklearn.utils import check_random_state

ANNOY_METRICS = {
    "dot": "dot",
    "cosine": "angular",
    "manhattan": "manhattan",
    "euclidean": "euclidean",
}


def _handler(req: NNPluginRequest) -> dict:
    log(f"[annoy] request metric={req.metric} n_neighbors={req.n_neighbors}")
    params = req.params or {}
    log(f"[annoy] params={summarize_params(params)}")

    x_path = resolve_x_path(req)
    X = np.load(x_path, allow_pickle=False)

    metric = req.metric
    n_neighbors = int(req.n_neighbors)
    n_trees = int(params.get("n_trees", 50))
    search_k = int(params.get("search_k", -1))
    random_state = int(params.get("random_state", 42))
    n_jobs = int(params.get("n_jobs", -1))

    annoy_space = ANNOY_METRICS[metric]
    index = AnnoyIndex(X.shape[1], annoy_space)

    rng = check_random_state(random_state)
    index.set_seed(rng.randint(np.iinfo(np.int32).max))

    for i, row in enumerate(X):
        index.add_item(int(i), row.tolist())

    index.build(n_trees, n_jobs=n_jobs)

    distances = np.zeros((X.shape[0], n_neighbors), dtype=np.float32)
    indices = np.zeros((X.shape[0], n_neighbors), dtype=np.int32)

    def getnns(i: int):
        nn_indices, nn_distances = index.get_nns_by_item(
            int(i), n_neighbors, search_k=search_k, include_distances=True
        )
        indices[i] = nn_indices
        distances[i] = nn_distances

    if n_jobs == 1:
        for i in range(X.shape[0]):
            getnns(i)
    else:
        from joblib import Parallel, delayed  # pylint: disable=import-outside-toplevel

        Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(getnns)(i) for i in range(X.shape[0])
        )

    result = save_neighbors_npz(req.output.result_path, indices, distances)  # type: ignore[arg-type]
    return result


if __name__ == "__main__":
    run_nn_plugin({"annoy": _handler}, description="drnb Annoy NN plugin")
