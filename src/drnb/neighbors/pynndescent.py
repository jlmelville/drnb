import numpy as np
import pynndescent

PYNNDESCENT_METRICS = pynndescent.pynndescent_.pynnd_dist.named_distances

PYNNDESCENT_DEFAULTS = dict(
    metric_kwds=None,
    random_state=42,
    low_memory=True,
    n_trees=None,
    n_iters=None,
    max_candidates=60,
    n_jobs=-1,
)


def pynndescent_neighbors(
    X,
    n_neighbors=15,
    metric="euclidean",
    metric_kwds=None,
    random_state=42,
    low_memory=True,
    n_trees=None,
    n_iters=None,
    max_candidates=60,
    n_jobs=-1,
    return_distance=True,
    verbose=False,
):
    # these values from the UMAP source code
    if n_trees is None:
        n_trees = min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
    if n_iters is None:
        n_iters = max(5, int(round(np.log2(X.shape[0]))))

    knn_search_index = pynndescent.NNDescent(
        X,
        n_neighbors=n_neighbors,
        metric=metric,
        metric_kwds=metric_kwds,
        random_state=random_state,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=max_candidates,
        low_memory=low_memory,
        n_jobs=n_jobs,
        verbose=verbose,
        compressed=False,
    )
    knn_indices, knn_dists = knn_search_index.neighbor_graph
    if return_distance:
        return knn_indices, knn_dists
    return knn_indices


#     init_graph: np.ndarray (optional, default=None)
#     2D array of indices of candidate neighbours of the shape
#     (data.shape[0], n_neighbours). If the j-th neighbour of the i-th
#     instances is unknown, use init_graph[i, j] = -1
# init_dist: np.ndarray (optional, default=None)
#     2D array with the same shape as init_graph,
#     such that metric(data[i], data[init_graph[i, j]]) equals
#     init_dist[i, j]
