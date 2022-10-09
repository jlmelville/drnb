import numpy as np
import pynndescent
from numba import jit, prange
from pynndescent.utils import deheap_sort, make_heap, simple_heap_push

from drnb.distances import distance_function

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


def pynndescent_exact_neighbors(
    data, n_neighbors, metric="euclidean", return_distance=True
):
    if metric == "euclidean":
        dist_fun = distance_function("squared_euclidean")
    else:
        dist_fun = distance_function(metric)
    nbrs = _brute_force_knn(data, n_neighbors, dist_fun)
    if return_distance:
        if metric == "euclidean":
            return nbrs[0], np.sqrt(nbrs[1])
        return nbrs[0], nbrs[1]
    return nbrs[0]


@jit(nopython=True, parallel=True)
def _brute_force_knn(data, n_neighbors, dist_fun):
    n_items = data.shape[0]
    heap = make_heap(n_items, n_neighbors)
    # pylint: disable=not-an-iterable
    for i in prange(n_items):
        data_i = data[i]
        priorities = heap[1][i]
        indices = heap[0][i]
        for j in range(n_items):
            d = dist_fun(data_i, data[j])
            simple_heap_push(
                priorities,
                indices,
                d,
                j,
            )
    return deheap_sort(heap[0], heap[1])
