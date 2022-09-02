import numpy as np
import pynndescent
import sklearn.metrics
from annoy import AnnoyIndex
from hnswlib import Index as HnswIndex
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state


def dmat(x):
    if isinstance(x, tuple) and len(x) == 2:
        x = x[0]
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    return sklearn.metrics.pairwise_distances(x)


def sklearn_neighbors(
    X,
    n_neighbors=15,
    algorithm="auto",
    metric="minkowski",
    metric_kwds=None,
    n_jobs=-1,
    return_distance=True,
):

    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        n_jobs=n_jobs,
        metric=metric,
        metric_params=metric_kwds,
    ).fit(X)

    knn = nn.kneighbors(X, return_distance=return_distance)

    if return_distance:
        return knn[1], knn[0]
    return knn


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


def hnsw_neighbors(
    X,
    n_neighbors=15,
    metric="euclidean",
    M=16,
    ef_construction=200,
    random_state=42,
    n_jobs=-1,
    return_distance=True,
):

    # adapted from openTSNE
    hnsw_space = {
        "cosine": "cosine",
        "dot": "ip",
        "euclidean": "l2",
        "ip": "ip",
        "l2": "l2",
    }[metric]

    random_state = check_random_state(random_state)
    random_seed = random_state.randint(np.iinfo(np.int32).max)

    index = HnswIndex(space=hnsw_space, dim=X.shape[1])

    # Initialize HNSW Index
    index.init_index(
        max_elements=X.shape[0],
        ef_construction=ef_construction,
        M=M,
        random_seed=random_seed,
    )

    # Build index tree from data
    index.add_items(X, num_threads=n_jobs)

    # Set ef parameter for (ideal) precision/recall
    index.set_ef(min(2 * n_neighbors, index.get_current_count()))

    # Query for kNN
    indices, distances = index.knn_query(X, k=n_neighbors + 1, num_threads=n_jobs)

    if return_distance:
        return indices, distances
    return indices


# also adapted from openTSNE
def annoy_neighbors(
    X,
    n_neighbors=15,
    metric="euclidean",
    random_state=42,
    n_trees=50,
    search_k=-1,
    n_jobs=-1,
    return_distance=True,
):
    N = X.shape[0]

    annoy_aliases = {
        "cosine": "angular",
        "l1": "manhattan",
        "l2": "euclidean",
        "taxicab": "manhattan",
    }
    metric = annoy_aliases.get(metric, "euclidean")

    index = AnnoyIndex(X.shape[1], metric)

    random_state = check_random_state(random_state)
    index.set_seed(random_state.randint(np.iinfo(np.int32).max))

    for i in range(N):
        index.add_item(i, X[i])

    index.build(n_trees, n_jobs=n_jobs)

    # Return the nearest neighbors in the training set
    distances = np.zeros((N, n_neighbors))
    indices = np.zeros((N, n_neighbors)).astype(int)

    def getnns(i):
        indices_i, distances_i = index.get_nns_by_item(
            i, n_neighbors, search_k=search_k, include_distances=True
        )
        indices[i] = indices_i
        distances[i] = distances_i

    if n_jobs == 1:
        for i in range(N):
            getnns(i)
    else:
        from joblib import Parallel, delayed  # pylint: disable=import-outside-toplevel

        Parallel(n_jobs=n_jobs, require="sharedmem")(
            delayed(getnns)(i) for i in range(N)
        )

    if return_distance:
        return indices, distances
    return indices


#     init_graph: np.ndarray (optional, default=None)
#     2D array of indices of candidate neighbours of the shape
#     (data.shape[0], n_neighbours). If the j-th neighbour of the i-th
#     instances is unknown, use init_graph[i, j] = -1
# init_dist: np.ndarray (optional, default=None)
#     2D array with the same shape as init_graph,
#     such that metric(data[i], data[init_graph[i, j]]) equals
#     init_dist[i, j]
