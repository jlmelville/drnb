import numpy as np
from annoy import AnnoyIndex
from sklearn.utils import check_random_state

ANNOY_DEFAULTS = {
    "n_trees": 50,
    "search_k": -1,
    "random_state": 42,
    "n_jobs": -1,
}

ANNOY_METRICS = {
    "dot": "dot",
    "cosine": "angular",
    "manhattan": "manhattan",
    "euclidean": "euclidean",
}


# also adapted from openTSNE
def annoy_neighbors(
    X: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    random_state: int = 42,
    n_trees: int = 50,
    search_k: int = -1,
    n_jobs: int = -1,
    return_distance: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the nearest neighbors using the Annoy library. If `return_distance` is
    True, the function will return both the indices and the distances to the neighbors.
    Otherwise, it will only return the indices."""
    N = X.shape[0]
    annoy_space = ANNOY_METRICS[metric]

    index = AnnoyIndex(X.shape[1], annoy_space)

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
