import numpy as np

# pylint: disable=no-name-in-module
from hnswlib import Index as HnswIndex
from sklearn.utils import check_random_state

HNSW_DEFAULTS = {
    "M": 16,
    "ef_construction": 200,
    "random_state": 42,
    "n_jobs": -1,
}

# adapted from openTSNE
# basically l2 or ip
# cosine = ip with normalization
HNSW_METRICS = {
    "cosine": "cosine",
    "dot": "ip",
    "euclidean": "l2",
    "l2": "l2",
}


def hnsw_neighbors(
    X: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    M: int = 16,
    ef_construction: int = 200,
    random_state: int = 42,
    n_jobs: int = -1,
    return_distance: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the nearest neighbors using the HNSW library. If `return_distance` is
    True, the function will return both the indices and the distances to the neighbors.
    Otherwise, it will only return the indices."""
    hnsw_space = HNSW_METRICS[metric]

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
        if metric == "euclidean":
            distances = np.sqrt(distances)
        return indices, distances
    return indices
