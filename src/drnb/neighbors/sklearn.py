import numpy as np
from sklearn.neighbors import NearestNeighbors

SKLEARN_METRICS = {
    "cosine": "cosine",
    "manhattan": "manhattan",
    "euclidean": "euclidean",
}

SKLEARN_DEFAULTS = {
    "algorithm": "auto",
    "n_jobs": -1,
}


def sklearn_neighbors(
    X: np.ndarray,
    n_neighbors: int = 15,
    algorithm="auto",
    metric: str = "euclidean",
    metric_kwds=None,
    n_jobs: int = -1,
    return_distance: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the nearest neighbors using the sklearn library. If `return_distance` is
    True, the function will return both the indices and the distances to the neighbors.
    Otherwise, it will only return the indices."""
    sklearn_space = SKLEARN_METRICS[metric]
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        n_jobs=n_jobs,
        metric=sklearn_space,
        metric_params=metric_kwds,
    ).fit(X)

    knn = nn.kneighbors(X, return_distance=return_distance)

    if return_distance:
        return knn[1], knn[0]
    return knn
