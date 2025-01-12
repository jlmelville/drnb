import numpy as np
from numba import jit, prange

from drnb.distances import distance_function
from drnb.types import Callable


def neighbor_distances(
    data: np.ndarray, idx: np.ndarray, distance: str = "euclidean"
) -> np.ndarray:
    """Compute the distances between the data points and the neighbors in `idx`."""
    dist_fun = distance_function(distance)
    return _neighbor_distances(data, idx, dist_fun)


@jit(nopython=True, parallel=True)
def _neighbor_distances(
    data: np.ndarray,
    idx: np.ndarray,
    dist_fun: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Compute the distances between the data points and the neighbors in `idx`."""
    n_items = data.shape[0]
    n_neighbors = idx.shape[1]
    distances = np.empty(idx.shape, dtype=np.float32)
    # pylint: disable=not-an-iterable
    for i in prange(n_items):
        idxi = idx[i]
        di = data[i]
        for j in range(n_neighbors):
            distances[i, j] = dist_fun(di, data[idxi[j]])
    return distances
