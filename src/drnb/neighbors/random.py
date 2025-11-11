from typing import Tuple, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from . import NearestNeighbors
from .distances import neighbor_distances


def sort_neighbors(idx: np.ndarray, dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort the neighbors by distance."""
    # sort each row by ascending distance
    dist_ind = dist.argsort()
    dist = np.take_along_axis(dist, dist_ind, axis=1)
    idx = np.take_along_axis(idx, dist_ind, axis=1)
    return idx, dist


def logn_neighbors(n_items: int | NDArray) -> int:
    """Compute the log number of neighbors based on the number of items."""
    if isinstance(n_items, np.ndarray):
        n_items = n_items.shape[0]
    return np.ceil(np.log(n_items)).astype(int)


def random_neighbors(
    data: np.ndarray,
    n_neighbors: int = None,
    distance: str = "euclidean",
    random_state: int = 42,
) -> NearestNeighbors:
    """Compute random neighbors for each data point."""
    n_items = data.shape[0]
    if n_neighbors is None:
        n_neighbors = logn_neighbors(n_items)
    idx = random_idx(n_items, n_neighbors, random_state=random_state)
    dist = neighbor_distances(data, idx, distance)

    idx, dist = sort_neighbors(idx, dist)
    return NearestNeighbors(idx=idx, dist=dist)


def random_idx(n: int, n_neighbors: int, random_state: int = 42) -> np.ndarray:
    """Generate random indices for neighbors."""
    result = np.empty((n, n_neighbors), dtype=int)
    rng = np.random.default_rng(random_state)
    for i in range(n):
        result[i] = rng.choice(n - 1, size=n_neighbors, replace=False)
        for j in range(n_neighbors):
            if result[i, j] >= i:
                result[i, j] += 1
    return result


def mid_near_neighbors(
    data: np.ndarray,
    n_neighbors: int = 8,
    metric: str = "euclidean",
    n_random: int = 6,
    mid_index: int = 1,
    random_state: int = 42,
) -> NearestNeighbors:
    """Generate n_neighbors mid-near neighbors for each data point. To generate each
    neighbor, n_random random neighbors are generated and after ordering by distance,
    the mid_index-th neighbor is selected."""
    if mid_index > n_random - 1:
        raise ValueError(f"mid_index must be in range [0, {n_random - 1}]")
    n = data.shape[0]
    mn_idx = np.empty((n, n_neighbors), dtype=int)
    mn_dist = np.empty((n, n_neighbors), dtype=float)
    random_state = check_random_state(random_state)
    for i in range(n_neighbors):
        seed = random_state.randint(np.iinfo(np.uint32).max)
        rnbrs = random_neighbors(
            data=data, n_neighbors=n_random, distance=metric, random_state=seed
        )
        mn_idx[:, i] = rnbrs.idx[:, mid_index]
        rnbrs.dist = cast(NDArray, rnbrs.dist)
        mn_dist[:, i] = rnbrs.dist[:, mid_index]
    mn_idx, mn_dist = sort_neighbors(mn_idx, mn_dist)
    return NearestNeighbors(idx=mn_idx, dist=mn_dist)


def random_sample_nbrs(
    idx: np.ndarray, dist: np.ndarray | None, n_neighbors: int
) -> NearestNeighbors:
    """Randomly sample n_neighbors neighbors for each data point.

    Args:
        idx: The indices of the neighbors.
        dist: The distances to the neighbors.
        n_neighbors: The number of neighbors to sample.

    Returns:
        A NearestNeighbors object with the sampled neighbors.
    """
    mask = np.array(
        [np.random.permutation(idx.shape[1]) < n_neighbors for _ in range(idx.shape[0])]
    )
    rand_idx = idx[mask.reshape(idx.shape[0], -1)].reshape(idx.shape[0], n_neighbors)
    rand_dist = (
        dist[mask.reshape(idx.shape[0], -1)].reshape(idx.shape[0], n_neighbors)
        if dist is not None
        else None
    )
    return NearestNeighbors(idx=rand_idx, dist=rand_dist)
