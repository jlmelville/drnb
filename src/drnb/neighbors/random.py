import numpy as np

from . import NearestNeighbors
from .distances import neighbor_distances


def random_neighbors(data, n_neighbors=None, distance="euclidean", random_state=42):
    n_items = data.shape[0]
    if n_neighbors is None:
        n_neighbors = np.ceil(np.log(n_items)).astype(int)
    idx = random_idx(n_items, n_neighbors, random_state=random_state)
    dist = neighbor_distances(data, idx, distance)

    # sort each row by ascending distance
    dist_ind = dist.argsort()
    dist = np.take_along_axis(dist, dist_ind, axis=1)
    idx = np.take_along_axis(idx, dist_ind, axis=1)
    return NearestNeighbors(idx=idx, dist=dist)


def random_idx(n, n_neighbors, random_state=42):
    result = np.empty((n, n_neighbors), dtype=int)
    rng = np.random.default_rng(random_state)
    for i in range(n):
        result[i] = rng.choice(n - 1, size=n_neighbors, replace=False)
        for j in range(n_neighbors):
            if result[i, j] >= i:
                result[i, j] += 1
    return result
