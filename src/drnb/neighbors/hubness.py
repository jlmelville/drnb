import numpy as np
import scipy.sparse
from numba import jit, prange
from scipy.sparse.csgraph import connected_components


def k_occurrences(idx, n_neighbors=None):
    if n_neighbors is None:
        n_neighbors = idx.shape[1]
    if n_neighbors > idx.shape[1]:
        raise ValueError(f"{n_neighbors} > {idx.shape[1]}")

    return _k_occurrences(idx, n_neighbors)


@jit(nopython=True)
def _k_occurrences(idx, n_neighbors):
    n_items = idx.shape[0]
    result = np.zeros(dtype=np.int32, shape=n_items)

    for i in range(n_items):
        for j in range(n_neighbors):
            result[idx[i][j]] += 1

    return result


def s_occurrences(idx, n_neighbors=None):
    if n_neighbors is None:
        n_neighbors = idx.shape[1]
    if n_neighbors > idx.shape[1]:
        raise ValueError(f"{n_neighbors} > {idx.shape[1]}")

    return _s_occurrences(idx, n_neighbors)


@jit(nopython=True, parallel=True)
def _s_occurrences(idx, n_neighbors):
    n_items = idx.shape[0]
    result = np.zeros(dtype=np.int32, shape=n_items)

    # pylint: disable=not-an-iterable
    for i in prange(n_items):
        for j in range(n_neighbors):
            nbr_ij = idx[i, j]
            for k in range(n_neighbors):
                if idx[nbr_ij, k] == i:
                    result[i] += 1
                    break
    return result


def get_n_components(nbrs):
    wadj_graph = nn_to_sparse(nbrs)
    return connected_components(csgraph=wadj_graph, directed=True, return_labels=False)


@jit(nopython=True)
def _nn_to_sparse(idx, dist):
    n_items = idx.shape[0]
    n_neighbors = idx.shape[1]
    size = idx.size
    rows = np.zeros(size, dtype=np.int32)
    cols = np.zeros(size, dtype=np.int32)
    vals = np.zeros(size, dtype=np.float32)

    for i in range(n_items):
        inbrs = i * n_neighbors
        for j in range(n_neighbors):
            idx1d = inbrs + j
            rows[idx1d] = i
            cols[idx1d] = idx[i, j]
            vals[idx1d] = dist[i, j]

    return rows, cols, vals


def nn_to_sparse(nbrs):
    idx = nbrs.idx
    dist = nbrs.dist
    n_items = idx.shape[0]
    rows, cols, vals = _nn_to_sparse(idx, dist)

    return scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n_items, n_items))
