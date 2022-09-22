import pickle

import numpy as np
import pandas as pd
import scipy.sparse
from numba import jit, prange
from scipy.sparse.csgraph import connected_components

from drnb.io import data_relative_path
from drnb.io.dataset import get_dataset_info
from drnb.log import log
from drnb.neighbors import read_neighbors


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


def n_components(nbrs):
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


# https://stackoverflow.com/a/38547818/4096483
def describe(df):
    if isinstance(df, np.ndarray):
        df = pd.Series(df)
    d = df.describe()
    return pd.concat(
        [d, df.agg(["skew"]), pd.Series(df[df == 0].count(), index=["#0"])]
    )


def get_nbrs(name, n_neighbors):
    nbrs = read_neighbors(
        name,
        n_neighbors=n_neighbors + 1,
        exact=True,
    )
    if nbrs is None:
        raise ValueError(f"Couldn't get {n_neighbors} for {name}")
    nbrs.idx = nbrs.idx[:, 1:]
    nbrs.dist = nbrs.dist[:, 1:]
    return nbrs


def nbr_stats(name, n_neighbors):
    nbrs = get_nbrs(name, n_neighbors)
    ko_desc, ko = ko_data(nbrs)
    so_desc, so = so_data(nbrs)
    nc = n_components(nbrs)
    ndim = get_dataset_info(name).n_dim
    return dict(
        name=name,
        ndim=ndim,
        idx_path=nbrs.info.idx_path,
        nc=nc,
        n_neighbors=n_neighbors,
        ko_desc=ko_desc,
        so_desc=so_desc,
        ko=ko,
        so=so,
    )


def ko_data(nbrs, n_neighbors=None):
    if n_neighbors is None:
        n_neighbors = nbrs.idx.shape[1]
    ko = k_occurrences(nbrs.idx, n_neighbors=n_neighbors)
    ko_desc = describe(ko)
    ko_desc.name = nbrs.info.name
    return ko_desc, ko


def so_data(nbrs, n_neighbors=None):
    if n_neighbors is None:
        n_neighbors = nbrs.idx.shape[1]
    so = s_occurrences(nbrs.idx, n_neighbors=n_neighbors)
    so_desc = describe(so)
    so_desc.name = nbrs.info.name
    return so_desc, so


def idx_to_stats_path(name, idx_path):
    return idx_path.parent / "".join(
        [name] + idx_path.suffixes[:-2] + [".stats", ".npy"]
    )


def write_nbr_stats(nstats):
    idx_path = nstats["idx_path"]
    stats_path = idx_to_stats_path(nstats["name"], idx_path)
    log.info("Writing pkl format to %s", data_relative_path(stats_path))
    with open(stats_path, "wb") as f:
        pickle.dump(nstats, f, pickle.HIGHEST_PROTOCOL)


def read_nbr_stats(name, n_neighbors):
    nbrs = get_nbrs(name, n_neighbors)
    # unlike the neighbor data itself, the stats file has to be an exact match
    # otherwise we won't find it
    if nbrs.info.n_nbrs != n_neighbors + 1:
        raise FileNotFoundError(f"Couldn't find nbr stats for {name}, {n_neighbors}")
    stats_path = idx_to_stats_path(nbrs.info.name, nbrs.info.idx_path)
    with open(stats_path, "rb") as f:
        return pickle.load(f)
