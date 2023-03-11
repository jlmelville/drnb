import pickle

import numpy as np
import pandas as pd
import scipy.sparse
from numba import jit, prange
from scipy.sparse.csgraph import connected_components

from drnb.io import data_relative_path
from drnb.io.dataset import get_dataset_info, list_available_datasets
from drnb.log import log
from drnb.neighbors import read_neighbors
from drnb.util import islisty


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

    idx = idx[:, :n_neighbors]
    return _s_occurrences(idx)


@jit(nopython=True, parallel=True)
def _s_occurrences(idx):
    n_items = idx.shape[0]
    n_neighbors = idx.shape[1]
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


@jit(nopython=True)
def _nn_to_sparse_binary(idx):
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
            vals[idx1d] = 1

    return rows, cols, vals


# symmetrize operation to carry out on matrix and its transpose:
# "and" to carry out AND operation (mutual nearest neighbors only)
# "or" to carry out OR operation (undirected nearest neighbors)
def nn_to_sparse(nbrs, symmetrize=None):
    if isinstance(nbrs, np.ndarray):
        idx = nbrs
        dist = None
    elif islisty(nbrs):
        idx = nbrs[0]
        dist = nbrs[1]
    else:
        idx = nbrs.idx
        dist = nbrs.dist
    n_items = idx.shape[0]

    if dist is None:
        rows, cols, vals = _nn_to_sparse_binary(idx)
    else:
        rows, cols, vals = _nn_to_sparse(idx, dist)

    # creates an asymmetric adjacency matrix from the undirected nn graph
    dmat = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n_items, n_items))

    if symmetrize is not None:
        # convert the asymmetric adjacency matrix to symmetric
        if symmetrize == "or":
            # not sure there is a common word for this relationship: undirected?
            # (i needs to be a neighbor of j OR vice versa)
            dmat = dmat.maximum(dmat.transpose()).tocoo()
        elif symmetrize == "and":
            # use mutual neighbors (i needs to be a neighbor of j AND vice versa)
            dmat = dmat.minimum(dmat.transpose()).tocoo()
        else:
            raise ValueError(f"Unknown symmetrization '{symmetrize}'")
    return dmat


# https://stackoverflow.com/a/38547818/4096483
def describe(df, count_zeros=True):
    if isinstance(df, np.ndarray):
        df = pd.Series(df)
    d = df.describe()

    aggs = [d, df.agg(["skew"])]
    if count_zeros:
        aggs.append(pd.Series(df[df == 0].count(), index=["#0"]))

    return pd.concat(aggs)


def get_nbrs(name, n_neighbors, metric="euclidean"):
    nbrs = read_neighbors(name, n_neighbors=n_neighbors + 1, exact=True, metric=metric)
    if nbrs is None:
        raise ValueError(
            f"Couldn't get exact {n_neighbors} neighbors"
            f" with metric '{metric}' for '{name}'"
        )
    nbrs.idx = nbrs.idx[:, 1:]
    nbrs.dist = nbrs.dist[:, 1:]
    return nbrs


def nbr_stats(name, n_neighbors, metric="euclidean"):
    nbrs = get_nbrs(name, n_neighbors, metric=metric)
    ko_desc, ko = ko_data(nbrs)
    so_desc, so = so_data(nbrs)
    nc = n_components(nbrs)
    data_info = get_dataset_info(name)
    return dict(
        name=name,
        n_dim=data_info["n_dim"],
        n_items=data_info["n_items"],
        idx_path=nbrs.info.idx_path,
        n_components=nc,
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


def read_nbr_stats(name, n_neighbors, metric="euclidean"):
    nbrs = get_nbrs(name, n_neighbors, metric=metric)
    # unlike the neighbor data itself, the stats file has to be an exact match
    # otherwise we won't find it
    if nbrs.info.n_nbrs != n_neighbors + 1:
        raise FileNotFoundError(f"Couldn't find nbr stats for {name}, {n_neighbors}")
    stats_path = idx_to_stats_path(nbrs.info.name, nbrs.info.idx_path)
    with open(stats_path, "rb") as f:
        return pickle.load(f)


def format_df(
    df,
    stats,
    drop_cols=None,
    rename_cols=None,
    n_neighbors_norm_cols=None,
    n_items_pct_cols=None,
    int_cols=None,
    float_cols=None,
):
    if drop_cols is not None:
        df = df.drop(columns=drop_cols)
    if rename_cols is not None:
        df = df.rename(columns=rename_cols)
    if n_neighbors_norm_cols is not None:
        for col_to_norm in n_neighbors_norm_cols:
            df[f"n{col_to_norm}"] = df[col_to_norm].astype(float) / stats["n_neighbors"]
    if n_items_pct_cols is not None:
        for col_to_norm in n_items_pct_cols:
            df[f"{col_to_norm}%"] = 100.0 * df[col_to_norm] / stats["n_items"]

    df[int_cols] = df[int_cols].astype(np.int32)
    # pylint: disable=consider-using-f-string
    df[float_cols] = df[float_cols].applymap("{0:.2f}".format)
    return df


def fetch_nbr_stats(name, n_neighbors, metric="euclidean", cache=True):
    try:
        return read_nbr_stats(name, n_neighbors, metric=metric)
    except FileNotFoundError:
        log.info(
            "Calculating neighbor stats for %s n_neighbors = %d", name, n_neighbors
        )
        stats = nbr_stats(name, n_neighbors, metric=metric)
        if cache:
            log.info("Caching neighbor stats")
            write_nbr_stats(stats)
        return stats


def nbr_stats_summary(n_neighbors, names=None, metric="euclidean", cache=True):
    if names is None:
        names = list_available_datasets()
    if not islisty(names):
        names = [names]

    summaries = []
    for name in names:
        stats_df = _nbr_stats_summary(name, n_neighbors, metric=metric, cache=cache)
        if not stats_df.empty:
            summaries.append(stats_df)

    return pd.concat(summaries)


def _nbr_stats_summary(name, n_neighbors, metric="euclidean", cache=True):
    try:
        stats = fetch_nbr_stats(name, n_neighbors, metric=metric, cache=cache)
    except ValueError:
        log.info("Skipping neighbor data for %s", name)
        return pd.DataFrame()

    kdf = pd.DataFrame(stats["ko_desc"]).T
    kdf = format_df(
        kdf,
        stats,
        drop_cols=["mean", "std", "25%", "75%"],
        rename_cols={
            "count": "n_items",
            "50%": "kmedian",
            "min": "kmin",
            "max": "kmax",
            "skew": "kskew",
            "#0": "k#0",
        },
        n_neighbors_norm_cols=["kmax", "kmedian"],
        n_items_pct_cols=["k#0"],
        int_cols=["n_items", "kmin", "kmax", "k#0"],
        float_cols=["nkmax", "nkmedian", "k#0%", "kskew"],
    )

    sdf = pd.DataFrame(stats["so_desc"]).T
    sdf = format_df(
        sdf,
        stats,
        drop_cols=["count", "std", "25%", "75%", "mean", "max"],
        rename_cols={
            "50%": "smedian",
            "min": "smin",
            "skew": "sskew",
            "#0": "s#0",
        },
        n_neighbors_norm_cols=["smedian"],
        n_items_pct_cols=["s#0"],
        int_cols=["smin", "s#0"],
        float_cols=["nsmedian", "s#0%", "sskew"],
    )

    df = pd.concat([kdf, sdf], axis=1)
    df.insert(1, "n_comps", stats["n_components"])
    df.insert(1, "n_dim", stats["n_dim"])
    return df
