import pickle
from pathlib import Path
from typing import List, Literal, Tuple, cast

import numpy as np
import pandas as pd
import scipy.sparse
from numba import jit, prange
from scipy.sparse.csgraph import connected_components

from drnb.dimension import mle_global
from drnb.io import data_relative_path
from drnb.io.dataset import get_dataset_info, list_available_datasets
from drnb.log import log
from drnb.neighbors import NearestNeighbors, read_neighbors
from drnb.neighbors.localscale import locally_scaled_neighbors
from drnb.neighbors.nbrinfo import replace_n_neighbors_in_path
from drnb.neighbors.random import random_sample_nbrs
from drnb.util import islisty


def k_occurrences(idx: np.ndarray, n_neighbors: int | None = None) -> np.ndarray:
    """Count occurrences of items in the k-nearest neighbors of all items (i.e. the
    size of the reverse nearest neighbors list)."""
    if n_neighbors is None:
        n_neighbors = idx.shape[1]
    if n_neighbors > idx.shape[1]:
        raise ValueError(f"{n_neighbors} > {idx.shape[1]}")

    return _k_occurrences(idx, n_neighbors)


@jit(nopython=True)
def _k_occurrences(idx: np.ndarray, n_neighbors: int) -> np.ndarray:
    n_items = idx.shape[0]
    result = np.zeros(dtype=np.int32, shape=n_items)

    for i in range(n_items):
        for j in range(n_neighbors):
            result[idx[i][j]] += 1

    return result


def s_occurrences(idx: np.ndarray, n_neighbors: int | None = None) -> np.ndarray:
    """Count occurrences of shared neighbors of all items (i.e. the size of the mutual
    nearest neighbor lists)."""
    if n_neighbors is None:
        n_neighbors = idx.shape[1]
    if n_neighbors > idx.shape[1]:
        raise ValueError(f"{n_neighbors} > {idx.shape[1]}")

    idx = idx[:, :n_neighbors]
    return _s_occurrences(idx)


@jit(nopython=True, parallel=True)
def _s_occurrences(idx: np.ndarray) -> np.ndarray:
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


def n_components(
    nbrs: np.ndarray | Tuple | NearestNeighbors,
) -> int:
    """Calculate the number of connected components in the nearest neighbor graph."""
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
# "mean" to carry out mean operation (average of matrix and its transpose)
def nn_to_sparse(
    nbrs: np.ndarray | Tuple | NearestNeighbors,
    symmetrize: Literal["or", "and", "mean"] | None = None,
) -> scipy.sparse.coo_matrix:
    """Convert nearest neighbors to a sparse matrix. Optionally symmetrize the matrix.
    Symmetrization can be "or" (undirected), "and" (mutual), or "mean" (average).
    """
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
        elif symmetrize == "mean":
            rows, cols = dmat.row, dmat.col
            values = dmat.data

            # Concatenating the (row, col, value) with (col, row, value)
            all_rows = np.concatenate([rows, cols])
            all_cols = np.concatenate([cols, rows])
            all_values = np.concatenate([values, values])

            # Create a coordinate matrix for all these values
            new_coo = scipy.sparse.coo_matrix(
                (all_values, (all_rows, all_cols)), shape=dmat.shape
            )

            # Find duplicates and average the values
            rows, cols = np.unique(
                np.vstack([new_coo.row, new_coo.col]), axis=1, return_inverse=True
            )
            split_cols = np.split(cols, 2)
            rows = split_cols[0]
            cols = split_cols[1]
            sum_values = np.bincount(rows * dmat.shape[1] + cols, weights=new_coo.data)
            count_values = np.bincount(rows * dmat.shape[1] + cols)
            mean_values = sum_values / count_values

            # Create the new symmetrized COO matrix
            dmat = scipy.sparse.coo_matrix(
                (mean_values, (rows, cols)), shape=dmat.shape
            )
        else:
            raise ValueError(f"Unknown symmetrization '{symmetrize}'")
    return dmat


# https://stackoverflow.com/a/38547818/4096483
def describe(df: pd.Series | np.ndarray, count_zeros=True) -> pd.Series:
    """Describe a pandas Series or numpy array, including skewness and count of zeros."""
    if isinstance(df, np.ndarray):
        df = pd.Series(df)
    d = df.describe()

    aggs = [d, df.agg(["skew"])]
    if count_zeros:
        aggs.append(pd.Series(df[df == 0].count(), index=["#0"]))

    return pd.concat(aggs)


def get_nbrs(
    name: str, n_neighbors: int, metric: str = "euclidean"
) -> NearestNeighbors:
    """Retrieve exact nearest neighbors for a dataset.

    Gets the k-nearest neighbors for a dataset, excluding self-neighbors. The actual
    number of neighbors retrieved is n_neighbors + 1 to account for self-neighbors
    which are then removed.

    Args:
        name: Name of the dataset to retrieve neighbors for.
        n_neighbors: Number of neighbors to retrieve (excluding self).
        metric: Distance metric to use (default: "euclidean").

    Returns:
        NearestNeighbors: Object containing neighbor indices and distances.

    Raises:
        ValueError: If exact neighbors cannot be retrieved for the given parameters.
    """
    nbrs = read_neighbors(
        name,
        n_neighbors=n_neighbors + 1,
        exact=True,
        metric=metric,
        return_distance=True,
    )
    if nbrs is None:
        log.warning(
            "Couldn't get exact %d neighbors with metric '%s' for '%s'",
            n_neighbors,
            metric,
            name,
        )
        raise ValueError(
            f"Couldn't get exact {n_neighbors} neighbors"
            f" with metric '{metric}' for '{name}'"
        )
    nbrs = cast(NearestNeighbors, nbrs)
    nbrs.idx = cast(np.ndarray, nbrs.idx)
    nbrs.dist = cast(np.ndarray, nbrs.dist)
    nbrs.idx = nbrs.idx[:, 1:]
    nbrs.dist = nbrs.dist[:, 1:]
    return nbrs


def calculate_nbr_stats(
    name: str,
    n_neighbors: int,
    metric: str = "euclidean",
    transform: Literal["local", "random"] | None = None,
) -> dict:
    """Calculate comprehensive neighbor statistics for a dataset.

    Computes various statistics about the neighborhood structure including:
    - k-occurrences (reverse nearest neighbor counts)
    - s-occurrences (mutual nearest neighbor counts)
    - number of connected components
    - intrinsic dimensionality estimate

    Args:
        name: Name of the dataset to analyze.
        n_neighbors: Number of neighbors to use in calculations.
        metric: Distance metric used for neighbor calculations (default: "euclidean").
        transform: Optionally, transform the neighbors by random sampling (random) or
            local scaling (local). Note that both methods will increase the number of
            neighbors to n_neighbors + 50.

    Returns:
        dict: Dictionary containing:
            - name: Dataset name
            - n_dim: Original dimensionality
            - n_items: Number of items in dataset
            - idx_path: Path to the index file
            - n_components: Number of connected components
            - dint: Intrinsic dimensionality estimate
            - n_neighbors: Number of neighbors used
            - ko_desc: k-occurrences statistics
            - so_desc: s-occurrences statistics
            - ko: Raw k-occurrences data
            - so: Raw s-occurrences data

    Note:
        Returns an empty dict if neighbors cannot be retrieved (e.g., if n_neighbors
        is too high for the dataset).
    """
    try:
        if transform == "local" or transform == "random":
            extended_n_neighbors = n_neighbors + 50
        else:
            extended_n_neighbors = n_neighbors

        nbrs = get_nbrs(name, extended_n_neighbors, metric=metric)
    except ValueError:
        # nbrs may be None if n_neighbors is too high for the dataset
        return {}

    if transform == "local":
        nbrs.idx, nbrs.dist = locally_scaled_neighbors(
            nbrs.idx, nbrs.dist, l=n_neighbors, m=extended_n_neighbors
        )
    elif transform == "random":
        info = nbrs.info
        nbrs = random_sample_nbrs(nbrs.idx, nbrs.dist, n_neighbors)
        nbrs.info = info

    ko_desc, ko = ko_data(nbrs)
    so_desc, so = so_data(nbrs)
    nc = n_components(nbrs)
    mle_dint = mle_global(nbrs.dist, remove_self=True)
    data_info = get_dataset_info(name)

    if nbrs.info is not None and nbrs.info.idx_path is not None:
        idx_path = nbrs.info.idx_path
        idx_path = replace_n_neighbors_in_path(idx_path, n_neighbors)
    else:
        idx_path = None

    return {
        "name": name,
        "n_dim": data_info["n_dim"],
        "n_items": data_info["n_items"],
        "idx_path": idx_path,
        "n_components": nc,
        "dint": mle_dint,
        "n_neighbors": n_neighbors,
        "ko_desc": ko_desc,
        "so_desc": so_desc,
        "ko": ko,
        "so": so,
    }


def ko_data(
    nbrs: np.ndarray | Tuple | NearestNeighbors, n_neighbors: int | None = None
) -> Tuple[pd.Series, np.ndarray]:
    """Get k-occurrences data for a dataset. Returns a pandas Series with the
    statistics and a numpy array with the actual data."""
    if n_neighbors is None:
        n_neighbors = nbrs.idx.shape[1]
    ko = k_occurrences(nbrs.idx, n_neighbors=n_neighbors)
    ko_desc = describe(ko)
    ko_desc.name = nbrs.info.name
    return ko_desc, ko


def so_data(
    nbrs: np.ndarray | Tuple | NearestNeighbors, n_neighbors: int | None = None
) -> Tuple[pd.Series, np.ndarray]:
    """Get s-occurrences data for a dataset. Returns a pandas Series with the
    statistics and a numpy array with the actual data."""
    if n_neighbors is None:
        n_neighbors = nbrs.idx.shape[1]
    so = s_occurrences(nbrs.idx, n_neighbors=n_neighbors)
    so_desc = describe(so)
    so_desc.name = nbrs.info.name
    return so_desc, so


def idx_to_stats_path(name: str, idx_path: Path) -> Path:
    """Get the path to the neighbor stats file for a dataset, given the name and the
    path to the index file."""
    return idx_path.parent / "".join(
        [name] + idx_path.suffixes[:-2] + [".stats", ".npy"]
    )


def write_nbr_stats(nbr_stats: dict, overwrite: bool = False):
    """Write neighbor statistics to a file."""
    idx_path = nbr_stats["idx_path"]
    stats_path = idx_to_stats_path(nbr_stats["name"], idx_path)
    if not stats_path.exists() or overwrite:
        log.info("Writing pkl format to %s", data_relative_path(stats_path))
        with open(stats_path, "wb") as f:
            pickle.dump(nbr_stats, f, pickle.HIGHEST_PROTOCOL)


def read_nbr_stats(name: str, n_neighbors: int, metric: str = "euclidean") -> dict:
    """Read neighbor statistics from a file."""
    nbrs = get_nbrs(name, n_neighbors, metric=metric)
    if nbrs.info is None:
        raise ValueError("Neighborhood data has no file info")
    # unlike the neighbor data itself, the stats file has to be an exact match
    # otherwise we won't find it
    if nbrs.info.n_nbrs != n_neighbors + 1:
        raise FileNotFoundError(f"Couldn't find nbr stats for {name}, {n_neighbors}")
    stats_path = idx_to_stats_path(nbrs.info.name, nbrs.info.idx_path)
    with open(stats_path, "rb") as f:
        return pickle.load(f)


def format_df(
    df: pd.DataFrame,
    nbr_stats: dict,
    drop_cols: List[str] | None = None,
    rename_cols: List[str] | None = None,
    n_neighbors_norm_cols: List[str] | None = None,
    n_items_pct_cols: List[str] | None = None,
    int_cols: List[str] | None = None,
    float_cols: List[str] | None = None,
) -> pd.DataFrame:
    """Format and normalized a DataFrame containing neighbor statistics

    - drop_cols: columns to drop
    - rename_cols: columns to rename
    - n_neighbors_norm_cols: columns to normalize by n_neighbors
    - n_items_pct_cols: columns to normalize by n_items
    - int_cols: columns to convert to int32
    - float_cols: columns to format as float with 2 decimal places
    """
    if drop_cols is not None:
        df = df.drop(columns=drop_cols)
    if rename_cols is not None:
        df = df.rename(columns=rename_cols)
    if n_neighbors_norm_cols is not None:
        for col_to_norm in n_neighbors_norm_cols:
            df[f"n{col_to_norm}"] = (
                df[col_to_norm].astype(float) / nbr_stats["n_neighbors"]
            )
    if n_items_pct_cols is not None:
        for col_to_norm in n_items_pct_cols:
            df[f"{col_to_norm}%"] = 100.0 * df[col_to_norm] / nbr_stats["n_items"]

    df[int_cols] = df[int_cols].astype(np.int32)
    # pylint: disable=consider-using-f-string
    df[float_cols] = df[float_cols].map("{0:.2f}".format)
    return df


def fetch_nbr_stats(
    name: str,
    n_neighbors: int,
    metric: str = "euclidean",
    transform: Literal["local", "random"] | None = None,
    cache: bool = True,
    verbose: bool = False,
) -> dict:
    """Fetch or calculate neighbor statistics for a dataset with caching support.

    Attempts to load pre-computed neighbor statistics from cache. If not found,
    calculates the statistics and optionally caches them for future use.

    Args:
        name: Name of the dataset to analyze.
        n_neighbors: Number of neighbors to use in calculations.
        metric: Distance metric to use for neighbor calculations (default: "euclidean").
        transform: Optionally, transform the neighbors by random sampling (random) or
            local scaling (local). Note that both methods will increase the number of
            neighbors to n_neighbors + 50.
        cache: Whether to cache computed statistics (default: True).
        verbose: Whether to log verbose messages (default: False).
    Returns:
        dict: Dictionary containing neighbor statistics including:
        - name: Dataset name
        - n_dim: Original dimensionality
        - n_items: Number of items
        - idx_path: Path to index file
        - n_components: Number of connected components
        - dint: Intrinsic dimensionality estimate
        - n_neighbors: Number of neighbors used
        - ko_desc: k-occurrences statistics
        - so_desc: s-occurrences statistics
        - ko: Raw k-occurrences data
        - so: Raw s-occurrences data

    Note:
        Returns an empty dict if n_neighbors is too high for the dataset.
    """
    if transform is None:
        try:
            return read_nbr_stats(name, n_neighbors, metric=metric)
        except FileNotFoundError:
            pass

    if verbose:
        if transform:
            transform_str = f" ({transform=})"
        else:
            transform_str = ""
        log.info(
            "Calculating neighbor stats for %s n_neighbors = %d%s",
            name,
            n_neighbors,
            transform_str,
        )
    nbr_stats = calculate_nbr_stats(
        name, n_neighbors, metric=metric, transform=transform
    )
    if nbr_stats and cache and transform is None:
        # only cache the original neighbor info not the transformed neighbors
        # stats may be empty if n_neighbors is too high for the dataset
        if verbose:
            log.info(
                "Caching neighbor stats for %s n_neighbors = %d if needed",
                name,
                n_neighbors,
            )
        write_nbr_stats(nbr_stats)
    return nbr_stats


def nbr_stats_summary(
    n_neighbors: int,
    names: List[str] | str | None = None,
    metric: str = "euclidean",
    transform: Literal["local", "random"] | None = None,
    cache: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Generate a summary DataFrame of neighbor statistics for multiple datasets.

    Creates a comprehensive summary of neighborhood statistics including k-occurrences,
    s-occurrences, intrinsic dimensionality, and connected components for one or more
    datasets.

    Args:
        n_neighbors: Number of neighbors to use in calculations.

        names: Dataset name(s) to analyze. Can be:

        - None: analyzes all available datasets
        - str: analyzes a single dataset
        - List[str]: analyzes multiple specified datasets

        metric: Distance metric to use (default: "euclidean").
        cache: Whether to use/update cached statistics (default: True).

    Returns:
        pd.DataFrame: Summary statistics with columns including:
        - n_dim: Original dimensionality
        - dint: Intrinsic dimensionality estimate
        - n_comps: Number of connected components
        - n_items: Number of items
        - kmin/kmax/kmedian: k-occurrence statistics
        - smin/smedian: s-occurrence statistics
        - k#0/s#0: Count of zero occurrences
        - kskew/sskew: Skewness measures

        Additional normalized columns (prefixed with 'n') and
        percentage columns (suffixed with '%') are also included.

    Note:
        Skips datasets where neighbor calculations fail (e.g., if n_neighbors
        is too high) and returns an empty DataFrame if no valid results are found.
    """
    if names is None:
        names = list_available_datasets()
    if not islisty(names):
        names = [names]

    summaries = []
    for name in names:
        stats_df = _nbr_stats_summary(
            name,
            n_neighbors,
            metric=metric,
            transform=transform,
            cache=cache,
            verbose=verbose,
        )
        if not stats_df.empty:
            summaries.append(stats_df)
    if not summaries:
        if verbose:
            log.warning("No neighbor stats found for %s", names)
        return pd.DataFrame()
    return pd.concat(summaries)


def _nbr_stats_summary(
    name: str,
    n_neighbors: int,
    metric: str = "euclidean",
    transform: Literal["local", "random"] | None = None,
    cache: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    try:
        stats = fetch_nbr_stats(
            name,
            n_neighbors,
            metric=metric,
            cache=cache,
            transform=transform,
            verbose=verbose,
        )
    except ValueError:
        if verbose:
            log.info("Skipping neighbor data for %s", name)
        return pd.DataFrame()

    if not stats:
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
    df.insert(1, "dint", f"{stats['dint']:.2f}")
    df.insert(1, "n_comps", stats["n_components"])
    df.insert(1, "n_dim", stats["n_dim"])
    return df
