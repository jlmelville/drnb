import itertools
from collections import defaultdict
from pathlib import Path

import sklearn.metrics

import drnb.neighbors.sklearn as sknbrs
from drnb.io import data_relative_path, get_data_path, numpyfy, read_data, write_npy
from drnb.log import log

from . import annoy, faiss, hnsw, pynndescent
from .nbrinfo import NbrInfo, NearestNeighbors


def calculate_neighbors(
    data,
    n_neighbors=15,
    metric="euclidean",
    method="approximate",
    return_distance=True,
    verbose=False,
    method_kwds=None,
    name=None,
):
    n_items = data.shape[0]
    if n_items < n_neighbors:
        log.warning(
            "%d neighbors requested but only %d items in the data", n_neighbors, n_items
        )
        n_neighbors = n_items

    if method in ("exact", "approximate"):
        if method == "exact":
            method = find_exact_method(metric)
        else:
            method = find_fast_method(metric)
    if verbose and method == "sklearn" and n_items > 10000 or data.shape[1] > 10000:
        log.warning(
            "Using sklearn to find exact nearest neighbors: this might take a while"
        )

    if method == "sklearn":
        nn_func = sknbrs.sklearn_neighbors
        default_method_kwds = sknbrs.SKLEARN_DEFAULTS
    elif method == "faiss":
        nn_func = faiss.faiss_neighbors
        default_method_kwds = faiss.FAISS_DEFAULTS
    elif method == "pynndescent":
        nn_func = pynndescent.pynndescent_neighbors
        default_method_kwds = pynndescent.PYNNDESCENT_DEFAULTS
    elif method == "hnsw":
        nn_func = hnsw.hnsw_neighbors
        default_method_kwds = hnsw.HNSW_DEFAULTS
    elif method == "annoy":
        nn_func = annoy.annoy_neighbors
        default_method_kwds = annoy.ANNOY_DEFAULTS
    else:
        raise ValueError(f"Unknown nearest neighbor method '{method}'")

    # dict merge operator Python 3.9+ only
    if method_kwds is None:
        method_kwds = default_method_kwds
    else:
        method_kwds = default_method_kwds | method_kwds

    if verbose:
        log.info(
            f"Finding {n_neighbors} neighbors using {method} "
            + f"with {metric} metric and params: {method_kwds}"
        )

    nn = nn_func(
        data,
        n_neighbors=n_neighbors,
        metric=metric,
        return_distance=return_distance,
        **method_kwds,
    )
    nn_info = NbrInfo(
        name=name,
        n_nbrs=n_neighbors,
        method=method,
        metric=metric,
        exact=is_exact_method(method),
        has_distances=return_distance,
        idx_path=None,
        dist_path=None,
    )
    if return_distance:
        return NearestNeighbors(idx=nn[0], dist=nn[1], info=nn_info)
    return NearestNeighbors(idx=nn, dist=None, info=nn_info)


def dmat(x):
    if isinstance(x, tuple) and len(x) == 2:
        x = x[0]
    x = numpyfy(x)
    return sklearn.metrics.pairwise_distances(x)


def zip_algs(metrics, method_name):
    return list(zip(metrics, itertools.cycle((method_name,))))


def is_exact_method(method):
    return method in ("faiss", "sklearn")


def find_exact_method(metric):
    return preferred_exact_methods()[metric][0]


def find_fast_method(metric):
    return preferred_fast_methods()[metric][0]


# For a given metric, return the suggested method for approximate nearest neighbors
# FAISS is included because it's fast
def preferred_fast_methods():
    metric_algs = defaultdict(list)
    for metric, method in (
        zip_algs(faiss.FAISS_METRICS, "faiss")
        + zip_algs(pynndescent.PYNNDESCENT_METRICS.keys(), "pynndescent")
        + zip_algs(hnsw.HNSW_METRICS.keys(), "hnsw")
        + zip_algs(annoy.ANNOY_METRICS.keys(), "annoy")
    ):
        metric_algs[metric].append(method)

    return metric_algs


def preferred_exact_methods():
    metric_algs = defaultdict(list)
    for metric, method in zip_algs(faiss.FAISS_METRICS, "faiss") + zip_algs(
        sknbrs.SKLEARN_METRICS.keys(), "sklearn"
    ):
        metric_algs[metric].append(method)

    return metric_algs


# pylint: disable=too-many-return-statements
def find_candidate_neighbors_info(
    name,
    data_path=None,
    sub_dir="nn",
    n_neighbors=1,
    metric="euclidean",
    method=None,
    exact=None,
    return_distance=True,
):
    if name is None:
        return None

    nn_dir_path = get_data_path(data_path=data_path, sub_dir=sub_dir)
    # probable nn files
    nn_file_paths = list(Path.glob(nn_dir_path, name + "*.idx.*"))
    if not nn_file_paths:
        return None

    # actual nn info
    nn_infos = [
        NbrInfo.from_path(nn_file_path, ignore_bad_path=True)
        for nn_file_path in nn_file_paths
        if nn_file_path.stem.startswith(name)
    ]
    nn_infos = [nn_info for nn_info in nn_infos if nn_info is not None]
    if not nn_infos:
        return None

    # filter on metric
    nn_infos = [nn_info for nn_info in nn_infos if nn_info.metric == metric]
    if not nn_infos:
        return None

    # If exact or approximate was specifically asked for (exact=None means "don't care")
    if exact is not None:
        if exact:
            nn_infos = [nn_info for nn_info in nn_infos if nn_info.exact]
        else:
            nn_infos = [nn_info for nn_info in nn_infos if not nn_info.exact]
        if not nn_infos:
            return None

    if method is not None:
        nn_infos = [nn_info for nn_info in nn_infos if nn_info.method == method]
        if not nn_infos:
            return None

    if return_distance:
        nn_infos = [nn_info for nn_info in nn_infos if nn_info.has_distances]
        if not nn_infos:
            return None

    # those nbr files with sufficient number of neighbors
    nn_infos = [nn_info for nn_info in nn_infos if nn_info.n_nbrs >= n_neighbors]
    if not nn_infos:
        return None

    # the smallest file which has enough neighbors
    candidate_info = sorted(nn_infos, key=lambda x: x.n_nbrs)[0]

    return candidate_info


def read_neighbors(
    name,
    n_neighbors=15,
    metric="euclidean",
    method=None,
    exact=False,
    data_path=None,
    sub_dir="nn",
    return_distance=True,
    verbose=False,
):
    candidate_info = find_candidate_neighbors_info(
        name,
        n_neighbors=n_neighbors,
        metric=metric,
        method=method,
        exact=exact,
        return_distance=return_distance,
        data_path=data_path,
        sub_dir=sub_dir,
    )

    if candidate_info is not None:
        if verbose:
            log.info(
                "Found pre-calculated neighbors file: %s",
                data_relative_path(candidate_info.idx_path),
            )
        idx = read_data(
            dataset=candidate_info.name,
            suffix=candidate_info.idx_suffix,
            data_path=data_path,
            sub_dir=sub_dir,
            as_numpy=True,
            verbose=False,
        )
        idx = idx[:, :n_neighbors]
        if return_distance:
            dist = read_data(
                dataset=candidate_info.name,
                suffix=candidate_info.dist_suffix,
                data_path=data_path,
                sub_dir=sub_dir,
                as_numpy=True,
                verbose=False,
            )
            dist = dist[:, :n_neighbors]
            return NearestNeighbors(idx=idx, dist=dist, info=candidate_info)
        return NearestNeighbors(idx=idx, dist=None, info=candidate_info)

    if verbose:
        log.info("No suitable pre-calculated neighbors available")
    return None


def get_neighbors(
    name,
    n_neighbors=15,
    metric="euclidean",
    method="approximate",
    return_distance=True,
    verbose=False,
    # used only by read
    data_path=None,
    sub_dir="nn",
    cache=True,
    # used only by calc
    data=None,
    method_kwds=None,
):
    if method is not None:
        if method == "exact":
            read_exact = True
            read_method = None
        elif method == "approximate":
            read_exact = False
            read_method = None
        else:
            read_exact = None
            read_method = method
    else:
        read_exact = None
        read_method = None

    neighbor_data = read_neighbors(
        name,
        n_neighbors=n_neighbors,
        metric=metric,
        method=read_method,
        exact=read_exact,
        data_path=data_path,
        sub_dir=sub_dir,
        return_distance=return_distance,
        verbose=verbose,
    )
    if neighbor_data is not None:
        return neighbor_data
    if data is None:
        raise ValueError("Must provide data to calculate neighbors")
    neighbor_data = calculate_neighbors(
        data,
        n_neighbors=n_neighbors,
        metric=metric,
        method=method,
        return_distance=return_distance or cache,
        verbose=verbose,
        method_kwds=method_kwds,
        name=name,
    )
    if cache:
        if name is None:
            log.warning("Asked for caching but no name provided to save under")
        else:
            if verbose:
                log.info("Caching calculated neighbor data")
            write_neighbors(
                neighbor_data,
                data_path=data_path,
                sub_dir=sub_dir,
                create_sub_dir=True,
                verbose=verbose,
            )
    return neighbor_data


def write_neighbors(
    neighbor_data,
    data_path=None,
    sub_dir="nn",
    create_sub_dir=True,
    verbose=False,
):
    # e.g. mnist.150.euclidean.exact.faiss.dist.npy
    write_npy(
        neighbor_data.idx,
        neighbor_data.info.name,
        suffix=neighbor_data.info.idx_suffix,
        data_path=data_path,
        sub_dir=sub_dir,
        create_sub_dir=create_sub_dir,
        verbose=verbose,
    )
    if neighbor_data.dist is not None:
        write_npy(
            neighbor_data.dist,
            neighbor_data.info.name,
            suffix=neighbor_data.info.dist_suffix,
            data_path=data_path,
            sub_dir=sub_dir,
            create_sub_dir=create_sub_dir,
            verbose=verbose,
        )


# Used in a pipeline
def get_neighbors_with_ctx(data, metric, n_neighbors, knn_params=None, ctx=None):
    if knn_params is None:
        knn_params = {}
    knn_defaults = dict(
        method="exact",
        cache=True,
        verbose=True,
    )
    if ctx is not None:
        knn_defaults.update(
            dict(data_path=ctx.data_path, sub_dir=ctx.nn_sub_dir, name=ctx.name)
        )
    full_knn_params = knn_defaults | knn_params

    return get_neighbors(
        data=data,
        n_neighbors=n_neighbors,
        metric=metric,
        return_distance=True,
        **full_knn_params,
    )
