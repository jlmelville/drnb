import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse.csgraph
import sklearn.metrics

import drnb.neighbors.sklearn as sknbrs
from drnb.io import data_relative_path, get_path, read_data, write_data
from drnb.log import log
from drnb.preprocess import numpyfy
from drnb.util import FromDict, Jsonizable, default_dict, default_list, islisty

from . import annoy, faiss, hnsw, pynndescent
from .nbrinfo import NbrInfo, NearestNeighbors


def n_connected_components(graph):
    return scipy.sparse.csgraph.connected_components(graph)[0]


def create_nn_func(method):
    if method == "sklearn":
        nn_func = sknbrs.sklearn_neighbors
        default_method_kwds = sknbrs.SKLEARN_DEFAULTS
    elif method == "faiss":
        nn_func = faiss.faiss_neighbors
        default_method_kwds = faiss.FAISS_DEFAULTS
    elif method == "pynndescent":
        nn_func = pynndescent.pynndescent_neighbors
        default_method_kwds = pynndescent.PYNNDESCENT_DEFAULTS
    elif method == "pynndescentbf":
        nn_func = pynndescent.pynndescent_exact_neighbors
        default_method_kwds = {}
    elif method == "hnsw":
        nn_func = hnsw.hnsw_neighbors
        default_method_kwds = hnsw.HNSW_DEFAULTS
    elif method == "annoy":
        nn_func = annoy.annoy_neighbors
        default_method_kwds = annoy.ANNOY_DEFAULTS
    else:
        raise ValueError(f"Unknown nearest neighbor method '{method}'")

    return nn_func, default_method_kwds


def calculate_neighbors(
    data,
    n_neighbors=15,
    metric="euclidean",
    method="approximate",
    return_distance=True,
    include_self=True,
    verbose=False,
    method_kwds=None,
    name=None,
):
    n_neighbors = int(n_neighbors)
    if not include_self:
        eff_n_neighbors = n_neighbors + 1
    else:
        eff_n_neighbors = n_neighbors

    n_items = data.shape[0]
    if n_items < eff_n_neighbors:
        log.warning(
            "%d neighbors requested but only %d items in the data",
            eff_n_neighbors,
            n_items,
        )
        eff_n_neighbors = n_items

    if method in ("exact", "approximate"):
        if method == "exact":
            method = find_exact_method(metric)
        else:
            method = find_fast_method(metric)
        if verbose:
            log.info("Using '%s' to find nearest neighbors", method)
    if verbose and method != "faiss" and (n_items > 10000 or data.shape[1] > 10000):
        log.warning("Exact nearest neighbors search with %s might take a while", method)

    nn_func, default_method_kwds = create_nn_func(method)

    # dict merge operator Python 3.9+ only
    if method_kwds is None:
        method_kwds = default_method_kwds
    else:
        method_kwds = default_method_kwds | method_kwds

    if verbose:
        log.info(
            f"Finding {eff_n_neighbors} neighbors using {method} "
            + f"with {metric} metric and params: {method_kwds}"
        )

    try:
        nn = nn_func(
            data,
            n_neighbors=eff_n_neighbors,
            metric=metric,
            return_distance=return_distance,
            **method_kwds,
        )
    except RuntimeError as e:
        if method == "faiss":
            log.warning(
                "faiss neighbors failed (out of memory?), falling back to pynndescent"
            )
            method = "pynndescent"
            nn_func = pynndescent.pynndescent_neighbors
            default_method_kwds = pynndescent.PYNNDESCENT_DEFAULTS
            nn = nn_func(
                data,
                n_neighbors=eff_n_neighbors,
                metric=metric,
                return_distance=return_distance,
                **method_kwds,
            )
        else:
            raise e

    if not include_self:
        # remove the first "self" item
        if return_distance:
            nn = (nn[0][:, 1:], nn[1][:, 1:])
        else:
            nn = nn[:, 1:]

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
        zip_algs(faiss.faiss_metrics(), "faiss")
        + zip_algs(pynndescent.PYNNDESCENT_METRICS.keys(), "pynndescent")
        + zip_algs(hnsw.HNSW_METRICS.keys(), "hnsw")
        + zip_algs(annoy.ANNOY_METRICS.keys(), "annoy")
    ):
        metric_algs[metric].append(method)

    return metric_algs


def preferred_exact_methods():
    metric_algs = defaultdict(list)
    for metric, method in (
        zip_algs(faiss.faiss_metrics(), "faiss")
        + zip_algs(sknbrs.SKLEARN_METRICS.keys(), "sklearn")
        + zip_algs(pynndescent.PYNNDESCENT_METRICS.keys(), "pynndescentbf")
    ):
        metric_algs[metric].append(method)

    return metric_algs


# pylint: disable=too-many-return-statements
def find_candidate_neighbors_info(
    name,
    drnb_home=None,
    sub_dir="nn",
    n_neighbors=1,
    metric="euclidean",
    method=None,
    exact=None,
    return_distance=True,
):
    if name is None:
        return None

    try:
        nn_dir_path = get_path(drnb_home=drnb_home, sub_dir=sub_dir)
    except FileNotFoundError:
        return None

    # probable nn files
    nn_file_paths = list(Path.glob(nn_dir_path, name + ".*.idx.*"))
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
    candidate_infos = sorted(nn_infos, key=lambda x: x.n_nbrs)
    candidate_info = candidate_infos[0]

    # favor npy or pkl files over csv
    preferred_exts = [".npy", ".pkl"]
    for cinfo in candidate_infos:
        if cinfo.idx_path.suffix in preferred_exts:
            return cinfo
    return candidate_info


def read_neighbors(
    name,
    n_neighbors=15,
    metric="euclidean",
    method=None,
    exact=False,
    drnb_home=None,
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
        drnb_home=drnb_home,
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
            drnb_home=drnb_home,
            sub_dir=sub_dir,
            as_numpy=True,
            verbose=False,
        )
        idx = idx[:, :n_neighbors]
        if return_distance:
            dist = read_data(
                dataset=candidate_info.name,
                suffix=candidate_info.dist_suffix,
                drnb_home=drnb_home,
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


def get_exact_neighbors(
    name,
    n_neighbors=15,
    metric="euclidean",
    return_distance=True,
    verbose=False,
    drnb_home=None,
    sub_dir="nn",
    cache=True,
    data=None,
    method_kwds=None,
):
    return get_neighbors(
        name=name,
        n_neighbors=n_neighbors,
        metric=metric,
        method="exact",
        return_distance=return_distance,
        verbose=verbose,
        drnb_home=drnb_home,
        sub_dir=sub_dir,
        cache=cache,
        data=data,
        method_kwds=method_kwds,
    )


def get_neighbors(
    name,
    n_neighbors=15,
    metric="euclidean",
    method="approximate",
    return_distance=True,
    verbose=False,
    # used only by read
    drnb_home=None,
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
        drnb_home=drnb_home,
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
                drnb_home=drnb_home,
                sub_dir=sub_dir,
                create_sub_dir=True,
                verbose=verbose,
            )
    return neighbor_data


def write_neighbors(
    neighbor_data,
    drnb_home=None,
    sub_dir="nn",
    create_sub_dir=True,
    file_type="npy",
    verbose=False,
):
    # e.g. mnist.150.euclidean.exact.faiss.dist.npy
    if neighbor_data.info.name is None:
        raise ValueError("No neighbor data info name")

    idx_paths = write_data(
        x=neighbor_data.idx,
        name=neighbor_data.info.name,
        suffix=neighbor_data.info.idx_suffix,
        drnb_home=drnb_home,
        sub_dir=sub_dir,
        create_sub_dir=create_sub_dir,
        verbose=verbose,
        file_type=file_type,
    )
    dist_paths = []
    if neighbor_data.dist is not None:
        dist_paths = write_data(
            x=neighbor_data.dist,
            name=neighbor_data.info.name,
            suffix=neighbor_data.info.dist_suffix,
            drnb_home=drnb_home,
            sub_dir=sub_dir,
            create_sub_dir=create_sub_dir,
            verbose=verbose,
            file_type=file_type,
        )
    return idx_paths, dist_paths


# Used in a pipeline
def get_neighbors_with_ctx(
    data, metric, n_neighbors, knn_params=None, ctx=None, return_distance=True
):
    if knn_params is None:
        knn_params = {}
    knn_defaults = dict(method="exact", cache=True, verbose=True, name=None)
    if ctx is not None:
        knn_defaults.update(
            dict(drnb_home=ctx.drnb_home, sub_dir=ctx.nn_sub_dir, name=ctx.dataset_name)
        )
    full_knn_params = knn_defaults | knn_params

    return get_neighbors(
        data=data,
        n_neighbors=n_neighbors,
        metric=metric,
        return_distance=return_distance,
        **full_knn_params,
    )


# this should only be used for creating reduced neighbor data for writing
def slice_neighbors(neighbors_data, n_neighbors):
    if neighbors_data.info.n_nbrs < n_neighbors:
        raise ValueError("Not enough neighbors")
    idx = neighbors_data.idx[:, :n_neighbors]
    dist = None
    if neighbors_data.dist is not None:
        dist = neighbors_data.dist[:, :n_neighbors]
    info = NbrInfo(**neighbors_data.info.__dict__)
    info.n_nbrs = n_neighbors
    return NearestNeighbors(idx=idx, dist=dist, info=info)


@dataclass
class NeighborsRequest(FromDict, Jsonizable):
    n_neighbors: list = default_list([15])
    method: str = "exact"
    metric: str = default_list(["euclidean"])
    file_types: list = default_list(["pkl"])
    params: dict = default_dict()
    verbose: bool = False

    def create_neighbors(self, data, dataset_name, nbr_dir="nn", suffix=None):
        if not self.n_neighbors:
            log.info("Neighbor request but no n_neighbors specified")
            return []
        max_n_neighbors = np.max(self.n_neighbors)

        nbrs_name = dataset_name
        if suffix is not None or suffix:
            nbrs_name = f"{nbrs_name}-{suffix}"

        if not islisty(self.metric):
            self.metric = [self.metric]
        for metric in self.metric:
            neighbors_data = calculate_neighbors(
                data=data,
                n_neighbors=max_n_neighbors,
                metric=metric,
                return_distance=True,
                verbose=self.verbose,
                name=nbrs_name,
                **self.params,
            )

            neighbors_output_paths = []
            for n_neighbors in self.n_neighbors:
                try:
                    sliced_neighbors = slice_neighbors(neighbors_data, n_neighbors)
                    idx_paths, dist_paths = write_neighbors(
                        neighbor_data=sliced_neighbors,
                        sub_dir=nbr_dir,
                        create_sub_dir=True,
                        file_type=self.file_types,
                        verbose=self.verbose,
                    )
                    neighbors_output_paths += idx_paths + dist_paths
                except ValueError:
                    log.warning(
                        "Unable to save %d neighbors (probably not enough neigbors)",
                        n_neighbors,
                    )
        return neighbors_output_paths


#   # for method = "exact" or "approximate" we can't know what algo we will get
#   # so need to nest the names inside?
#   method_kwds = dict("annoy"=dict(), hnsw=dict() ... )
def create_neighbors_request(neighbors_kwds):
    if neighbors_kwds is None:
        return None
    for key in ["metric", "n_neighbors"]:
        if key in neighbors_kwds and not islisty(neighbors_kwds[key]):
            neighbors_kwds[key] = [neighbors_kwds[key]]
    if "verbose" not in neighbors_kwds:
        neighbors_kwds["verbose"] = True
    neighbors_request = NeighborsRequest.new(**neighbors_kwds)
    log.info("Requesting one extra neighbor to account for self-neighbor")
    neighbors_request.n_neighbors = [
        n_nbrs + 1 for n_nbrs in neighbors_request.n_neighbors
    ]
    return neighbors_request
