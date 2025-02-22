import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import sklearn.metrics

import drnb.neighbors.sklearn as sknbrs
from drnb.io import data_relative_path, get_path, read_data, write_data
from drnb.log import log
from drnb.neighbors.nbrinfo import NearestNeighbors
from drnb.preprocess import numpyfy
from drnb.types import DataSet
from drnb.util import FromDict, Jsonizable, islisty

from . import annoy, faiss, hnsw, pynndescent
from .nbrinfo import NbrInfo


def n_connected_components(graph: scipy.sparse.coo_matrix) -> int:
    """Return the number of connected components in a graph."""
    return scipy.sparse.csgraph.connected_components(graph)[0]


def create_nn_func(method: str) -> Tuple[Callable, Dict]:
    """Create a nearest neighbor function and default keyword arguments."""
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
    data: np.ndarray,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    method: str = "approximate",
    return_distance: bool = True,
    include_self: bool = True,
    verbose: bool = False,
    method_kwds: dict | None = None,
    name: str = "",
) -> NearestNeighbors:
    """Calculate nearest neighbors for a given data set, using a specified method.
    By default, approximate nearest neighbors are found. For exact nearest neighbors,
    use method='exact'."""
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
            method = _find_exact_method(metric)
        else:
            method = _find_fast_method(metric)
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
        exact=_is_exact_method(method),
        has_distances=return_distance,
        idx_path=None,
        dist_path=None,
    )
    if return_distance:
        return NearestNeighbors(idx=nn[0], dist=nn[1], info=nn_info)
    return NearestNeighbors(idx=nn, dist=None, info=nn_info)


def dmat(x: DataSet | np.ndarray) -> np.ndarray:
    """Calculate the pairwise distance matrix for a given data set."""
    if isinstance(x, tuple) and len(x) == 2:
        x = x[0]
    x = numpyfy(x)
    return sklearn.metrics.pairwise_distances(x)


def _zip_algs(metrics, method_name):
    return list(zip(metrics, itertools.cycle((method_name,))))


def _is_exact_method(method):
    return method in ("faiss", "sklearn")


def _find_exact_method(metric):
    return _preferred_exact_methods()[metric][0]


def _find_fast_method(metric):
    return _preferred_fast_methods()[metric][0]


# For a given metric, return the suggested method for approximate nearest neighbors
# FAISS is included because it's fast
def _preferred_fast_methods():
    metric_algs = defaultdict(list)
    for metric, method in (
        _zip_algs(faiss.faiss_metrics(), "faiss")
        + _zip_algs(pynndescent.PYNNDESCENT_METRICS.keys(), "pynndescent")
        + _zip_algs(hnsw.HNSW_METRICS.keys(), "hnsw")
        + _zip_algs(annoy.ANNOY_METRICS.keys(), "annoy")
    ):
        metric_algs[metric].append(method)

    return metric_algs


def _preferred_exact_methods():
    metric_algs = defaultdict(list)
    for metric, method in (
        _zip_algs(faiss.faiss_metrics(), "faiss")
        + _zip_algs(sknbrs.SKLEARN_METRICS.keys(), "sklearn")
        + _zip_algs(pynndescent.PYNNDESCENT_METRICS.keys(), "pynndescentbf")
    ):
        metric_algs[metric].append(method)

    return metric_algs


# pylint: disable=too-many-return-statements
def find_candidate_neighbors_info(
    name: str | None = None,
    drnb_home: Path | str | None = None,
    sub_dir: str = "nn",
    n_neighbors: int = 1,
    metric: str = "euclidean",
    method: str | None = None,
    exact: bool | None = None,
    return_distance: bool = True,
    verbose: bool = False,
) -> NbrInfo | None:
    """Find the most suitable pre-calculated nearest neighbors info for a dataset.

    This function searches for pre-calculated nearest neighbor files that match the
    specified criteria, using a series of filtering steps:

    1. Finds all neighbor files matching the dataset name
    2. Filters for the specified metric
    3. Filters for exact/approximate calculation method if specified
    4. Filters for specific neighbor calculation method if specified
    5. Filters for presence of distance information if required
    6. Filters for sufficient number of neighbors
    7. Returns the candidate with the smallest number of neighbors that meets all criteria

    Args:
        name: Dataset name to search for. If None, returns None.
        drnb_home: Root directory for data files. If None, uses default.
        sub_dir: Subdirectory containing neighbor files (default: "nn").
        n_neighbors: Minimum number of neighbors required (default: 1).
        metric: Distance metric to match (default: "euclidean").
        method: Specific neighbor calculation method to match (e.g., "faiss", "sklearn").
            If None, accepts any method.
        exact: Filter by exact/approximate calculation:
        - True: only exact methods
        - False: only approximate methods
        - None: accept either
        return_distance: If True, only accept files with distance information.
        verbose: If True, log detailed information about the search process.

    Returns:
        NbrInfo | None: Information about the most suitable neighbor file found, or None if:
        - No name provided
        - No neighbor directory found
        - No neighbor files found for dataset
        - No files match the specified metric
        - No files match exact/approximate preference
        - No files match the specified method
        - No files have distance information (if required)
        - No files have sufficient neighbors

    Note:
        When multiple suitable files exist, prefers .npy or .pkl formats over .csv
        for better performance.
    """
    if name is None:
        if verbose:
            log.warning("No name provided to find candidate neighbors info")
        return None

    try:
        nn_dir_path = get_path(drnb_home=drnb_home, sub_dir=sub_dir)
    except FileNotFoundError as e:
        if verbose:
            log.warning("No neighbors directory found at %s", e.filename)
        return None

    # probable nn files
    nn_file_paths = list(Path.glob(nn_dir_path, name + ".*.idx.*"))
    if not nn_file_paths:
        if verbose:
            log.warning("No neighbors files found for %s", name)
        return None

    # actual nn info
    nn_infos = [
        NbrInfo.from_path(nn_file_path, ignore_bad_path=True)
        for nn_file_path in nn_file_paths
        if nn_file_path.stem.startswith(name)
    ]
    nn_infos = [nn_info for nn_info in nn_infos if nn_info is not None]
    if not nn_infos:
        if verbose:
            log.warning("No neighbors info found for %s", name)
        return None

    # filter on metric
    nn_infos = [nn_info for nn_info in nn_infos if nn_info.metric == metric]
    if not nn_infos:
        if verbose:
            log.warning("No neighbors info found for %s with metric %s", name, metric)
        return None

    # If exact or approximate was specifically asked for (exact=None means "don't care")
    if exact is not None:
        if exact:
            nn_infos = [nn_info for nn_info in nn_infos if nn_info.exact]
        else:
            nn_infos = [nn_info for nn_info in nn_infos if not nn_info.exact]
        if not nn_infos:
            if verbose:
                log.warning("No neighbors info found for %s with exact=%s", name, exact)
            return None

    if method is not None:
        nn_infos = [nn_info for nn_info in nn_infos if nn_info.method == method]
        if not nn_infos:
            if verbose:
                log.warning(
                    "No neighbors info found for %s with method %s", name, method
                )
            return None

    if return_distance:
        nn_infos = [nn_info for nn_info in nn_infos if nn_info.has_distances]
        if not nn_infos:
            if verbose:
                log.warning(
                    "No neighbors info found for %s with return_distance=%s",
                    name,
                    return_distance,
                )
            return None

    # those nbr files with sufficient number of neighbors
    nn_infos = [nn_info for nn_info in nn_infos if nn_info.n_nbrs >= n_neighbors]
    if not nn_infos:
        if verbose:
            log.warning(
                "No neighbors info found for %s with n_neighbors=%d", name, n_neighbors
            )
        return None

    nn_infos = cast(List[NbrInfo], nn_infos)
    # the smallest file which has enough neighbors
    candidate_infos = sorted(nn_infos, key=lambda x: x.n_nbrs)
    candidate_info = candidate_infos[0]

    # favor npy or pkl files over csv
    preferred_exts = [".npy", ".pkl"]
    for cinfo in candidate_infos:
        if cinfo.idx_path is not None and cinfo.idx_path.suffix in preferred_exts:
            if verbose:
                log.info("Found pre-calculated neighbors file: %s", cinfo.idx_path)
            return cinfo
    if verbose:
        log.info("No suitable pre-calculated neighbors available")
    return candidate_info


def read_neighbors(
    name: str,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    method: str = None,
    exact: bool | None = False,
    drnb_home: Path | str | None = None,
    sub_dir: str = "nn",
    return_distance: bool = True,
    verbose: bool = False,
) -> NearestNeighbors | None:
    """Read pre-calculated nearest neighbors from disk. If no suitable neighbors are
    found, return None. If `exact` is True, only exact neighbors are considered. If
    `exact` is False, only approximate neighbors are considered. If `exact` is None,
    both exact and approximate neighbors are considered. If `method` is not None, only
    neighbors calculated with that method are considered. If `return_distance` is True,
    only neighbors with distances are considered."""
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
    name: str | None = None,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    return_distance: bool = True,
    verbose: bool = False,
    drnb_home: Path | str | None = None,
    sub_dir: str = "nn",
    cache: bool = True,
    data: np.ndarray | None = None,
    method_kwds: dict | None = None,
) -> NearestNeighbors:
    """Get exact nearest neighbors for a given dataset by reading pre-calculated results
    or calculating directly. One or both of `name` and `data` must be provided,
    depending on the desired behavior. If `name` is provided, try to read the
    pre-calculated neighbors. Otherwise, calculate them using `data`. If `cache` is True,
    save the calculated neighbors to disk using `name`. If `return_distance` is True,
    return both the indices and the distances to the neighbors. Otherwise, return only
    the indices."""
    if name is None or not name:
        cache = False

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


def calculate_exact_neighbors(
    data: np.ndarray | None = None,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    return_distance: bool = True,
    include_self: bool = True,
    verbose: bool = False,
    method_kwds: Optional[dict] = None,
    name: str = "",
) -> NearestNeighbors:
    """Calculate exact nearest neighbors for a given dataset."""
    return calculate_neighbors(
        data,
        n_neighbors=n_neighbors,
        metric=metric,
        method="exact",
        return_distance=return_distance,
        include_self=include_self,
        verbose=verbose,
        method_kwds=method_kwds,
        name=name,
    )


def get_neighbors(
    name: str | None = None,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    method: str = "approximate",
    return_distance: bool = True,
    verbose: bool = False,
    # used by read/cache
    drnb_home: Path | str | None = None,
    sub_dir: str = "nn",
    cache: bool = True,
    # used only by calc
    data: np.ndarray | None = None,
    method_kwds: dict | None = None,
) -> NearestNeighbors:
    """Get nearest neighbors for a given dataset by reading pre-calculated results or
    calculating directly. One or both of `name` and `data` must be provided, depending
    on the desired behavior. If `name` is provided, try to read the pre-calculated
    neighbors. Otherwise, calculate them using `data`. If `cache` is True, save the
    calculated neighbors to disk using `name`. If `return_distance` is True, return
    both the indices and the distances to the neighbors. Otherwise, return only the
    indices."""
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
        if name is None or not name:
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
    neighbor_data: NearestNeighbors,
    drnb_home: Path | str | None = None,
    sub_dir: str = "nn",
    create_sub_dir: bool = True,
    file_type: str | List[str] = "npy",
    verbose: bool = False,
) -> Tuple[List[Path], List[Path]]:
    """Write nearest neighbors data to disk."""
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


# this should only be used for creating reduced neighbor data for writing
def _slice_neighbors(neighbors_data, n_neighbors):
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
    """Request for creating neighbors.

    Attributes:
        n_neighbors: Number of neighbors to calculate.
        method: Method to use for calculating neighbors.
        metric: Metric to use for calculating neighbors.
        file_types: File types to save neighbors as.
        params: Additional parameters for calculating neighbors.
        verbose: Whether to output verbose logging
    """

    n_neighbors: List[int] = field(default_factory=lambda: [15])
    method: str = "exact"
    metric: str | List[str] = field(default_factory=lambda: ["euclidean"])
    file_types: List[str] = field(default_factory=lambda: ["pkl"])
    params: Dict = field(default_factory=dict)
    verbose: bool = False

    def create_neighbors(
        self,
        data: np.ndarray,
        dataset_name: str,
        nbr_dir: str = "nn",
        suffix: str | None = None,
    ) -> List[Path]:
        """Create neighbors for a given dataset."""
        if not self.n_neighbors:
            log.info("Neighbor request but no n_neighbors specified")
            return []
        max_n_neighbors = np.max(self.n_neighbors)

        nbrs_name = dataset_name
        if suffix is not None or suffix:
            nbrs_name = f"{nbrs_name}-{suffix}"

        if not islisty(self.metric):
            self.metric = [self.metric]
        self.metric = cast(List[str], self.metric)
        neighbors_output_paths = []
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

            for n_neighbors in self.n_neighbors:
                try:
                    sliced_neighbors = _slice_neighbors(neighbors_data, n_neighbors)
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
def create_neighbors_request(neighbors_kwds: dict | None) -> NeighborsRequest | None:
    """Create a neighbors request from a dictionary of keyword arguments."""
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
