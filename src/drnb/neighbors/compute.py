from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, cast

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import sklearn.metrics
from drnb_nn_plugin_sdk import env_flag

from drnb.log import log
from drnb.neighbors.nbrinfo import NbrInfo, NearestNeighbors
from drnb.neighbors.store import read_neighbors, write_neighbors
from drnb.preprocess import numpyfy
from drnb.types import DataSet
from drnb.util import FromDict


def n_connected_components(graph: scipy.sparse.coo_matrix) -> int:
    """Return the number of connected components in a graph."""
    return scipy.sparse.csgraph.connected_components(graph)[0]


NN_PLUGIN_DEFAULTS: dict[str, dict[str, int | bool]] = {
    "annoy": {"n_trees": 50, "search_k": -1, "random_state": 42, "n_jobs": -1},
    "hnsw": {"M": 16, "ef_construction": 200, "random_state": 42, "n_jobs": -1},
    "faiss": {"use_gpu": True},
    "torchknn": {},
}

_ANNOY_METRICS = ("dot", "cosine", "manhattan", "euclidean")
_HNSW_METRICS = ("cosine", "dot", "euclidean", "l2")
_FAISS_METRICS = ("cosine", "euclidean")
_TORCHKNN_METRICS = ("euclidean", "cosine")


def dmat(x: DataSet | np.ndarray) -> np.ndarray:
    """Calculate the pairwise distance matrix for a given data set."""
    if isinstance(x, tuple) and len(x) == 2:
        x = x[0]
    x = numpyfy(x)
    return sklearn.metrics.pairwise_distances(x)


def create_nn_func(method: str) -> tuple[Callable, dict]:
    """Create a nearest neighbor function and default keyword arguments."""
    if method == "sklearn":
        from . import sklearn as sknbrs

        nn_func = sknbrs.sklearn_neighbors
        default_method_kwds = sknbrs.SKLEARN_DEFAULTS
    elif method == "pynndescent":
        from . import pynndescent as pynndescent_mod

        nn_func = pynndescent_mod.pynndescent_neighbors
        default_method_kwds = pynndescent_mod.PYNNDESCENT_DEFAULTS
    elif method == "pynndescentbf":
        from . import pynndescent as pynndescent_mod

        nn_func = pynndescent_mod.pynndescent_exact_neighbors
        default_method_kwds = {}
    else:
        raise ValueError(f"Unknown nearest neighbor method '{method}'")

    return nn_func, default_method_kwds


def _lookup_nn_plugin(method: str):
    """Return plugin spec if a neighbor plugin is registered for the method."""
    try:
        from drnb.nnplugins.registry import get_registry

        return get_registry().lookup(method)
    except (FileNotFoundError, NotADirectoryError):
        return None


def _plugin_available(method: str) -> bool:
    return _lookup_nn_plugin(method) is not None


def _plugin_default_params(method: str) -> dict:
    if method not in NN_PLUGIN_DEFAULTS:
        raise ValueError(f"Unknown NN plugin method '{method}'")
    return NN_PLUGIN_DEFAULTS[method]


@dataclass
class NeighborsComputationResult:
    idx: np.ndarray
    dist: np.ndarray | None
    has_distances: bool


def _compute_neighbors_plugin(
    *,
    method: str,
    plugin_spec,
    data: np.ndarray,
    eff_n_neighbors: int,
    metric: str,
    return_distance: bool,
    include_self: bool,
    verbose: bool,
    method_kwds: dict | None,
    name: str,
    drnb_home: Path | str | None,
    data_sub_dir: str,
    nn_sub_dir: str,
    experiment_name: str | None,
    quiet_failures: bool,
    quiet_plugin_logs: bool,
) -> NeighborsComputationResult:
    from drnb.nnplugins.external import NNPluginContextInfo, run_external_neighbors

    params = _plugin_default_params(method)
    if method_kwds is not None:
        params = params | method_kwds
    if verbose:
        log.info(
            "Finding %d neighbors using plugin %s with %s metric and params: %s",
            eff_n_neighbors,
            method,
            metric,
            params,
        )
    ctx = NNPluginContextInfo(
        dataset_name=name or None,
        drnb_home=Path(drnb_home) if drnb_home else None,
        data_sub_dir=data_sub_dir or "data",
        nn_sub_dir=nn_sub_dir or "nn",
        experiment_name=experiment_name,
    )
    nn = run_external_neighbors(
        method=method,
        spec=plugin_spec,
        data=data,
        n_neighbors=eff_n_neighbors,
        metric=metric,
        params=params,
        return_distance=return_distance,
        ctx=ctx,
        neighbor_name=name or None,
        quiet_failures=quiet_failures,
        quiet_plugin_logs=quiet_plugin_logs,
    )
    idx = nn.idx
    dist = nn.dist if return_distance else None
    if not include_self:
        idx = idx[:, 1:]
        if return_distance and dist is not None:
            dist = dist[:, 1:]

    has_distances = return_distance and dist is not None
    return NeighborsComputationResult(idx=idx, dist=dist, has_distances=has_distances)


def _compute_neighbors_builtin(
    data: np.ndarray,
    eff_n_neighbors: int,
    metric: str,
    method: str,
    return_distance: bool,
    include_self: bool,
    verbose: bool,
    method_kwds: dict | None,
) -> NeighborsComputationResult:
    nn_func, default_method_kwds = create_nn_func(method)

    if method_kwds is None:
        method_kwds = default_method_kwds
    else:
        method_kwds = default_method_kwds | method_kwds

    if verbose:
        log.info(
            "Finding %d neighbors using %s with %s metric and params: %s",
            eff_n_neighbors,
            method,
            metric,
            method_kwds,
        )

    nn = nn_func(
        data,
        n_neighbors=eff_n_neighbors,
        metric=metric,
        return_distance=return_distance,
        **method_kwds,
    )

    if return_distance:
        idx, dist = nn
    else:
        idx = nn
        dist = None

    if not include_self:
        idx = idx[:, 1:]
        if return_distance and dist is not None:
            dist = dist[:, 1:]

    return NeighborsComputationResult(
        idx=idx,
        dist=dist if return_distance else None,
        has_distances=return_distance,
    )


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
    drnb_home: Path | str | None = None,
    data_sub_dir: str = "data",
    nn_sub_dir: str = "nn",
    experiment_name: str | None = None,
    quiet_plugin_failures: bool = False,
    quiet_plugin_logs: bool = False,
) -> NearestNeighbors:
    """Calculate nearest neighbors for a given data set."""
    from drnb.nnplugins.external import NNPluginWorkspaceError

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

    methods = [method]
    if method in ("exact", "approximate"):
        if method == "exact":
            methods = _preferred_exact_methods()[metric]
        else:
            methods = _preferred_fast_methods()[metric]

    # this loop is used to hedge against neighbor plugin failures due to a failed
    # installation of FAISS (which is more than likely). It will try FAISS if it can,
    # and then fall back to the next best method. In general, the other neighbor plugins
    # should have installed just fine, so if there are other failures, that's not a
    # good sign.
    for method in methods:
        # if FAISS is not available, skip it
        if method == "faiss" and not env_flag("DRNB_NN_FAISS", True):
            continue
        if verbose:
            log.info("Using '%s' to find nearest neighbors", method)
        if verbose and method != "faiss" and (n_items > 10000 or data.shape[1] > 10000):
            log.warning(
                "Exact nearest neighbors search with %s might take a while", method
            )

        plugin_spec = _lookup_nn_plugin(method)

        computation_result = None
        try:
            if plugin_spec is not None:
                computation_result = _compute_neighbors_plugin(
                    method=method,
                    plugin_spec=plugin_spec,
                    data=data,
                    eff_n_neighbors=eff_n_neighbors,
                    metric=metric,
                    return_distance=return_distance,
                    include_self=include_self,
                    verbose=verbose,
                    method_kwds=method_kwds,
                    name=name,
                    drnb_home=drnb_home,
                    data_sub_dir=data_sub_dir,
                    nn_sub_dir=nn_sub_dir,
                    experiment_name=experiment_name,
                    quiet_failures=quiet_plugin_failures,
                    quiet_plugin_logs=quiet_plugin_logs,
                )
            else:
                computation_result = _compute_neighbors_builtin(
                    data=data,
                    eff_n_neighbors=eff_n_neighbors,
                    metric=metric,
                    method=method,
                    return_distance=return_distance,
                    include_self=include_self,
                    verbose=verbose,
                    method_kwds=method_kwds,
                )
        except NNPluginWorkspaceError as e:
            msg = f"Error computing neighbors with {method}: {e}"
            if quiet_plugin_failures:
                log.info(msg)
            elif method == "faiss":
                log.info(msg)
            else:
                log.error(msg)
            continue
        break

    if computation_result is None:
        raise ValueError(f"Failed to compute neighbors with {methods=}")

    nn_info = NbrInfo(
        name=name,
        n_nbrs=n_neighbors,
        method=method,
        metric=metric,
        exact=_is_exact_method(method),
        has_distances=computation_result.has_distances,
        idx_path=None,
        dist_path=None,
    )
    if return_distance:
        return NearestNeighbors(
            idx=computation_result.idx, dist=computation_result.dist, info=nn_info
        )
    return NearestNeighbors(idx=computation_result.idx, dist=None, info=nn_info)


def get_neighbors(
    name: str | None = None,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    method: str = "approximate",
    return_distance: bool = True,
    verbose: bool = False,
    drnb_home: Path | str | None = None,
    sub_dir: str = "nn",
    cache: bool = True,
    data: np.ndarray | None = None,
    method_kwds: dict | None = None,
    quiet_plugin_failures: bool = False,
    quiet_plugin_logs: bool = False,
) -> NearestNeighbors:
    """Read cached neighbors or calculate them on demand."""
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
        raise ValueError(
            "Unable to find neighbors data on disk; please provide data to calculate"
        )

    neighbor_data = calculate_neighbors(
        data=data,
        n_neighbors=n_neighbors,
        metric=metric,
        method=method,
        return_distance=return_distance,
        verbose=verbose,
        method_kwds=method_kwds,
        name=name or "",
        drnb_home=drnb_home,
        data_sub_dir="data",
        nn_sub_dir=sub_dir,
        quiet_plugin_failures=quiet_plugin_failures,
        quiet_plugin_logs=quiet_plugin_logs,
    )
    if cache and name:
        write_neighbors(
            neighbor_data=neighbor_data,
            drnb_home=drnb_home,
            sub_dir=sub_dir,
            create_sub_dir=True,
            verbose=verbose,
        )
    return neighbor_data


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
    quiet_plugin_failures: bool = False,
    quiet_plugin_logs: bool = False,
) -> NearestNeighbors:
    """Get exact nearest neighbors for a dataset."""
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
        quiet_plugin_failures=quiet_plugin_failures,
        quiet_plugin_logs=quiet_plugin_logs,
    )


def calculate_exact_neighbors(
    data: np.ndarray | None = None,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    return_distance: bool = True,
    include_self: bool = True,
    verbose: bool = False,
    method_kwds: dict | None = None,
    name: str = "",
    quiet_plugin_failures: bool = False,
    quiet_plugin_logs: bool = False,
) -> NearestNeighbors:
    """Convenience wrapper for exact neighbors."""
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
        quiet_plugin_failures=quiet_plugin_failures,
        quiet_plugin_logs=quiet_plugin_logs,
    )


def _zip_algs(metrics, method_name):
    return list(zip(metrics, itertools.cycle((method_name,))))


def _is_exact_method(method):
    return method in ("faiss", "torchknn", "sklearn")


def _find_exact_method(metric):
    return _preferred_exact_methods()[metric][0]


def _find_fast_method(metric):
    return _preferred_fast_methods()[metric][0]


def _preferred_fast_methods():
    from . import pynndescent as pynndescent_mod

    metric_algs = defaultdict(list)
    for metric, method in _zip_algs(_FAISS_METRICS, "faiss"):
        metric_algs[metric].append(method)

    for metric, method in _zip_algs(_TORCHKNN_METRICS, "torchknn"):
        metric_algs[metric].append(method)

    for metric, method in _zip_algs(
        pynndescent_mod.PYNNDESCENT_METRICS.keys(), "pynndescent"
    ):
        metric_algs[metric].append(method)
    if _plugin_available("hnsw"):
        for metric, method in _zip_algs(_HNSW_METRICS, "hnsw"):
            metric_algs[metric].append(method)
    if _plugin_available("annoy"):
        for metric, method in _zip_algs(_ANNOY_METRICS, "annoy"):
            metric_algs[metric].append(method)
    return metric_algs


def _preferred_exact_methods():
    from . import pynndescent as pynndescent_mod
    from . import sklearn as sknbrs

    metric_algs = defaultdict(list)
    for metric, method in _zip_algs(_FAISS_METRICS, "faiss"):
        metric_algs[metric].append(method)
    for metric, method in _zip_algs(_TORCHKNN_METRICS, "torchknn"):
        metric_algs[metric].append(method)
    for metric, method in _zip_algs(sknbrs.SKLEARN_METRICS.keys(), "sklearn"):
        metric_algs[metric].append(method)
    for metric, method in _zip_algs(
        pynndescent_mod.PYNNDESCENT_METRICS.keys(), "pynndescentbf"
    ):
        metric_algs[metric].append(method)

    return metric_algs


def _slice_neighbors(
    neighbors_data: NearestNeighbors, n_neighbors: int
) -> NearestNeighbors:
    if neighbors_data.info is None:
        raise ValueError("Cannot slice neighbors without metadata")
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
class NeighborsRequest(FromDict):
    """Request for creating neighbors."""

    n_neighbors: list[int] = field(default_factory=lambda: [15])
    method: str = "exact"
    metric: str | list[str] = field(default_factory=lambda: ["euclidean"])
    file_types: list[str] = field(default_factory=lambda: ["pkl"])
    params: dict = field(default_factory=dict)
    verbose: bool = False
    quiet_plugin_failures: bool = False
    quiet_plugin_logs: bool = False

    def create_neighbors(
        self,
        data: np.ndarray,
        dataset_name: str,
        nbr_dir: str = "nn",
        suffix: str | None = None,
    ) -> list[Path]:
        """Create neighbors for a given dataset."""
        if not self.n_neighbors:
            log.info("Neighbor request but no n_neighbors specified")
            return []
        max_n_neighbors = np.max(self.n_neighbors)

        nbrs_name = dataset_name
        if suffix:
            nbrs_name = f"{nbrs_name}-{suffix}"

        if not isinstance(self.metric, (list, tuple)):
            self.metric = [self.metric]
        self.metric = cast(list[str], self.metric)

        neighbors_output_paths: list[Path] = []
        for metric in self.metric:
            neighbors_data = calculate_neighbors(
                data=data,
                n_neighbors=max_n_neighbors,
                metric=metric,
                return_distance=True,
                verbose=self.verbose,
                name=nbrs_name,
                **self.params,
                quiet_plugin_failures=self.quiet_plugin_failures,
                quiet_plugin_logs=self.quiet_plugin_logs,
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
                        "Unable to save %d neighbors (probably not enough neighbors)",
                        n_neighbors,
                    )
        return neighbors_output_paths


def create_neighbors_request(neighbors_kwds: dict | None) -> NeighborsRequest | None:
    """Create a neighbors request from configuration."""
    if neighbors_kwds is None:
        return None
    for key in ["metric", "n_neighbors"]:
        if key in neighbors_kwds and not isinstance(neighbors_kwds[key], (list, tuple)):
            neighbors_kwds[key] = [neighbors_kwds[key]]
    if "verbose" not in neighbors_kwds:
        neighbors_kwds["verbose"] = True
    neighbors_request = NeighborsRequest.new(**neighbors_kwds)
    log.info("Requesting one extra neighbor to account for self-neighbor")
    neighbors_request.n_neighbors = [
        n_nbrs + 1 for n_nbrs in neighbors_request.n_neighbors
    ]
    return neighbors_request
