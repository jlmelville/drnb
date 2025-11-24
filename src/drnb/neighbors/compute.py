from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import sklearn.metrics

from drnb.log import log
from drnb.neighbors.nbrinfo import NbrInfo, NearestNeighbors
from drnb.neighbors.store import read_neighbors, write_neighbors
from drnb.nnplugins.external import NNPluginContextInfo, run_external_neighbors
from drnb.preprocess import numpyfy
from drnb.types import DataSet
from drnb.util import FromDict


def n_connected_components(graph: scipy.sparse.coo_matrix) -> int:
    """Return the number of connected components in a graph."""
    return scipy.sparse.csgraph.connected_components(graph)[0]


NN_PLUGIN_DEFAULTS: dict[str, dict[str, int]] = {
    "annoy": {"n_trees": 50, "search_k": -1, "random_state": 42, "n_jobs": -1},
}

_ANNOY_METRICS = ("dot", "cosine", "manhattan", "euclidean")


def dmat(x: DataSet | np.ndarray) -> np.ndarray:
    """Calculate the pairwise distance matrix for a given data set."""
    if isinstance(x, tuple) and len(x) == 2:
        x = x[0]
    x = numpyfy(x)
    return sklearn.metrics.pairwise_distances(x)


def create_nn_func(method: str) -> Tuple[Callable, Dict]:
    """Create a nearest neighbor function and default keyword arguments."""
    # if method == "annoy":
    #     raise ValueError(
    #         "annoy neighbors must be provided by the NN plugin (no in-process support)"
    #     )
    if method == "sklearn":
        from . import sklearn as sknbrs

        nn_func = sknbrs.sklearn_neighbors
        default_method_kwds = sknbrs.SKLEARN_DEFAULTS
    elif method == "faiss":
        from . import faiss as faiss_mod

        nn_func = faiss_mod.faiss_neighbors
        default_method_kwds = faiss_mod.FAISS_DEFAULTS
    elif method == "pynndescent":
        from . import pynndescent as pynndescent_mod

        nn_func = pynndescent_mod.pynndescent_neighbors
        default_method_kwds = pynndescent_mod.PYNNDESCENT_DEFAULTS
    elif method == "pynndescentbf":
        from . import pynndescent as pynndescent_mod

        nn_func = pynndescent_mod.pynndescent_exact_neighbors
        default_method_kwds = {}
    elif method == "hnsw":
        from . import hnsw as hnsw_mod

        nn_func = hnsw_mod.hnsw_neighbors
        default_method_kwds = hnsw_mod.HNSW_DEFAULTS
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


def _plugin_default_params(method: str) -> dict:
    if method not in NN_PLUGIN_DEFAULTS:
        raise ValueError(f"Unknown NN plugin method '{method}'")
    return NN_PLUGIN_DEFAULTS[method]


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
) -> NearestNeighbors:
    """Calculate nearest neighbors for a given data set."""
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

    plugin_spec = _lookup_nn_plugin(method)
    if plugin_spec is not None:
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
        )
        idx = nn.idx
        dist = nn.dist
        if not include_self:
            if return_distance:
                idx = idx[:, 1:]
                dist = dist[:, 1:] if dist is not None else None
            else:
                idx = idx[:, 1:]
        nn_info = NbrInfo(
            name=name,
            n_nbrs=n_neighbors,
            metric=metric,
            exact=_is_exact_method(method),
            method=method,
            has_distances=return_distance and dist is not None,
            idx_path=None,
            dist_path=None,
        )
        if return_distance:
            return NearestNeighbors(idx=idx, dist=dist, info=nn_info)
        return NearestNeighbors(idx=idx, dist=None, info=nn_info)

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

    try:
        nn = nn_func(
            data,
            n_neighbors=eff_n_neighbors,
            metric=metric,
            return_distance=return_distance,
            **method_kwds,
        )
    except RuntimeError as exc:
        if method == "faiss":
            log.warning(
                "faiss neighbors failed (out of memory?), falling back to pynndescent"
            )
            method = "pynndescent"
            from . import pynndescent as pynndescent_mod

            nn_func = pynndescent_mod.pynndescent_neighbors
            default_method_kwds = pynndescent_mod.PYNNDESCENT_DEFAULTS
            nn = nn_func(
                data,
                n_neighbors=eff_n_neighbors,
                metric=metric,
                return_distance=return_distance,
                **method_kwds,
            )
        else:
            raise exc

    if not include_self:
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
    )


def _zip_algs(metrics, method_name):
    return list(zip(metrics, itertools.cycle((method_name,))))


def _is_exact_method(method):
    return method in ("faiss", "sklearn")


def _find_exact_method(metric):
    return _preferred_exact_methods()[metric][0]


def _find_fast_method(metric):
    return _preferred_fast_methods()[metric][0]


def _preferred_fast_methods():
    from . import faiss as faiss_mod
    from . import hnsw as hnsw_mod
    from . import pynndescent as pynndescent_mod

    metric_algs = defaultdict(list)
    for metric, method in (
        _zip_algs(faiss_mod.faiss_metrics(), "faiss")
        + _zip_algs(pynndescent_mod.PYNNDESCENT_METRICS.keys(), "pynndescent")
        + _zip_algs(hnsw_mod.HNSW_METRICS.keys(), "hnsw")
        + _zip_algs(_ANNOY_METRICS, "annoy")
    ):
        metric_algs[metric].append(method)

    return metric_algs


def _preferred_exact_methods():
    from . import faiss as faiss_mod
    from . import pynndescent as pynndescent_mod
    from . import sklearn as sknbrs

    metric_algs = defaultdict(list)
    for metric, method in (
        _zip_algs(faiss_mod.faiss_metrics(), "faiss")
        + _zip_algs(sknbrs.SKLEARN_METRICS.keys(), "sklearn")
        + _zip_algs(pynndescent_mod.PYNNDESCENT_METRICS.keys(), "pynndescentbf")
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
        if suffix:
            nbrs_name = f"{nbrs_name}-{suffix}"

        if not isinstance(self.metric, (list, tuple)):
            self.metric = [self.metric]
        self.metric = cast(List[str], self.metric)

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
