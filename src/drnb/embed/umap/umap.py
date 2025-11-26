from dataclasses import dataclass
from typing import Any, Dict, Tuple, cast

import numpy as np
import pynndescent
import umap
from scipy.sparse import coo_matrix

import drnb.embed
import drnb.embed.base
import drnb.neighbors.compute as nbrs
from drnb.embed.context import EmbedContext, get_neighbors_with_ctx
from drnb.log import log
from drnb.neighbors.compute import n_connected_components
from drnb.types import EmbedResult
from drnb.util import get_method_and_args
from drnb.yinit import (
    spca,
    spectral_graph_embed,
    tsvd_warm_spectral,
    umap_graph_spectral_init,
)


# https://github.com/lmcinnes/umap/issues/848
class DummyNNDescent(pynndescent.NNDescent):
    """A subclass of NNDescent which exists purely to escape the scrutiny of a
    validation type check in UMAP when using pre-computed knn."""

    # pylint: disable=super-init-not-called
    def __init__(self):
        return


def umap_knn(
    idx: np.ndarray, dist: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, pynndescent.NNDescent]:
    """Construct a tuple of nearest neighbors for UMAP."""
    # we aren't going to transform new data so we don't need the search index
    # to actually work
    dummy_search_index = DummyNNDescent()
    return (
        idx,
        dist,
        dummy_search_index,
    )


def umap_spectral_init(
    x: np.ndarray,
    knn: Tuple[np.ndarray, np.ndarray] | nbrs.NearestNeighbors | None = None,
    metric: str = "euclidean",
    n_neighbors: int = 15,
    random_state: int = 42,
    tsvdw: bool = False,
    tsvdw_tol: float = 1e-5,
    jitter: bool = True,
) -> np.ndarray:
    """Initialize UMAP embedding using spectral initialization."""
    if knn is None:
        nbr_data = nbrs.calculate_neighbors(
            x,
            n_neighbors=n_neighbors,
            metric=metric,
            method="pynndescent",
            return_distance=True,
            method_kwds={"random_state": random_state},
        )
        knn = [nbr_data.idx, nbr_data.dist]
    knn_fss = umap_graph(knn)

    nc = n_connected_components(knn_fss)
    if nc > 1:
        log.warning("UMAP graph has %d components", nc)

    return spectral_graph_embed(knn_fss, random_state, tsvdw, tsvdw_tol, jitter)


def umap_graph(
    knn: Tuple[np.ndarray, np.ndarray] | nbrs.NearestNeighbors, x: np.ndarray = None
) -> coo_matrix:
    """Construct the fuzzy simplicial set (symmetric affinity matrix) from a k-nearest
    neighbors graph."""
    if isinstance(knn, nbrs.NearestNeighbors):
        knn = [knn.idx, knn.dist]
    if x is None:
        x = np.empty((knn[0].shape[0], 0), dtype=np.int8)
    knn_fss, _, _ = umap.umap_.fuzzy_simplicial_set(
        X=x,
        knn_indices=knn[0],
        knn_dists=knn[1],
        n_neighbors=knn[0].shape[1],
        random_state=None,
        metric=None,
        return_dists=None,
    )
    return knn_fss


@dataclass
class Umap(drnb.embed.base.Embedder):
    """Embedder for UMAP.

    Attributes:
        use_precomputed_knn: Whether to use precomputed nearest neighbors.
        drnb_init: Method for initializing UMAP.
        precomputed_init: Precomputed initial coordinates for embedding.
    """

    use_precomputed_knn: bool = True
    drnb_init: str | None = None
    precomputed_init: np.ndarray | None = None

    def update_params(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> dict:
        """Update parameters for UMAP embedding, including initialization and
        pre-computed nearest neighbors."""
        knn_params = {}
        if isinstance(self.use_precomputed_knn, dict):
            knn_params = dict(self.use_precomputed_knn)
            self.use_precomputed_knn = True

        metric = params.get("metric", "euclidean")
        n_neighbors = params.get("n_neighbors", 15)
        if self.use_precomputed_knn and ctx is not None:
            log.info("Using precomputed knn")

            precomputed_knn = get_neighbors_with_ctx(
                x, metric, n_neighbors, knn_params=knn_params, ctx=ctx
            )

            params["precomputed_knn"] = umap_knn(
                precomputed_knn.idx, precomputed_knn.dist
            )
            # also UMAP complains when a precomputed knn is used with a smaller dataset
            # unless this flag is set
            params["force_approximation_algorithm"] = True

        if self.precomputed_init is not None:
            log.info("Using precomputed initial coordinates")
            params["init"] = self.precomputed_init
        elif self.drnb_init is not None:
            drnb_init, init_params = get_method_and_args(self.drnb_init, {})
            init_params = cast(Dict[str, Any], init_params)
            if drnb_init == "spca":
                params["init"] = spca(x)
            elif drnb_init == "global_spectral":
                params["init"] = umap_graph_spectral_init(
                    x,
                    knn=params.get("precomputed_knn"),
                    metric=metric,
                    n_neighbors=n_neighbors,
                    op=init_params.get("op", "intersection"),
                    global_weight=init_params.get("global_weight", 0.2),
                    random_state=params.get("random_state", 42),
                    global_neighbors=init_params.get("global_neighbors", "random"),
                    n_global_neighbors=init_params.get("n_global_neighbors"),
                    tsvdw=init_params.get("tsvdw", False),
                    tsvdw_tol=init_params.get("tsvdw_tol", 1e-5),
                )
            elif drnb_init == "tsvd_spectral":
                log.info("Initializing via truncated SVD-warmed spectral")
                graph = umap_graph(params["precomputed_knn"], x)
                params["init"] = tsvd_warm_spectral(
                    graph,
                    dim=2,
                    random_state=params.get("random_state", 42),
                )
            else:
                raise ValueError(f"Unknown drnb initialization '{self.drnb_init}'")

        if isinstance(x, np.ndarray) and x.shape[0] == x.shape[1]:
            params["metric"] = "precomputed"

        # tqdm doesn't behave well in a notebook so if verbose is set, unless tqdm_kwds
        # has been set explicitly, disable it
        if params.get("verbose", False) and "tqdm_kwds" not in params:
            params["tqdm_kwds"] = {"disable": True}

        return params

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        params = self.update_params(x, params, ctx)
        if "random_state" in params:
            # UMAP will complain if n_jobs is not 1 when random_state is set
            # so just shut it up here
            params["n_jobs"] = 1

        log.info("Running UMAP")
        embedder = umap.UMAP(
            **params,
        )
        embedded = embedder.fit_transform(x)
        log.info("Embedding completed")

        if params.get("densmap", False) and params.get("output_dens", False):
            embedded = {
                "coords": embedded[0],
                "dens_ro": embedded[1],
                "dens_re": embedded[2],
            }

        return embedded
