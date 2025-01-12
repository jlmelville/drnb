from dataclasses import dataclass

import numpy as np

import drnb.embed
import drnb.embed.base
from drnb.embed.context import EmbedContext, get_neighbors_with_ctx
from drnb.log import log
from drnb.types import EmbedResult
from drnb.yinit import binary_graph_spectral_init, umap_graph_spectral_init


@dataclass
class UmapSpectral(drnb.embed.base.Embedder):
    """Embedder using just the spectral embedding initialization used in UMAP"""

    use_precomputed_knn: bool = True

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        if self.use_precomputed_knn:
            log.info("Using precomputed knn")
            metric = params.get("metric", "euclidean")
            n_neighbors = params.get("n_neighbors", 15)
            precomputed_knn = get_neighbors_with_ctx(x, metric, n_neighbors, ctx=ctx)
            params["knn"] = [precomputed_knn.idx, precomputed_knn.dist]

        log.info("Running UMAP Spectral Embedding")
        embedded = umap_graph_spectral_init(x, **params)
        log.info("Embedding completed")

        return embedded


@dataclass
class BinaryGraphSpectral(drnb.embed.base.Embedder):
    """Embedder using a binary weighted version of the spectral embedding initialization
    used in UMAP"""

    use_precomputed_knn: bool = True

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        if self.use_precomputed_knn:
            log.info("Using precomputed knn")
            metric = params.get("metric", "euclidean")
            n_neighbors = params.get("n_neighbors", 15)
            precomputed_knn = get_neighbors_with_ctx(x, metric, n_neighbors, ctx=ctx)
            params["knn"] = [precomputed_knn.idx, precomputed_knn.dist]

        log.info("Running Binary Graph Spectral Embedding")
        embedded = binary_graph_spectral_init(x, **params)
        log.info("Embedding completed")

        return embedded
