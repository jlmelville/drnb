from dataclasses import dataclass

import drnb.embed
import drnb.neighbors as nbrs
from drnb.embed.umap import umap_spectral_init
from drnb.log import log


@dataclass
class UmapSpectral(drnb.embed.Embedder):
    use_precomputed_knn: bool = True

    def embed_impl(self, x, params, ctx=None):
        if self.use_precomputed_knn:
            log.info("Using precomputed knn")
            metric = params.get("metric", "euclidean")
            n_neighbors = params.get("n_neighbors", 15)
            precomputed_knn = nbrs.get_neighbors_with_ctx(
                x, metric, n_neighbors, ctx=ctx
            )
            params["knn"] = [precomputed_knn.idx, precomputed_knn.dist]

        return embed_umap_spectral(x, params)


def embed_umap_spectral(
    x,
    params,
):
    log.info("Running UMAP Spectral Embedding")
    embedded = umap_spectral_init(x, **params)
    log.info("Embedding completed")

    return embedded
