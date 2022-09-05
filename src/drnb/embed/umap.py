from dataclasses import dataclass

import numpy as np
import pynndescent
import umap

import drnb.embed
import drnb.neighbors as knn
from drnb.log import log


@dataclass
class Umap(drnb.embed.Embedder):
    use_precomputed_knn: bool = False

    def embed_impl(self, x, params, ctx=None):
        knn_params = {}
        if isinstance(self.use_precomputed_knn, dict):
            knn_params = dict(self.use_precomputed_knn)
            self.use_precomputed_knn = True

        if self.use_precomputed_knn:
            log.info("Using precomputed knn")
            metric = params.get("metric", "euclidean")
            n_neighbors = params.get("n_neighbors", 15)
            precomputed_knn = knn.get_neighbors_with_ctx(
                x, metric, n_neighbors, knn_params=knn_params, ctx=ctx
            )

            # third argument is a dummy search index to escape the scrutiny of UMAP's
            # validation: but we aren't going to every transform new data so we don't
            # need it
            precomputed_knn = (
                precomputed_knn.idx,
                precomputed_knn.dist,
                pynndescent.NNDescent(np.array([0]).reshape((1, 1)), n_neighbors=0),
            )
            params["precomputed_knn"] = precomputed_knn

        return embed_umap(x, params)


def embed_umap(
    x,
    params,
):
    if isinstance(x, np.ndarray) and x.shape[0] == x.shape[1]:
        params["metric"] = "precomputed"

    log.info("Running UMAP")
    embedder = umap.UMAP(
        **params,
    )
    embedded = embedder.fit_transform(x)
    log.info("Embedding completed")

    if params.get("densmap", False) and params.get("output_dens", False):
        embedded = dict(coords=embedded[0], dens_ro=embedded[1], dens_re=embedded[2])

    return embedded
