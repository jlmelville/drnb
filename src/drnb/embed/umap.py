from dataclasses import dataclass

import numpy as np
import pynndescent
import umap

import drnb.embed
import drnb.neighbors as knn
from drnb.log import log


# A subclass of NNDescent which exists purely to escape the scrutiny of a validation
# type check in UMAP when using pre-computed knn.
# https://github.com/lmcinnes/umap/issues/848
class DummyNNDescent(pynndescent.NNDescent):
    # pylint: disable=super-init-not-called
    def __init__(self):
        return


@dataclass
class Umap(drnb.embed.Embedder):
    use_precomputed_knn: bool = True

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

            # we aren't going to transform new data so we don't need the search index
            # to actually work
            dummy_search_index = DummyNNDescent()
            precomputed_knn = (
                precomputed_knn.idx,
                precomputed_knn.dist,
                dummy_search_index,
            )
            params["precomputed_knn"] = precomputed_knn
            # also UMAP complains when a precomputed knn is used with a smaller dataset
            # unless this flag is set
            params["force_approximation_algorithm"] = True

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
