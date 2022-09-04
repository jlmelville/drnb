from dataclasses import dataclass

import numpy as np
import pynndescent
import umap

import drnb.embed
import drnb.neighbors as knn


@dataclass
class Umap(drnb.embed.Embedder):
    use_precomputed_knn: bool = False

    def embed_impl(self, x, params, ctx=None):
        if self.use_precomputed_knn:
            if ctx is not None:
                metric = params.get("metric", "euclidean")
                n_neighbors = params.get("n_neighbors", 15)
                precomputed_knn = knn.get_neighbors(
                    data=x,
                    n_neighbors=n_neighbors,
                    metric=metric,
                    method="exact",
                    return_distance=True,
                    verbose=True,
                    data_path=ctx.data_path,
                    sub_dir=ctx.nn_sub_dir,
                    name=ctx.name,
                    cache=False,
                )
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

    embedder = umap.UMAP(
        **params,
    )
    embedded = embedder.fit_transform(x)

    if params.get("densmap", False) and params.get("output_dens", False):
        embedded = dict(coords=embedded[0], dens_ro=embedded[1], dens_re=embedded[2])

    return embedded
