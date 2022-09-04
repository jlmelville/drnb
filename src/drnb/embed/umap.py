from dataclasses import dataclass

import numpy as np
import pynndescent
import umap

import drnb.embed
import drnb.neighbors as knn


@dataclass
class Umap(drnb.embed.Embedder):
    def embed(self, x, ctx=None):
        embedder_kwds = dict(self.embedder_kwds)
        if ctx is not None:
            if "_nbrs" in embedder_kwds:
                if embedder_kwds["_nbrs"]:
                    metric = embedder_kwds.get("metric", "euclidean")
                    n_neighbors = embedder_kwds.get("n_neighbors", 15)
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
                        pynndescent.NNDescent(
                            np.array([0]).reshape((1, 1)), n_neighbors=0
                        ),
                    )
                    embedder_kwds["precomputed_knn"] = precomputed_knn
                del embedder_kwds["_nbrs"]
        return embed_umap(x, embedder_kwds)


def embed_umap(
    x,
    embedder_kwds,
):
    if isinstance(x, np.ndarray) and x.shape[0] == x.shape[1]:
        embedder_kwds["metric"] = "precomputed"

    embedder = umap.UMAP(
        **embedder_kwds,
    )
    embedded = embedder.fit_transform(x)

    if embedder_kwds.get("densmap", False) and embedder_kwds.get("output_dens", False):
        embedded = dict(coords=embedded[0], dens_ro=embedded[1], dens_re=embedded[2])

    return embedded
