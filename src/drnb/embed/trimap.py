from dataclasses import dataclass

import numpy as np
import trimap

import drnb.embed
import drnb.embed.base
from drnb.embed.context import EmbedContext, get_neighbors_with_ctx
from drnb.log import log
from drnb.types import EmbedResult


@dataclass
class Trimap(drnb.embed.base.Embedder):
    """Trimap embedding.

    Attributes:
        init (str | None): Optional initialization method ("PCA" or "random")
        return_every (int | None): Optional return every nth iteration
        use_precomputed_knn (bool | None): Use precomputed knn or not
    """

    init: str | None = None
    return_every: int | None = None
    use_precomputed_knn: bool | None = True

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        knn_params = {}
        if isinstance(self.use_precomputed_knn, dict):
            knn_params = dict(self.use_precomputed_knn)
            self.use_precomputed_knn = True

        if self.use_precomputed_knn:
            log.info("Using precomputed knn")
            metric = params.get("metric", "euclidean")
            n_neighbors = params.get("n_neighbors", 15)
            precomputed_knn = get_neighbors_with_ctx(
                x, metric, n_neighbors, knn_params=knn_params, ctx=ctx
            )
            params["knn_tuple"] = (precomputed_knn.idx, precomputed_knn.dist)

        return trimap_embed(x, params, self.init, self.return_every)


# pylint: disable=protected-access
def trimap_embed(
    x: np.ndarray, params: dict, init: str, return_every: int | None
) -> np.ndarray | dict:
    """Embed data using TriMap."""
    # these settings just for the (rare case) when return_seq and return_every are set
    orig_return_every = trimap.trimap_._RETURN_EVERY
    if return_every is None:
        return_every = orig_return_every

    if params.get("return_seq", False):
        # https://github.com/eamid/trimap/issues/23
        # setting return_seq=True and init to "pca" or "random" will cause a crash
        if isinstance(init, str):
            init = None
            log.info("Setting init=None to avoid crash with return_seq=True")

    log.info("Running TriMap")
    # This is in a try block just to try really hard to put the internals of trimap
    # back to how we found them even if an exception gets thrown
    try:
        if return_every != orig_return_every:
            trimap.trimap_._RETURN_EVERY = return_every
        embedder = trimap.TRIMAP(n_dims=2, **params)
        result = embedder.fit_transform(x, init=init)

        if params.get("return_seq", False):
            embedded = {"coords": result[:, :, -1], "snapshots": {}}
            for i in range(result.shape[-1]):
                embedded["snapshots"][f"it_{return_every * i}"] = result[:, :, i]
        else:
            embedded = result
        log.info("Embedding completed")
        return embedded
    finally:
        # put _RETURN_EVERY back if necessary
        if trimap.trimap_._RETURN_EVERY != orig_return_every:
            trimap.trimap_._RETURN_EVERY = orig_return_every
