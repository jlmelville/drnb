import ncvis
import numpy as np

import drnb.embed
import drnb.embed.base
from drnb.embed.context import EmbedContext
from drnb.types import EmbedResult


class NCVis(drnb.embed.base.Embedder):
    """NCVis embedder.

    Possible params:

    * d=2 (int): Number of dimensions.
    * n_threads=-1 (int): Number of threads to use.
    * n_neighbors=15 (int): Number of neighbors.
    * M=16 (int): Number of landmarks.
    * ef_construction=200 (int): Construction parameter.
    * random_seed=42 (int): Random seed.
    * n_epochs=50 (int): Number of epochs.
    * n_init_epochs=20 (int): Number of initialization epochs.
    * spread=1.0 (float): Spread.
    * min_dist=0.4 (float): Minimum distance.
    * a=None (float): A.
    * b=None (float): B.
    * alpha=1.0 (float): Alpha.
    * alpha_Q=1.0 (float): Alpha Q.
    * n_noise=None (int): Number of noise points.
    * distance="euclidean" (str): Distance metric.
    """

    def embed_impl(
        self, x: np.ndarray, params: dict, _: EmbedContext | None = None
    ) -> EmbedResult:
        return drnb.embed.fit_transform_embed(
            x,
            params,
            ncvis.NCVis,
            "Sparse Random Projection",
        )
