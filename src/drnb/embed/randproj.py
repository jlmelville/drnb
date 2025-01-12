from dataclasses import dataclass

import numpy as np
import sklearn.random_projection

import drnb.embed
import drnb.embed.base
from drnb.embed.context import EmbedContext
from drnb.types import EmbedResult


@dataclass
class RandProj(drnb.embed.base.Embedder):
    """Sparse Random Projection embedder."""

    def embed_impl(
        self, x: np.ndarray, params: dict, _: EmbedContext | None = None
    ) -> EmbedResult:
        return drnb.embed.fit_transform_embed(
            x,
            params,
            sklearn.random_projection.SparseRandomProjection,
            "Sparse Random Projection",
            n_components=2,
        )
