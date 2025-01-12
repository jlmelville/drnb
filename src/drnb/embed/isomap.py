from dataclasses import dataclass

import numpy as np
import sklearn.manifold

import drnb.embed.base
from drnb.embed.context import EmbedContext
from drnb.types import EmbedResult


@dataclass
class Isomap(drnb.embed.base.Embedder):
    """Embed the data using Isomap."""

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        return drnb.embed.fit_transform_embed(
            x, params, sklearn.manifold.Isomap, "Isomap", n_components=2
        )
