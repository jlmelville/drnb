from dataclasses import dataclass

import numpy as np
import sklearn.manifold

import drnb.embed
import drnb.embed.base
from drnb.embed.context import EmbedContext
from drnb.types import EmbedResult


@dataclass
class Mmds(drnb.embed.base.Embedder):
    """Metric Multidimensional Scaling (MMDS) embedding using sklearn."""

    def embed_impl(
        self, x: np.ndarray, params: dict, _: EmbedContext | None = None
    ) -> EmbedResult:
        return drnb.embed.fit_transform_embed(
            x,
            params,
            sklearn.manifold.MDS,
            "sklearn-MDS",
            n_components=2,
            metric=True,
            normalized_stress=False,
        )
