from dataclasses import dataclass

import numpy as np
import sklearn.decomposition

import drnb.embed
import drnb.embed.base
from drnb.embed.context import EmbedContext
from drnb.types import EmbedResult


@dataclass
class Pca(drnb.embed.base.Embedder):
    """PCA embedder."""

    random_state: int | None = None

    def embed_impl(
        self, x: np.ndarray, params: dict, _: EmbedContext | None = None
    ) -> EmbedResult:
        if self.random_state is not None:
            params["random_state"] = self.random_state

        return drnb.embed.fit_transform_embed(
            x, params, sklearn.decomposition.PCA, "PCA", n_components=2
        )
