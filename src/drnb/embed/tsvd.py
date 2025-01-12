from dataclasses import dataclass

import numpy as np
import sklearn.decomposition

import drnb.embed
import drnb.embed.base
from drnb.embed.context import EmbedContext
from drnb.preprocess import center
from drnb.types import EmbedResult


@dataclass
class Tsvd(drnb.embed.base.Embedder):
    """Truncated SVD embedding."""

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        x = center(x)
        return drnb.embed.fit_transform_embed(
            x,
            params,
            sklearn.decomposition.TruncatedSVD,
            "TruncatedSVD",
            n_components=2,
        )
