from dataclasses import dataclass

import numpy as np

import drnb.neighbors as nbrs
from drnb.embed.context import EmbedContext, get_neighbors_with_ctx
from drnb.log import log


@dataclass
class InitMixin:
    """Mixin for handling precomputed initial coordinates.

    Attributes:
        precomputed_init: Optional array of initial coordinates for embedding.
    """

    precomputed_init: np.ndarray | None = None

    def handle_precomputed_init(self, params: dict) -> dict:
        """Store precomputed initial coordinates in params."""
        if self.precomputed_init is not None:
            log.info("Using precomputed initial coordinates")
            params["init"] = self.precomputed_init
        return params


@dataclass
class KNNMixin:
    """Mixin for handling precomputed knn.

    Attributes:
        precomputed_knn: Optional precomputed NearestNeighbors.
    """

    precomputed_knn: nbrs.NearestNeighbors | None = None

    def handle_precomputed_knn(
        self,
        x: np.ndarray,
        params: dict,
        n_neighbors_default: int = 15,
        ctx: EmbedContext | None = None,
    ) -> dict:
        """Store precomputed knn in params."""
        if "n_neighbors" in params:
            n_neighbors = params["n_neighbors"]
            del params["n_neighbors"]
        else:
            n_neighbors = n_neighbors_default

        if self.precomputed_knn is not None:
            log.info("Using directly-provided precomputed knn")
            precomputed_knn = self.precomputed_knn
            if precomputed_knn.dist is None:
                raise ValueError("Must provide neighbor distance in precomputed knn")
        else:
            log.info("Looking up precomputed knn")
            precomputed_knn = get_neighbors_with_ctx(
                x, params.get("metric", "euclidean"), n_neighbors + 1, ctx=ctx
            )

        params["knn_idx"] = precomputed_knn.idx[:, 1:]
        params["knn_dist"] = precomputed_knn.dist[:, 1:]

        return params
