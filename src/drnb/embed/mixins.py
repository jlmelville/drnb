from dataclasses import dataclass
from typing import Optional, cast

import numpy as np

import drnb.neighbors as nbrs
from drnb.embed import EmbedContext
from drnb.log import log


@dataclass
class InitMixin:
    precomputed_init: Optional[np.ndarray] = None

    def handle_precomputed_init(self, params: dict):
        if self.precomputed_init is not None:
            log.info("Using precomputed initial coordinates")
            params["init"] = self.precomputed_init
        return params


@dataclass
class KNNMixin:
    precomputed_knn: Optional[nbrs.NearestNeighbors] = None

    def handle_precomputed_knn(
        self,
        x: np.ndarray,
        params: dict,
        n_neighbors_default: int = 15,
        ctx: Optional[EmbedContext] = None,
    ):
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
            precomputed_knn = nbrs.get_neighbors_with_ctx(
                x, params.get("metric", "euclidean"), n_neighbors + 1, ctx=ctx
            )
            precomputed_knn.dist = cast(np.ndarray, precomputed_knn.dist)

        params["knn_idx"] = precomputed_knn.idx[:, 1:]
        params["knn_dist"] = precomputed_knn.dist[:, 1:]

        return params
