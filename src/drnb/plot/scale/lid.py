from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from drnb.dimension import mle_local
from drnb.embed.context import (
    EmbedContext,
    get_neighbors_with_ctx,
    read_neighbors_with_ctx,
)
from drnb.neighbors.nbrinfo import NearestNeighbors
from drnb.plot.scale import ColorScale


@dataclass
class ColorByLid:
    """Color by the Levina-Bickel local intrinsic dimensionality estimate."""

    n_neighbors: int = 15
    metric: str = "euclidean"
    scale: ColorScale = field(default_factory=ColorScale)
    remove_self: bool = True
    eps: float = 1.0e-10
    knn_params: dict | None = field(default_factory=lambda: {"method": "sklearn"})

    def _target_neighbor_count(self) -> int:
        return self.n_neighbors + (1 if self.remove_self else 0)

    def _neighbor_method(self) -> str | None:
        if self.knn_params is None:
            return None
        method = self.knn_params.get("method")
        return method if isinstance(method, str) else None

    def _read_neighbors(
        self, ctx: EmbedContext | None, required_neighbors: int
    ) -> NearestNeighbors | None:
        if ctx is None:
            return None
        return read_neighbors_with_ctx(
            metric=self.metric,
            n_neighbors=required_neighbors,
            ctx=ctx,
            return_distance=True,
        )

    def _get_neighbor_distances(
        self, data: np.ndarray, ctx: EmbedContext | None
    ) -> tuple[np.ndarray, int]:
        required_neighbors = self._target_neighbor_count()
        neighbors = self._read_neighbors(ctx, required_neighbors)
        if neighbors is None:
            neighbors = get_neighbors_with_ctx(
                data=data,
                metric=self.metric,
                n_neighbors=required_neighbors,
                knn_params=self.knn_params,
                ctx=ctx,
                return_distance=True,
                quiet_failures=True,
            )

        if neighbors.dist is None:
            raise ValueError(
                "Nearest neighbor distances are required to compute intrinsic dimensionality"
            )

        available_neighbors = neighbors.dist.shape[1]
        if self.remove_self:
            available_neighbors -= 1
        if available_neighbors < 2:
            raise ValueError(
                f"Need at least two neighbors to estimate intrinsic dimensionality, "
                f"found {available_neighbors}"
            )

        effective_n_neighbors = min(self.n_neighbors, available_neighbors)
        columns_needed = effective_n_neighbors + (1 if self.remove_self else 0)
        distances = neighbors.dist[:, :columns_needed]
        return distances, effective_n_neighbors

    def __call__(
        self,
        data: np.ndarray,
        _: pd.DataFrame | pd.Series | np.ndarray | range | None,
        __: np.ndarray,
        ctx: EmbedContext | None = None,
    ) -> np.ndarray:
        distances, effective_n_neighbors = self._get_neighbor_distances(
            np.asarray(data), ctx
        )
        return mle_local(
            distances,
            eps=self.eps,
            n_neighbors=effective_n_neighbors,
            remove_self=self.remove_self,
        )

    def __str__(self) -> str:
        base = f"lid-{self.n_neighbors}-{self.metric}"
        method = self._neighbor_method()
        if method is not None and method != "sklearn":
            base = f"{base}-{method}"
        if not self.remove_self:
            base = f"{base}-self"
        return base
