from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import drnb.neighbors.hubness as hub
from drnb.embed.context import EmbedContext
from drnb.plot.scale import ColorScale


@dataclass
class ColorByKo:
    """Color by the k-occurrence of each point. The k-occurrence is the number of
    times a point is a neighbor of another point in the dataset.

    Attributes:
        n_neighbors: The number of neighbors to consider.
        scale: A ColorScale object.
        normalize: Whether to normalize the k-occurrence by the number of neighbors.
        log1p: Whether to take the log of the k-occurrence plus 1
    """

    n_neighbors: int = 15
    scale: ColorScale = field(default_factory=ColorScale)
    normalize: bool = True
    log1p: bool = False

    def __call__(
        self,
        _: np.ndarray,
        __: pd.DataFrame | pd.Series | np.ndarray | range | None,
        ___: np.ndarray,
        ctx: EmbedContext | None = None,
    ) -> np.ndarray:
        if ctx is None:
            return []
        res = np.array(hub.fetch_nbr_stats(ctx.dataset_name, self.n_neighbors)["ko"])
        if self.normalize:
            res = res / self.n_neighbors
        if self.log1p:
            return np.log1p(res)
        return res

    def __str__(self) -> str:
        return (
            f"ko-{self.n_neighbors}"
            f"{' log' if self.log1p else ''}{' norm' if self.normalize else ''}"
        )
