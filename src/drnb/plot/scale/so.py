from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import drnb.neighbors.hubness as hub
from drnb.embed.context import EmbedContext
from drnb.plot.scale import ColorScale


@dataclass
class ColorBySo:
    """Color by the s-occurrence of each point. The s-occurrence is the number of
    mutual neighbors of each point.

    Attributes:
        n_neighbors: The number of neighbors to consider.
        scale: A ColorScale object.
        normalize: Whether to normalize the s-occurrence by the number of neighbors.
        log1p: Whether to take the log of the s-occurrence plus 1
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
        res = np.array(hub.fetch_nbr_stats(ctx.dataset_name, self.n_neighbors)["so"])
        if self.normalize:
            res = res / self.n_neighbors
        if self.log1p:
            return np.log1p(res)
        return res

    def __str__(self) -> str:
        """Return a string representation of the ColorBySo object."""
        return (
            f"so-{self.n_neighbors}"
            f"{' log' if self.log1p else ''}{' norm' if self.normalize else ''}"
        )
