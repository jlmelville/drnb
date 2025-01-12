from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from drnb.embed.context import EmbedContext
from drnb.eval.nbrpres import NbrPreservationEval
from drnb.plot.scale import ColorScale


@dataclass
class ColorByNbrPres:
    """Color by the neighbor preservation of each point. The neighbor preservation is
    the fraction of a point's neighbors that are also neighbors in the embedded space.

    Attributes:
        n_neighbors: The number of neighbors to consider (default 15).
        scale: A ColorScale object (default None).
        normalize: Whether to normalize the neighbor preservation by the number of
            neighbors (default True).
        metric: The metric to use when calculating the neighbor preservation (default
            "euclidean").
    """

    n_neighbors: int = 15
    scale: ColorScale = field(default_factory=ColorScale)
    normalize: bool = True
    metric: str = "euclidean"

    def npe(self) -> NbrPreservationEval:
        """Create a NbrPreservationEval object."""
        return NbrPreservationEval(
            metric=self.metric, n_neighbors=self.n_neighbors, include_self=False
        )

    def __call__(
        self,
        data: np.ndarray,
        _: pd.DataFrame | pd.Series | np.ndarray | range | None,
        coords: np.ndarray,
        ctx: EmbedContext | None = None,
    ) -> np.ndarray:
        return self.npe().evaluatev(data, coords, ctx=ctx)[0]

    def __str__(self):
        return str(self.npe()) + f"{' norm' if self.normalize else ''}"

    def requires(self) -> dict:
        """Return the requirements for the ColorByNbrPres object."""
        return self.npe().requires()
