from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from drnb.embed.context import EmbedContext
from drnb.eval.rte import RandomTripletEval
from drnb.plot.scale import ColorScale


@dataclass
class ColorByRte:
    """Color by the random triplet accuracy of each point. The random triplet accuracy
    is the fraction of triplets that are correctly ordered in the embedded space.

    Attributes:
        n_triplets_per_point: The number of triplets to consider per point (default 5).
        scale: A ColorScale object (default None).
        normalize: Whether to normalize the random triplet accuracy by the number of
            triplets (default True).
        metric: The metric to use when calculating the random triplet accuracy (default
            "euclidean").
        random_state: The random state to use when calculating the random triplet
            accuracy (default None).
    """

    n_triplets_per_point: int = 5
    scale: ColorScale = field(default_factory=ColorScale)
    normalize: bool = True
    metric: str = "euclidean"
    random_state: int | None = None

    def rte(self) -> RandomTripletEval:
        """Create a RandomTripletEval object."""
        return RandomTripletEval(
            random_state=self.random_state,
            metric=self.metric,
            n_triplets_per_point=self.n_triplets_per_point,
        )

    def __call__(
        self,
        data: np.ndarray,
        _: pd.DataFrame | pd.Series | np.ndarray | range | None,
        coords: np.ndarray,
        ctx: EmbedContext | None = None,
    ) -> np.ndarray:
        return self.rte().evaluatev(data, coords, ctx)

    def __str__(self):
        return str(self.rte()) + f"{' norm' if self.normalize else ''}"

    def requires(self) -> dict:
        """Return the requirements for the ColorByRte object."""
        return self.rte().requires()
