from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from drnb.embed.context import EmbedContext
from drnb.eval.nbrpres import NbrPreservationEval
from drnb.types import EmbedResult


@dataclass
class NbrPreservationHistogram:
    """A histogram of the neighbor preservation evaluation for the embedded data.

    Attributes:
        n_neighbors: The number of neighbors to consider (default 15).
        metric: The metric to use when calculating the neighbor preservation (default
            "euclidean").
        normalize: Whether to normalize the neighbor preservation (default True).
    """

    n_neighbors: int = 15
    metric: str = "euclidean"
    normalize: bool = True

    def npe(self) -> NbrPreservationEval:
        """Create a NbrPreservationEval object."""
        return NbrPreservationEval(
            metric=self.metric, n_neighbors=self.n_neighbors, include_self=False
        )

    def plot(
        self,
        embedding_result: EmbedResult,
        data: np.ndarray | None = None,
        _: pd.DataFrame | pd.Series | np.ndarray | range | None = None,
        ctx: EmbedContext | None = None,
        __: Axes | None = None,
    ) -> Axes | None:
        """Plot a histogram of the neighbor preservation evaluation for the embedded
        data."""
        vec = self.npe().evaluatev(data, embedding_result["coords"], ctx=ctx)[0]
        plot = sns.histplot(x=vec, bins=self.n_neighbors + 1)
        plot.set(title=str(self))
        plot.set_xlabel("neighbor preservation")
        plt.show()

    def requires(self) -> dict:
        """Return the requirements for the NbrPreservationHistogram object."""
        return self.npe().requires()

    def __str__(self):
        return str(self.npe()) + f"{' norm' if self.normalize else ''}"
