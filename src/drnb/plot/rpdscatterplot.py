from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from drnb.embed.context import EmbedContext
from drnb.eval.rpc import RandomPairCorrelEval
from drnb.types import EmbedResult


@dataclass
class RandomPairDistanceScatterplot:
    """A scatterplot of the random pair distances for the embedded data.

    Attributes:
        n_triplets_per_point: The number of triplets to consider per point (default 5).
        metric: The metric to use when calculating the random pair distances (default
            "euclidean").
        random_state: The random state to use when calculating the random pair distances
            (default None).
    """

    n_triplets_per_point: int = 5
    metric: str = "euclidean"
    random_state: int | None = None

    def rpc(self) -> RandomPairCorrelEval:
        """Create a RandomPairCorrelEval object."""
        return RandomPairCorrelEval(
            random_state=self.random_state,
            metric=self.metric,
            n_triplets_per_point=self.n_triplets_per_point,
        )

    # We need to list all the variables names to make this look like a PlotterProtocol
    # pylint: disable=unused-argument
    def plot(
        self,
        embedding_result: EmbedResult,
        data: np.ndarray | None = None,
        y: pd.DataFrame | pd.Series | np.ndarray | range | None = None,
        ctx: EmbedContext | None = None,
        ax: Axes | None = None,
    ) -> Axes | None:
        """Plot a scatterplot of the random pair distances for the embedded data."""
        vec = self.rpc().evaluatev(data, embedding_result["coords"], ctx)
        plot = sns.scatterplot(x=vec[0], y=vec[1], s=1, alpha=0.5)
        plot.set(title=str(self))
        plot.set_xlabel("ambient distances")
        plot.set_ylabel("embedded distances")
        plt.show()

    def requires(self) -> dict:
        """Return the requirements for the RandomPairDistanceScatterplot object."""
        return self.rpc().requires()

    def __str__(self):
        return str(self.rpc())
