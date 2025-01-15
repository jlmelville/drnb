from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from drnb.embed.context import EmbedContext
from drnb.eval.rte import RandomTripletEval
from drnb.types import EmbedResult


@dataclass
class RandomTripletHistogram:
    """A histogram of the random triplet accuracy for the embedded data.

    Attributes:
        n_triplets_per_point: The number of triplets to consider per point (default 5).
        normalize: Whether to normalize the random triplet accuracy (default True).
        metric: The metric to use when calculating the random triplet accuracy (default
            "euclidean").
        random_state: The random state to use when calculating the random triplet
            accuracy (default None).
    """

    n_triplets_per_point: int = 5
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

    # pylint: disable=unused-argument
    def plot(
        self,
        embedding_result: EmbedResult,
        data: np.ndarray | None = None,
        y: pd.DataFrame | pd.Series | np.ndarray | range | None = None,
        ctx: EmbedContext | None = None,
        ax: Axes | None = None,
    ) -> Axes | None:
        """Plot a histogram of the random triplet accuracy for the embedded data."""
        vec = self.rte().evaluatev(data, embedding_result["coords"], ctx)
        plot = sns.histplot(x=vec, bins=self.n_triplets_per_point + 1)
        plot.set(title=str(self))
        plot.set_xlabel("random triplet accuracy")
        plt.show()

    def requires(self) -> dict:
        """Return the requirements for the RandomTripletHistogram object."""
        return self.rte().requires()

    def __str__(self):
        return str(self.rte()) + f"{' norm' if self.normalize else ''}"
