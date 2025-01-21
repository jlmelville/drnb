from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.isotonic import IsotonicRegression

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
        cex: The size of the points in the scatterplot (default 1).
        alpha_scale: The alpha value to use for the points in the scatterplot (default
            0.5).
        show_isotonic: Whether to show the isotonic regression line (default False).
    """

    n_triplets_per_point: int = 5
    metric: str = "euclidean"
    random_state: int | None = None
    cex: int = 1
    alpha_scale: float = 0.5
    show_isotonic: bool = False

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
        ambient_distances = vec[0]
        embedded_distances = vec[1]
        plot = sns.scatterplot(
            x=ambient_distances,
            y=embedded_distances,
            s=self.cex,
            alpha=self.alpha_scale,
            edgecolor=None,
        )

        if self.show_isotonic:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(ambient_distances, embedded_distances)

            # Get prediction for sorted x values
            ambient_sorted = np.sort(ambient_distances)
            embedded_iso = iso.predict(ambient_sorted)

            # Plot the isotonic regression line
            plot.plot(
                ambient_sorted, embedded_iso, "r-", lw=1, label="Isotonic Regression"
            )
            plot.legend()

        plot.set(title=str(self))
        plot.set_xlabel("ambient distances")
        plot.set_ylabel("embedded distances")
        plt.show()

    def requires(self) -> dict:
        """Return the requirements for the RandomPairDistanceScatterplot object."""
        return self.rpc().requires()

    def __str__(self):
        return str(self.rpc())
