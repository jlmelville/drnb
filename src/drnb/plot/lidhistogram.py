from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from drnb.embed.context import EmbedContext, read_dataset_from_ctx
from drnb.plot.scale.lid import ColorByLid
from drnb.types import EmbedResult


@dataclass
class LidHistogram:
    """A histogram of Levina-Bickel local intrinsic dimensionality (LID)."""

    n_neighbors: int = 15
    metric: str = "euclidean"
    remove_self: bool = True
    eps: float = 1.0e-10
    knn_params: dict | None = field(default_factory=lambda: {"method": "sklearn"})

    def lid(self) -> ColorByLid:
        """Create a ColorByLid helper with matching parameters."""
        return ColorByLid(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            remove_self=self.remove_self,
            eps=self.eps,
            knn_params=self.knn_params,
        )

    # pylint: disable=unused-argument
    def plot(
        self,
        embedding_result: EmbedResult,
        data: np.ndarray | None = None,
        y: pd.DataFrame | pd.Series | np.ndarray | range | None = None,
        ctx: EmbedContext | None = None,
        __: Axes | None = None,
    ) -> Axes | None:
        """Plot a histogram of local intrinsic dimensionality values."""
        lid_fn = self.lid()
        if data is None:
            if ctx is None:
                raise ValueError("Must provide data or ctx to compute LID histogram")
            data, _ = read_dataset_from_ctx(ctx)
        vec = lid_fn(np.asarray(data), None, embedding_result["coords"], ctx)
        plot = sns.histplot(x=vec, bins="auto")
        plot.set(title=str(self))
        plot.set_xlabel("local intrinsic dimensionality")
        plt.show()

    def requires(self) -> dict:
        """Return the requirements for the LidHistogram object."""
        return {
            "name": "neighbors",
            "metric": self.metric,
            "n_neighbors": self.n_neighbors,
        }

    def __str__(self) -> str:
        return str(self.lid())
