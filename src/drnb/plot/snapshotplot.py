from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from drnb.embed.context import EmbedContext
from drnb.log import log
from drnb.plot.align import kabsch_best_align
from drnb.plot.common import result_plot
from drnb.types import EmbedResult


@dataclass
class EmbeddingSnapshotPlot:
    """Plot the evolution of an embedding through its snapshots.

    Attributes:
        figsize: The size of each subplot as (width, height) in inches.
        n_cols: Number of columns in the grid layout. Rows will be calculated
        automatically.
        align: Whether to align snapshots using Kabsch alignment to the final embedding.
        cex: The size of the points in the scatterplots.
        alpha_scale: The alpha value for the points in the scatterplots.
    """

    figsize: tuple[float, float] = (8, 6)
    n_cols: int = 1
    align: bool = True
    cex: int = 1
    alpha_scale: float = 0.5

    def plot(
        self,
        embedding_result: EmbedResult,
        data: np.ndarray | None = None,
        y: np.ndarray | None = None,
        ctx: EmbedContext | None = None,
        ax: Axes | None = None,
    ) -> Axes | None:
        """Plot snapshots of the embedding evolution.

        If snapshots are not present in the embedding_result, returns None.
        """
        if "snapshots" not in embedding_result:
            log.warning("No snapshots found in embedding result")
            return None

        if ctx is None:
            ctx = embedding_result["context"]

        snapshots = embedding_result["snapshots"]
        n_snapshots = len(snapshots)
        n_rows = (n_snapshots + self.n_cols - 1) // self.n_cols

        # Calculate total figure size based on subplot size
        total_figsize = (
            self.figsize[0] * self.n_cols,
            self.figsize[1] * n_rows,
        )

        _, axes = plt.subplots(
            nrows=n_rows,
            ncols=self.n_cols,
            figsize=total_figsize,
            squeeze=False,
        )

        # Get reference coords for alignment if needed
        ref_coords = embedding_result["coords"] if self.align else None

        # Sort snapshot keys to ensure chronological order
        snapshot_keys = sorted(
            snapshots.keys(),
            key=lambda x: int(x.split("_")[1]),
        )

        for idx, key in enumerate(snapshot_keys):
            row = idx // self.n_cols
            col = idx % self.n_cols
            coords = snapshots[key]

            if self.align and ref_coords is not None:
                coords = kabsch_best_align(ref_coords, coords)

            ax = axes[row, col]
            snapshot_result = {"coords": coords, "context": ctx}

            result_plot(
                snapshot_result,
                ax=ax,
                title=f"Iteration {key.split('_')[1]}",
                cex=self.cex,
                alpha_scale=self.alpha_scale,
            )

        # Hide empty subplots
        for idx in range(n_snapshots, n_rows * self.n_cols):
            row = idx // self.n_cols
            col = idx % self.n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        return axes

    def requires(self) -> dict:
        """Return the requirements for the EmbeddingSnapshotPlot."""
        return {}

    def __str__(self) -> str:
        return "Embedding Evolution Plot"


def plot_embedding_snapshots(
    embedding_result: EmbedResult,
    *,
    figsize: tuple[float, float] = (8, 6),
    n_cols: int = 1,
    align: bool = True,
    cex: int | None = None,
    alpha_scale: float | None = None,
) -> Axes | None:
    """Plot snapshots showing the evolution of an embedding.

    Args:
        embedding_result: The embedding result containing snapshots to plot.
        data: Optional input data array.
        y: Optional target/label data for coloring points.
        ctx: Optional embedding context containing metadata.
        figsize: Size of the figure as (width, height) in inches.
        n_cols: Number of columns in the grid layout.
        align: Whether to align snapshots to the final embedding.
        cex: Size of the points in the scatterplots.
        alpha_scale: Alpha value for the points.

    Returns:
        The matplotlib axes array if successful, None if no snapshots found.
    """
    plotter = EmbeddingSnapshotPlot(
        figsize=figsize,
        n_cols=n_cols,
        align=align,
        cex=cex,
        alpha_scale=alpha_scale,
    )
    return plotter.plot(
        embedding_result=embedding_result,
    )
