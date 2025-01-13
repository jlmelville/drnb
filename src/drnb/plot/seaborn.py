import math
from dataclasses import dataclass
from typing import Any, Callable, Self, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.decomposition
from matplotlib.axes import Axes
from matplotlib.ticker import AutoLocator

from drnb.embed import get_coords
from drnb.embed.context import EmbedContext, read_dataset_from_ctx
from drnb.io.dataset import read_palette
from drnb.log import log
from drnb.plot.palette import palettize
from drnb.plot.util import hex_to_rgb, is_hex


# pylint:disable=too-many-statements
def sns_embed_plot(
    coords: np.ndarray,
    color_col: pd.DataFrame | pd.Series | np.ndarray | range | None = None,
    cex: float = 10.0,
    alpha_scale: float = 1.0,
    palette: dict | str | None = None,
    title: str = "",
    figsize: Tuple[float, float] | None = None,
    legend: bool = True,
    pc_axes: bool = False,
    flipx: bool = False,
    flipy: bool = False,
    show_axes: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Create a Seaborn scatter plot of the embedded data.

    Args:
        coords: The embedded data.
        color_col: The color column for the plot.
        cex: The size of the points in the plot.
        alpha_scale: The alpha scale of the points in the plot.
        palette: The palette to use for the plot.
        title: The title of the plot.
        figsize: The size of the figure.
        legend: Whether to show the legend.
        pc_axes: Whether to use principal component axes.
        flipx: Whether to flip the x-axis.
        flipy: Whether to flip the y-axis.
        ax: The axes to use for the plot.
    """
    if title is None:
        title = ""
    scatter_kwargs = {}

    if color_col is None:
        color_col = range(coords.shape[0])
    if isinstance(color_col, pd.DataFrame):
        if isinstance(palette, dict) and color_col.columns[-1] in palette:
            palette = palette[color_col.columns[-1]]
        # with a dataframe color col, the palette must be a dict mapping from
        # the color column name to another dict which maps the category "levels" to
        # colors, e.g. dict(smoker=dict(yes="red", no="blue"),
        #                   time=dict(Lunch="green", Dinner="red"))
        # always pick the last column in the dataframe as the color column
        color_col = color_col.iloc[:, -1]

    # Let's make the executive decision that target data needs to be explicitly set
    # as categorical when created to be treated as such during plotting:
    # this simplifies the dataset interface (target data is always a pandas dataframe)
    # and plain integers are treated as a basic index
    # Add hue_order if categorical
    if isinstance(color_col, pd.Series) and pd.api.types.is_categorical_dtype(
        color_col
    ):
        scatter_kwargs["hue_order"] = color_col.cat.categories

    # If the column contains manually-set hex codes, use them directly, no palette or
    # legend possible though
    if is_hex(color_col):
        scatter_kwargs = {
            "c": np.array([hex_to_rgb(hexcode, scale=True) for hexcode in color_col])
        }
        legend = False
    else:
        scatter_kwargs = {"hue": color_col}

    # At this point color_col should be one of: a range, a numpy array, a pandas series
    palette = palettize(color_col, palette)

    if palette is not None:
        scatter_kwargs["palette"] = palette
    else:
        scatter_kwargs["palette"] = "viridis"
    if ax is not None:
        scatter_kwargs["ax"] = ax
        legend = False
    else:
        if figsize is None:
            figsize = (6, 4)
        plt.figure(figsize=figsize)

    force_legend = isinstance(legend, str) and legend == "force"
    nlegcol = 0
    if legend:
        if isinstance(color_col, range):
            nlegcol = 1
        else:
            if isinstance(color_col, np.ndarray):
                nclasses = len(np.unique(color_col))
            else:
                nclasses = color_col.nunique()
            nlegcol = int(nclasses / (figsize[1] * 4)) + 1
        if nlegcol > 1:
            if legend and not force_legend:
                log.info("Not showing large legend")
            legend = force_legend
        elif force_legend:
            # in this case forced legend was unnecessary
            legend = True

    if pc_axes:
        coords = sklearn.decomposition.PCA(n_components=2).fit_transform(coords)
    if flipx:
        coords[:, 0] *= -1
    if flipy:
        coords[:, 1] *= -1

    plot = sns.scatterplot(
        x=coords[:, 0],
        y=coords[:, 1],
        s=cex,
        alpha=alpha_scale,
        legend=legend,
        **scatter_kwargs,
    )
    plot.set_title(title)

    # Configure axes, axis labels, and frame
    if not show_axes:
        plot.set_axis_off()
        plot.set_xticks([])
        plot.set_yticks([])
        for spine in plot.spines.values():
            spine.set_visible(False)
    else:
        plot.xaxis.set_major_locator(AutoLocator())
        plot.yaxis.set_major_locator(AutoLocator())

    if legend and (nlegcol == 1 or force_legend):
        leg_title = None
        if hasattr(color_col, "name"):
            leg_title = color_col.name
        if leg_title == "V1":
            leg_title = None
        handles, labels = plot.get_legend_handles_labels()
        for handle in handles:
            handle.set_alpha(1.0)
            if hasattr(handle, "set_sizes"):
                handle.set_sizes([100])
            else:
                handle.set_markersize(10)
        plt.legend(
            handles,
            labels,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            ncol=nlegcol,
            title=leg_title,
        )

    return plot


@dataclass
# pylint: disable=too-many-instance-attributes
class SeabornPlotter:
    """A Seaborn plotter object.

    Attributes:
        cex: The size of the points in the plot (default None).
        alpha_scale: The alpha scale of the points in the plot (default None).
        title: The title of the plot (default None).
        figsize: The size of the figure (default None).
        legend: Whether to show the legend (default True).
        palette: The palette to use for the plot (default None).
        color_by: The color_by function to use for the plot (default None).
        vmin: The minimum value for the color scale (default None).
        vmax: The maximum value for the color scale (default None).
        pc_axes: Whether to use principal component axes (default False).
        flipx: Whether to flip the x-axis (default False).
        flipy: Whether to flip the y-axis (default False).
        show_axes: Whether to show the axes (default False).
    """

    cex: int | None = None
    alpha_scale: float | None = None
    title: str | None = None
    figsize: Tuple[float, float] | None = None
    legend: bool = True
    palette: dict | str | None = None
    color_by: Any = None
    vmin: float | None = None
    vmax: float | None = None
    pc_axes: bool = False
    flipx: bool = False
    flipy: bool = False
    show_axes: bool = False

    @classmethod
    def new(cls, **kwargs) -> Self:
        """Create a new SeabornPlotter object."""
        return cls(**kwargs)

    def plot(
        self,
        embedded: tuple | dict | np.ndarray,
        data: np.ndarray | None = None,
        y: pd.DataFrame | pd.Series | np.ndarray | range | None = None,
        ctx: EmbedContext | None = None,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot the embedded data."""
        if data is None:
            if ctx is None:
                raise ValueError("Must provide data")
            dataset = read_dataset_from_ctx(ctx)
            data = dataset[0]
            if y is None:
                y = dataset[1]
        coords = get_coords(embedded)

        title = self.title
        palette = self.palette
        if palette is None:
            palette = self.get_palette(ctx)
        if not palette:
            palette = None

        # used for a color bar (if needed)
        scalar_mappable = None
        if isinstance(self.color_by, Callable):
            y = self.color_by(data, y, coords, ctx)
            if hasattr(self.color_by, "scale") and self.color_by.scale is not None:
                scalar_mappable = self.color_by.scale(y, self.vmin, self.vmax, palette)
                palette = self.color_by.scale.palette
                self.legend = False
                if title is None:
                    title = self.color_by

        # did a log-log plot of N vs the average 15-NN distance in the embedded space
        # multiplying the 15-NN distance by 100 gave a good-enough value for the
        # point size when figsize=(9, 6)
        if self.cex is None:
            cex = 100.0 * (
                10.0 ** (0.4591008 - 0.3722813 * math.log10(coords.shape[0]))
            )
        else:
            cex = self.cex

        if self.alpha_scale is None:
            estimated_cex = 10.0 ** (
                0.4591008 - 0.3722813 * math.log10(coords.shape[0])
            )
            alpha_scale = np.clip(estimated_cex * 2.0, 0.05, 0.8)
        else:
            alpha_scale = self.alpha_scale

        ax_out = sns_embed_plot(
            coords,
            color_col=y,
            cex=cex,
            alpha_scale=alpha_scale,
            palette=palette,
            title=title,
            figsize=self.figsize,
            legend=self.legend,
            pc_axes=self.pc_axes,
            flipx=self.flipx,
            flipy=self.flipy,
            ax=ax,
            show_axes=self.show_axes,
        )
        if ax is None:
            plt.show()
        if scalar_mappable is not None:
            if ax_out.get_legend() is not None:
                ax_out.get_legend().remove()
            ax_out.figure.colorbar(scalar_mappable)
        return ax_out

    def get_palette(self, ctx: EmbedContext | None) -> dict | None:
        """Get the palette for the SeabornPlotter object, if it exists."""
        if ctx is None:
            return None
        try:
            return read_palette(
                ctx.dataset_name,
                drnb_home=ctx.drnb_home,
                sub_dir=ctx.data_sub_dir,
                verbose=False,
            )
        except FileNotFoundError:
            return None

    def requires(self) -> dict:
        """Return the requirements for the SeabornPlotter object."""
        if self.color_by is not None and hasattr(self.color_by, "requires"):
            return self.color_by.requires()
        return {}


def sns_result_plot(embed_result: dict, *, ax=None, **kwargs):
    """Plot the result of an embedding using Seaborn."""
    SeabornPlotter(**kwargs).plot(
        embed_result["coords"], ctx=embed_result["context"], ax=ax
    )
