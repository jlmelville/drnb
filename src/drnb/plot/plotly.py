import math
from dataclasses import dataclass
from typing import Any, Callable, List, Self, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn.decomposition
from IPython.display import display
from matplotlib.axes import Axes

from drnb.embed import get_coords
from drnb.embed.context import (
    EmbedContext,
    get_neighbors_with_ctx,
    read_dataset_from_ctx,
)
from drnb.io.dataset import read_palette
from drnb.neighbors.nbrinfo import NearestNeighbors
from drnb.plot.palette import palettize
from drnb.types import EmbedResult


def plotly_embed_plot(
    coords: np.ndarray,
    color_col: pd.DataFrame | pd.Series | np.ndarray | None = None,
    cex: float = 10.0,
    alpha_scale: float = 1.0,
    palette: dict | str | None = None,
    title: str = "",
    figsize: Tuple[float, float] | None = None,
    legend: bool = True,
    pc_axes: bool = False,
    flipx: bool = False,
    flipy: bool = False,
    hover: pd.DataFrame | None = None,
) -> go.Figure:
    """Create a Plotly scatter plot of the embedded data.

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
        hover: The hover data for the plot.
    """
    scatter_kwargs = {}

    color_col_name = None
    if color_col is None:
        color_col = list(range(coords.shape[0]))
        legend = False
    if isinstance(color_col, pd.DataFrame):
        if isinstance(palette, dict) and color_col.columns[-1] in palette:
            palette = palette[color_col.columns[-1]]
        # with a dataframe color col, the palette must be a dict mapping from
        # the color column name to another dict which maps the category "levels" to
        # colors, e.g. dict(smoker=dict(yes="red", no="blue"),
        #                   time=dict(Lunch="green", Dinner="red"))
        # always pick the last column in the dataframe as the color column
        color_col = color_col.iloc[:, -1]

    if hasattr(color_col, "name"):
        color_col_name = str(color_col.name)

    if isinstance(color_col, pd.Series):
        if pd.api.types.is_integer_dtype(color_col):
            color_col = color_col.astype("category")
        else:
            # series -> numpy array
            color_col = color_col.values

    palette = palettize(color_col, palette)
    if palette is not None:
        if isinstance(palette, dict):
            scatter_kwargs["color_discrete_map"] = palette
        else:
            scatter_kwargs["color_discrete_sequence"] = palette

    if pc_axes:
        coords = sklearn.decomposition.PCA(n_components=2).fit_transform(coords)
    if flipx:
        coords[:, 0] *= -1
    if flipy:
        coords[:, 1] *= -1

    if figsize is None:
        figsize = (6.5, 5)

    # as we don't pass a dataframe, the color column is internally called "color"
    # override that on the legend with the actual name
    if color_col_name is not None and legend:
        scatter_kwargs["labels"] = {"color": color_col_name}

    # use any category ordering rather than data ordering for the legend/colors
    if pd.api.types.is_categorical_dtype(color_col):
        if isinstance(color_col, pd.Categorical):
            cats = color_col.categories
        else:
            cats = color_col.cat.categories
        scatter_kwargs["category_orders"] = {"color": cats.tolist()}

    if title is None:
        title = ""

    # I was unable to find a way to get plotly to take an arbitrary vector of hover_data
    # so we must bind everything into a dataframe
    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "color": color_col,
            "idx": pd.Series(list(range(coords.shape[0])), name="idx"),
            "index": hover.index,
        }
    )

    # idx MUST BE the first element of the custom data for clickability
    hover_data = ["idx", "index"]
    if hover is not None:
        hover_data += list(hover.columns)
        df = pd.concat(
            [
                df,
                hover.reset_index(drop=True),
            ],
            axis=1,
        )
    scatter_kwargs["hover_data"] = hover_data

    # pylint:disable=no-member
    plot = (
        px.scatter(
            df,
            x="x",
            y="y",
            color="color",
            title=str(title),
            width=figsize[0] * 100,
            height=figsize[1] * 100,
            **scatter_kwargs,
        )
        .update_traces(marker={"size": cex, "opacity": alpha_scale})
        .update_layout(
            showlegend=legend,
            coloraxis_showscale=legend,
            plot_bgcolor="rgba(0, 0, 0, 0)",
        )
        .update_xaxes(showline=True, linecolor="black", mirror=True)
        .update_yaxes(showline=True, linecolor="black", mirror=True)
    )

    return plot


# pylint: disable=too-many-instance-attributes
@dataclass
class PlotlyPlotter:
    """A Plotly plotter object.

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
        hover: The names of the columns in the target data to use as part of the hover
            data (default None).
        renderer: The renderer to use for the plot (default "jupyterlab").
        clickable: Whether to make the plot clickable (default False).
        clickable_n_neighbors: The number of neighbors to consider for clickability
            (default 15).
        clickable_metric: The metric to use for clickability (default "euclidean").
    """

    cex: float | None = None
    alpha_scale: float | None = None
    title: str | None = None
    figsize: Tuple[float, float] = None
    legend: bool = True
    palette: dict | str | None = None
    color_by: Any | None = None
    vmin: float | None = None
    vmax: float | None = None
    pc_axes: bool = False
    flipx: bool = False
    flipy: bool = False
    hover: List[str] | None = None
    renderer: str = "jupyterlab"
    clickable: bool = False
    clickable_n_neighbors: int = 15
    clickable_metric: str = "euclidean"

    @classmethod
    def new(cls, **kwargs) -> Self:
        """Create a new PlotlyPlotter object."""
        return cls(**kwargs)

    def plot(
        self,
        embedding_result: EmbedResult,
        data: np.ndarray | None = None,
        y: pd.DataFrame | pd.Series | np.ndarray | range | None = None,
        ctx: EmbedContext | None = None,
        _: Axes | None = None,
    ) -> None:
        """Plot the embedded data."""
        if data is None:
            if ctx is None:
                raise ValueError("Must provide data")
            dataset = read_dataset_from_ctx(ctx)
            data = dataset[0]
            if y is None:
                y = dataset[1]

        coords = get_coords(embedding_result)

        title = self.title
        palette = self.palette
        if palette is None:
            palette = self.get_palette(ctx)
        # Setting the palette to "False" means to force the palette off so you get
        # the glasbey colors (e.g. you think the original dataset palette is bad)
        if not palette:
            palette = None
        if isinstance(self.color_by, Callable):
            y = self.color_by(data, y, coords, ctx)

            # name the color bar after the metric name
            # as it can be a bit long, stop after the first "-" or " " encountered
            # (or 5 characters if no such character exists)
            yname = str(self.color_by)
            yname_break = len(yname)
            for ch in [" ", "-"]:
                br = yname.find(ch)
                if br != -1 and br < yname_break:
                    yname_break = br
            if yname_break == -1:
                yname_break = 5
            yname = yname[:yname_break]

            y = pd.Series(y, name=yname)
            if hasattr(self.color_by, "scale") and self.color_by.scale is not None:
                palette = self.color_by.scale.palette
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

        hover = None
        if self.hover is not None:
            if isinstance(y, pd.DataFrame):
                hover = y.loc[:, self.hover]
                hover.index = y.index

        fig = plotly_embed_plot(
            coords,
            color_col=y,
            cex=math.sqrt(cex),
            alpha_scale=alpha_scale,
            palette=palette,
            title=title,
            figsize=self.figsize,
            legend=self.legend,
            pc_axes=self.pc_axes,
            flipx=self.flipx,
            flipy=self.flipy,
            hover=hover,
        )

        if self.clickable:
            click_fig = self.make_clickable(fig, ctx)
            display(click_fig)
        else:
            # if jupyterlab-plotly extension is not installed, try one of:
            #  colab, iframe, iframe-connected, sphinx-gallery
            fig.show(renderer=self.renderer)

    def make_clickable(
        self, fig: go.Figure, ctx: EmbedContext | None
    ) -> go.FigureWidget | go.Figure:
        """Make the PlotlyPlotter object clickable."""
        if ctx is None:
            return fig
        knn = get_neighbors_with_ctx(
            data=None,
            metric=self.clickable_metric,
            n_neighbors=self.clickable_n_neighbors + 1,
            return_distance=False,
            ctx=ctx,
        )
        knn = knn.idx[:, 1:]

        return clickable_neighbors(fig, knn)

    def get_palette(self, ctx: EmbedContext | None) -> dict | None:
        """Get the palette for the PlotlyPlotter object, if it exists."""
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
        """Return the requirements for the PlotlyPlotter object."""
        if self.color_by is not None and hasattr(self.color_by, "requires"):
            return self.color_by.requires()
        return {}


# Setting a click-handler is made a lot harder by the points being split by category
# Perhaps there is a way to map back to the original position of the points that I
# don't know about. For now, this assumes that custom data was passed to the plot
# creation and the original index into the data is the first element of custom data
# for each point. Then a dict is built mapping from that original index to a tuple of
# the trace index and then index within that trace of the point. This allows to find
# the location of a point based on its original index into the plot data location
def clickable_neighbors(plot: go.Figure, nn: NearestNeighbors) -> go.FigureWidget:
    """Make a Plotly plot display the nearest neighbors when clicked."""
    f = go.FigureWidget(plot)

    selected_symbol = "diamond"
    unselected_symbol = f.data[0].marker.symbol
    selected_size = f.data[0].marker.size * 1.5

    normal_size = f.data[0].marker.size
    nbr_size = f.data[0].marker.size * 2

    # keep track of if a point has been clicked so we know whether to set or unset the
    # formatting
    clicked = [None] * len(f.data)

    # the size and symbols in the marker are scalars, but we are going to selectively
    # change some. This requires storing the full array of each property
    sizes = [None] * len(f.data)
    symbols = [None] * len(f.data)

    # dict to store a mapping from original index to (trace_idx, new_idx)
    plot_idx = {}

    for i, datum in enumerate(f.data):
        datum.marker.size = [datum.marker.size] * len(datum.x)
        clicked[i] = [False] * len(datum.x)
        sizes[i] = list(datum.marker.size)
        symbols[i] = [unselected_symbol] * len(datum.x)
        # assumes that the first element of each point's custom data is its index in
        # the original data
        for j, idx in enumerate(datum.customdata):
            plot_idx[idx[0]] = (i, j)

    # pylint:disable=unused-argument
    def highlight_nbrs(selector, points, trace):
        if not points.point_inds:
            return

        trace_index = points.trace_index
        idx = points.point_inds[0]
        # again, the 0 index is assuming that the first element of the custom data is
        # the old index
        old_idx = f.data[trace_index].customdata[idx][0]
        idx_nbrs = nn[old_idx]

        if clicked[trace_index][idx]:
            new_size = normal_size
            new_symbol = unselected_symbol
            click_size = normal_size
        else:
            new_size = nbr_size
            new_symbol = selected_symbol
            click_size = selected_size
        clicked[trace_index][idx] = not clicked[trace_index][idx]
        symbols[trace_index][idx] = new_symbol
        sizes[trace_index][idx] = click_size

        for j in idx_nbrs:
            nbr_trace_idx, nbr_new_idx = plot_idx[j]
            sizes[nbr_trace_idx][nbr_new_idx] = new_size

        with f.batch_update():
            for i, datum in enumerate(f.data):
                datum.marker.size = sizes[i]
            f.data[trace_index].marker.symbol = symbols[trace_index]

    for datum in f.data:
        datum.on_click(highlight_nbrs)

    return f


def plotly_result_plot(embed_result: dict, **kwargs):
    """Plot the result of an embedding using Plotly."""
    PlotlyPlotter(**kwargs).plot(embed_result["coords"], ctx=embed_result["context"])
