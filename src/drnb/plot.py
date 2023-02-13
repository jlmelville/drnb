import math
from dataclasses import dataclass
from typing import Any, Callable

import glasbey
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import sklearn.decomposition

import drnb.neighbors.hubness as hub
from drnb.embed import get_coords
from drnb.eval.nbrpres import NbrPreservationEval
from drnb.eval.rpc import RandomPairCorrelEval
from drnb.eval.rte import RandomTripletEval
from drnb.io.dataset import read_palette
from drnb.log import log
from drnb.util import evenly_spaced, get_method_and_args, islisty


class NoPlotter:
    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    # pylint: disable=unused-argument
    def plot(self, embedded, data, y, ctx=None):
        pass


@dataclass
class ColorScale:
    vmin: float = None
    vmax: float = None
    palette: Any = None

    @classmethod
    def new(cls, kwds):
        if kwds is None:
            return cls()
        return cls(**kwds)

    def __call__(self, y, vmin, vmax, palette):
        _vmin = self.vmin
        if _vmin is None:
            _vmin = vmin
        if _vmin is None:
            _vmin = y.min()

        _vmax = self.vmax
        if _vmax is None:
            _vmax = vmax
        if _vmax is None:
            _vmax = y.max()

        _palette = self.palette
        if _palette is None:
            _palette = palette

        norm = plt.Normalize(_vmin, _vmax)
        sm = plt.cm.ScalarMappable(cmap=_palette, norm=norm)
        sm.set_array([])

        return sm


@dataclass
class ColorByKo:
    n_neighbors: int = 15
    scale: ColorScale = ColorScale()
    normalize: bool = True
    log1p: bool = False

    # pylint: disable=unused-argument
    def __call__(self, data, target, coords, ctx=None):
        res = np.array(hub.fetch_nbr_stats(ctx.dataset_name, self.n_neighbors)["ko"])
        if self.normalize:
            res = res / self.n_neighbors
        if self.log1p:
            return np.log1p(res)
        return res

    def __str__(self):
        return (
            f"ko-{self.n_neighbors}"
            f"{' log' if self.log1p else ''}{' norm' if self.normalize else ''}"
        )


@dataclass
class ColorBySo:
    n_neighbors: int = 15
    scale: ColorScale = ColorScale()
    normalize: bool = True
    log1p: bool = False

    # pylint: disable=unused-argument
    def __call__(self, data, target, coords, ctx=None):
        res = np.array(hub.fetch_nbr_stats(ctx.dataset_name, self.n_neighbors)["so"])
        if self.normalize:
            res = res / self.n_neighbors
        if self.log1p:
            return np.log1p(res)
        return res

    def __str__(self):
        return (
            f"so-{self.n_neighbors}"
            f"{' log' if self.log1p else ''}{' norm' if self.normalize else ''}"
        )


@dataclass
class ColorByNbrPres:
    n_neighbors: int = 15
    scale: ColorScale = ColorScale()
    normalize: bool = True
    metric: str = "euclidean"

    def npe(self):
        return NbrPreservationEval(
            metric=self.metric, n_neighbors=self.n_neighbors, include_self=False
        )

    # pylint: disable=unused-argument
    def __call__(self, data, target, coords, ctx=None):
        return self.npe().evaluatev(data, coords, ctx=ctx)[0]

    def __str__(self):
        return str(self.npe()) + f"{' norm' if self.normalize else ''}"

    def requires(self):
        return self.npe().requires()


@dataclass
class ColorByRte:
    n_triplets_per_point: int = 5
    scale: ColorScale = ColorScale()
    normalize: bool = True
    metric: str = "euclidean"
    random_state: int = None

    def rte(self):
        return RandomTripletEval(
            random_state=self.random_state,
            metric=self.metric,
            n_triplets_per_point=self.n_triplets_per_point,
        )

    # pylint: disable=unused-argument
    def __call__(self, data, target, coords, ctx=None):
        return self.rte().evaluatev(data, coords, ctx)

    def __str__(self):
        return str(self.rte()) + f"{' norm' if self.normalize else ''}"

    def requires(self):
        return self.rte().requires()


@dataclass
class SeabornPlotter:
    cex: int = None
    alpha_scale: float = None
    title: str = None
    figsize: tuple = None
    legend: bool = True
    palette: Any = None
    color_by: Any = None
    vmin: float = None
    vmax: float = None
    pc_axes: bool = False
    flipx: bool = False
    flipy: bool = False

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def plot(self, embedded, data, y, ctx=None):
        coords = get_coords(embedded)

        title = self.title
        palette = self.palette
        if palette is None:
            palette = self.get_palette(ctx)
        sm = None
        if isinstance(self.color_by, Callable):
            y = self.color_by(data, y, coords, ctx)
            if hasattr(self.color_by, "scale") and self.color_by.scale is not None:
                sm = self.color_by.scale(y, self.vmin, self.vmax, palette)
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

        ax = sns_embed_plot(
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
        )
        if sm is not None:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            ax.figure.colorbar(sm)
        plt.show()

    def get_palette(self, ctx):
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

    def requires(self):
        reqs = []
        if self.color_by is not None:
            if hasattr(self.color_by, "requires"):
                reqs.append(self.color_by.requires())
        return reqs


@dataclass
class NbrPreservationHistogram:
    n_neighbors: int = 15
    metric: str = "euclidean"
    normalize: bool = True

    def npe(self):
        return NbrPreservationEval(
            metric=self.metric, n_neighbors=self.n_neighbors, include_self=False
        )

    def requires(self):
        return self.npe().requires()

    # pylint: disable=unused-argument
    def plot(self, embedded, data, y, ctx=None):
        vec = self.npe().evaluatev(data, embedded["coords"], ctx=ctx)[0]
        plot = sns.histplot(x=vec, bins=self.n_neighbors + 1)
        plot.set(title=str(self))
        plot.set_xlabel("neighbor preservation")
        plt.show()

    def __str__(self):
        return str(self.npe()) + f"{' norm' if self.normalize else ''}"


@dataclass
class RandomTripletHistogram:
    n_triplets_per_point: int = 5
    normalize: bool = True
    metric: str = "euclidean"
    random_state: int = None

    def rte(self):
        return RandomTripletEval(
            random_state=self.random_state,
            metric=self.metric,
            n_triplets_per_point=self.n_triplets_per_point,
        )

    # pylint: disable=unused-argument
    def plot(self, embedded, data, y, ctx=None):
        vec = self.rte().evaluatev(data, embedded["coords"], ctx)
        plot = sns.histplot(x=vec, bins=self.n_triplets_per_point + 1)
        plot.set(title=str(self))
        plot.set_xlabel("random triplet accuracy")
        plt.show()

    def __str__(self):
        return str(self.rte()) + f"{' norm' if self.normalize else ''}"

    def requires(self):
        return self.rte().requires()


@dataclass
class RandomPairDistanceScatterplot:
    n_triplets_per_point: int = 5
    metric: str = "euclidean"
    random_state: int = None

    def rpc(self):
        return RandomPairCorrelEval(
            random_state=self.random_state,
            metric=self.metric,
            n_triplets_per_point=self.n_triplets_per_point,
        )

    # pylint: disable=unused-argument
    def plot(self, embedded, data, y, ctx=None):
        vec = self.rpc().evaluatev(data, embedded["coords"], ctx)
        plot = sns.scatterplot(x=vec[0], y=vec[1], s=1, alpha=0.5)
        plot.set(title=str(self))
        plot.set_xlabel("ambient distances")
        plot.set_ylabel("embedded distances")
        plt.show()

    def __str__(self):
        return str(self.rpc())

    def requires(self):
        return self.rpc().requires()


def create_plotters(plot=True, plot_kwargs=None):
    plotter_cls = SeabornPlotter

    if plot_kwargs is None:
        plot_kwargs = {}
    if isinstance(plot, dict):
        plot_kwargs = plot
        plot = True
    if "plot" in plot_kwargs:
        plot = plot_kwargs["plot"]
        del plot_kwargs["plot"]

    if isinstance(plot, str):
        if plot == "seaborn":
            plotter_cls = SeabornPlotter
        elif plot == "plotly":
            plotter_cls = PlotlyPlotter
        else:
            raise ValueError(f"Unknown plot type {plot}")
        plot = True
    if not plot:
        return []

    plotters = []

    color_by = []
    if "color_by" in plot_kwargs:
        color_by = plot_kwargs["color_by"]
        if not islisty(color_by):
            color_by = [color_by]
        del plot_kwargs["color_by"]

    extras = []
    if "extras" in plot_kwargs:
        extras = plot_kwargs["extras"]
        del plot_kwargs["extras"]
    plotters.append(plotter_cls.new(**plot_kwargs))
    for cby in color_by:
        pkwargs1 = dict(plot_kwargs)
        pkwargs1["color_by"] = cby
        plotters.append(plotter_cls.new(**pkwargs1))

    for extra in extras:
        extra, extra_kwds = get_method_and_args(extra, {})

        if extra == "nnphist":
            plotters.append(NbrPreservationHistogram(**extra_kwds))
        elif extra == "rthist":
            plotters.append(RandomTripletHistogram(**extra_kwds))
        elif extra == "rpscatter":
            plotters.append(RandomPairDistanceScatterplot(**extra_kwds))
        else:
            raise ValueError(f"Unknown plot type '{extra}'")
    return plotters


def plot_embedded(embedded, y, plot_kwargs=None):
    if isinstance(embedded, tuple):
        coords = embedded[0]
    else:
        coords = embedded
    if plot_kwargs is None:
        plot_kwargs = {}
    sns_embed_plot(coords, y, **plot_kwargs)


# https://stackoverflow.com/a/67001213
def is_string_series(s: pd.Series):
    if isinstance(s.dtype, pd.StringDtype):
        # The series was explicitly created as a string series (Pandas>=1.0.0)
        return True
    if s.dtype == "object":
        # Object series, check each value
        return all((v is None) or isinstance(v, str) for v in s)
    return False


def hex_to_rgb(hexcode, scale=False):
    result = tuple(int(hexcode[i : i + 2], 16) for i in (1, 3, 5))
    if scale:
        result = tuple(x / 255 for x in result)
    return result


def rgb_to_hex(rgb):
    return matplotlib.colors.to_hex(rgb)


def is_hex(col):
    if islisty(col):
        col = pd.Series(col)
    if not isinstance(col, pd.Series):
        return False
    if not is_string_series(col):
        return False
    return np.all(
        col.str.startswith("#")
        & (col.str.len() == 7)
        & (col.str.slice(1).str.match(r"[A-Fa-f0-9]"))
    )


# use glasbey to extend a categorical palette if possible (or necessary)
def palettize(color_col, palette=None):
    # return early if the palette maps from category level to color
    if isinstance(palette, dict):
        return palette
    n_categories = None
    if pd.api.types.is_categorical_dtype(color_col):
        if hasattr(color_col, "categories"):
            n_categories = color_col.categories.nunique()
        else:
            n_categories = color_col.nunique()
    else:
        # if this isn't a categorical color column do nothing
        return palette

    if palette is None:
        # return a suitably sized categorical palette
        return glasbey.create_palette(n_categories)

    # a named or pre-defined palette, so check we have enough colors
    if isinstance(palette, str):
        palette = sns.color_palette(palette)
    # pylint: disable=protected-access
    if not (isinstance(palette, sns.palettes._ColorPalette) or islisty(palette)):
        raise ValueError(f"Unknown palette {palette}")

    n_colors = len(palette)
    if n_categories is not None:
        if n_colors < n_categories:
            palette = glasbey.extend_palette(palette, n_categories)
        elif n_colors > n_categories:
            palette = evenly_spaced(palette, n_categories)
        # if n_colors == n_categories then use the palette as-is

        # glasbey returns hex codes, but seaborn does not
        if not isinstance(palette[0], str):
            palette = [rgb_to_hex(color) for color in palette]
    return palette


# pylint:disable=too-many-statements
def sns_embed_plot(
    coords,
    color_col=None,
    cex=10,
    alpha_scale=1,
    palette=None,
    title="",
    figsize=None,
    legend=True,
    pc_axes=False,
    flipx=False,
    flipy=False,
):
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

    if isinstance(color_col, pd.Series) and pd.api.types.is_integer_dtype(color_col):
        color_col = color_col.astype("category")
    if is_hex(color_col):
        scatter_kwargs = {"c": np.array([hex_to_rgb(h, True) for h in color_col])}
        legend = False
    else:
        scatter_kwargs = {"hue": color_col}

    palette = palettize(color_col, palette)
    if palette is not None:
        scatter_kwargs["palette"] = palette

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

    if legend and (nlegcol == 1 or force_legend):
        leg_title = None
        if hasattr(color_col, "name"):
            leg_title = color_col.name
        if leg_title == "V1":
            leg_title = None
        plt.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            ncol=nlegcol,
            title=leg_title,
        )
        # plt.tight_layout()
    return plot


def plotly_embed_plot(
    coords,
    color_col=None,
    cex=10,
    alpha_scale=1,
    palette=None,
    title="",
    figsize=None,
    legend=True,
    pc_axes=False,
    flipx=False,
    flipy=False,
):
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
    pio.renderers.default = "colab"

    # as we don't pass a dataframe, the color column is internally called "color"
    # override that on the legend with the actual name
    if color_col_name is not None and legend:
        scatter_kwargs["labels"] = dict(color=color_col_name)

    # use any category ordering rather than data ordering for the legend/colors
    if pd.api.types.is_categorical_dtype(color_col):
        if isinstance(color_col, pd.Categorical):
            cats = color_col.categories
        else:
            cats = color_col.cat.categories
        scatter_kwargs["category_orders"] = dict(color=cats.tolist())

    if title is None:
        title = ""

    # pylint:disable=no-member
    plot = (
        px.scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            color=color_col,
            title=str(title),
            width=figsize[0] * 100,
            height=figsize[1] * 100,
            **scatter_kwargs,
        )
        .update_traces(marker=dict(size=cex, opacity=alpha_scale))
        .update_layout(
            showlegend=legend,
            coloraxis_showscale=legend,
            plot_bgcolor="rgba(0, 0, 0, 0)",
        )
        .update_xaxes(showline=True, linecolor="black", mirror=True)
        .update_yaxes(showline=True, linecolor="black", mirror=True)
    )

    return plot


@dataclass
class PlotlyPlotter:
    cex: int = None
    alpha_scale: float = None
    title: str = None
    figsize: tuple = None
    legend: bool = True
    palette: Any = None
    color_by: Any = None
    vmin: float = None
    vmax: float = None
    pc_axes: bool = False
    flipx: bool = False
    flipy: bool = False

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def plot(self, embedded, data, y, ctx=None):
        coords = get_coords(embedded)

        title = self.title
        palette = self.palette
        if palette is None:
            palette = self.get_palette(ctx)
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
        )
        fig.show()

    def get_palette(self, ctx):
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

    def requires(self):
        reqs = []
        if self.color_by is not None:
            if hasattr(self.color_by, "requires"):
                reqs.append(self.color_by.requires())
        return reqs
