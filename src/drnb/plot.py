import math
from dataclasses import dataclass
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import drnb.neighbors.hubness as hub
from drnb.embed import get_coords
from drnb.eval.nbrpres import NbrPreservationEval
from drnb.eval.rte import RandomTripletEval
from drnb.io import read_pickle
from drnb.log import log
from drnb.util import islisty


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
            f"ko{self.n_neighbors}"
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
            f"so{self.n_neighbors}"
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
        cex = 100.0 * (10.0 ** (0.4591008 - 0.3722813 * math.log10(coords.shape[0])))

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
        )
        if sm is not None:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
            ax.figure.colorbar(sm)

    def get_palette(self, ctx):
        if ctx is None:
            return None
        try:
            return read_pickle(
                ctx.dataset_name,
                suffix="target-palette",
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


def create_plotters(plot=True, plot_kwargs=None):
    if plot_kwargs is None:
        plot_kwargs = {}
    if isinstance(plot, dict):
        plot_kwargs = plot
        plot = True

    color_by = plot_kwargs.get("color_by")
    if islisty(color_by):
        # default plot
        pkwargs1 = dict(plot_kwargs)
        del pkwargs1["color_by"]
        uniplotters = create_plotters(pkwargs1)

        # Add extra plots
        for cby in color_by:
            pkwargs1 = dict(plot_kwargs)
            pkwargs1["color_by"] = cby
            uniplotters += create_plotters(pkwargs1)
        return uniplotters

    if plot:
        plotter_cls = SeabornPlotter
    else:
        plotter_cls = NoPlotter
    if plot_kwargs is None:
        plot_kwargs = {}

    plotter = plotter_cls.new(**plot_kwargs)
    return [plotter]


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


def sns_embed_plot(
    coords,
    color_col=None,
    cex=10,
    alpha_scale=1,
    palette=None,
    title="",
    figsize=None,
    legend=True,
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
