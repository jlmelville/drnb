from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from drnb.embed import get_coords
from drnb.io import read_pickle
from drnb.log import log


class NoPlotter:
    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def plot(self, embedded, y, ctx=None):
        pass


@dataclass
class SeabornPlotter:
    cex: int = 10
    alpha_scale: float = 1.0
    title: str = ""
    figsize: tuple = None
    legend: bool = True
    palette: Any = None

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    def plot(self, embedded, y, ctx=None):
        coords = get_coords(embedded)
        palette = self.get_palette(ctx)
        if palette is None:
            palette = self.palette
        sns_embed_plot(
            coords,
            color_col=y,
            cex=self.cex,
            alpha_scale=self.alpha_scale,
            palette=palette,
            title=self.title,
            figsize=self.figsize,
            legend=self.legend,
        )

    def get_palette(self, ctx):
        if ctx is None:
            return None
        try:
            return read_pickle(
                ctx.name,
                suffix="target-palette",
                data_path=ctx.data_path,
                sub_dir=ctx.data_sub_dir,
                verbose=True,
            )
        except FileNotFoundError:
            return None


def create_plotter(plot=True, plot_kwargs=None):
    if isinstance(plot, dict):
        plot_kwargs = plot
        plot = True

    if plot:
        plotter_cls = SeabornPlotter
    else:
        plotter_cls = NoPlotter
    if plot_kwargs is None:
        plot_kwargs = {}

    plotter = plotter_cls.new(**plot_kwargs)
    return plotter


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
    if not isinstance(col, pd.Series):
        # TODO: deal with a list/tuple or np array
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
        else:
            palette = None
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
        plt.tight_layout()
