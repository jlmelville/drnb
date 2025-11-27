from __future__ import annotations

from typing import Callable

from drnb.plot.protocol import PlotterProtocol
from drnb.util import get_method_and_args


def create_plotters(
    plot: bool | dict | str = True, plot_kwargs: dict | None = None
) -> list[PlotterProtocol]:
    """Create a list of plotters based on the plot and plot_kwargs arguments."""

    def _load_seaborn_plotter():
        from drnb.plot.seaborn import SeabornPlotter

        return SeabornPlotter

    def _load_plotly_plotter():
        from drnb.plot.plotly import PlotlyPlotter

        return PlotlyPlotter

    plotter_cls_factory: Callable[[], type] = _load_seaborn_plotter

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
            plotter_cls_factory = _load_seaborn_plotter
        elif plot == "plotly":
            plotter_cls_factory = _load_plotly_plotter
        else:
            raise ValueError(f"Unknown plot type {plot}")
        plot = True
    if not plot:
        return []

    plotters = []
    plotter_cls = plotter_cls_factory()

    color_by = []
    if "color_by" in plot_kwargs:
        color_by = plot_kwargs["color_by"]
        if not isinstance(color_by, (list, tuple)):
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
            from drnb.plot.nphistogram import NbrPreservationHistogram

            plotters.append(NbrPreservationHistogram(**extra_kwds))
        elif extra == "rthist":
            from drnb.plot.rthistogram import RandomTripletHistogram

            plotters.append(RandomTripletHistogram(**extra_kwds))
        elif extra == "rpscatter":
            from drnb.plot.rpdscatterplot import RandomPairDistanceScatterplot

            plotters.append(RandomPairDistanceScatterplot(**extra_kwds))
        else:
            raise ValueError(f"Unknown plot type '{extra}'")
    return plotters
