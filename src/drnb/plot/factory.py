from drnb.plot.nphistogram import NbrPreservationHistogram
from drnb.plot.plotly import PlotlyPlotter
from drnb.plot.protocol import PlotterProtocol
from drnb.plot.rpdscatterplot import RandomPairDistanceScatterplot
from drnb.plot.rthistogram import RandomTripletHistogram
from drnb.plot.seaborn import SeabornPlotter
from drnb.util import get_method_and_args


def create_plotters(
    plot: bool | dict | str = True, plot_kwargs: dict | None = None
) -> list[PlotterProtocol]:
    """Create a list of plotters based on the plot and plot_kwargs arguments."""
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
            plotters.append(NbrPreservationHistogram(**extra_kwds))
        elif extra == "rthist":
            plotters.append(RandomTripletHistogram(**extra_kwds))
        elif extra == "rpscatter":
            plotters.append(RandomPairDistanceScatterplot(**extra_kwds))
        else:
            raise ValueError(f"Unknown plot type '{extra}'")
    return plotters
