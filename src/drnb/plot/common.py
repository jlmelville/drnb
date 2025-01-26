from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from drnb.embed import get_coords, set_coords
from drnb.plot.align import kabsch_best_align
from drnb.plot.plotly import plotly_result_plot
from drnb.plot.seaborn import sns_result_plot


def result_plot(
    embed_result: dict,
    plot_type: Literal["sns", "plotly"] = "sns",
    *,
    fixed: np.ndarray | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes | None:
    """Plot the result of an embedding using the specified plot type.

    Args:
        embed_result: The embedding result to plot.
        plot_type: Type of plot to create ("sns" or "plotly").
        fixed: Optional reference coordinates for Kabsch alignment.
        ax: Optional matplotlib axes to plot on.
        **kwargs: Additional arguments passed to the underlying plot function.

    Returns:
        The matplotlib axes if using sns plot type, None for plotly.
    """
    if fixed is not None:
        coords = get_coords(embed_result)
        align_coords = kabsch_best_align(fixed, coords)
        embed_result = set_coords(embed_result, align_coords)

    if plot_type == "sns":
        sns_result_plot(embed_result, ax=ax, **kwargs)
    elif plot_type == "plotly":
        plotly_result_plot(embed_result, **kwargs)
    else:
        raise ValueError(f"Unknown plot_type {plot_type}")

    if fixed is not None:
        embed_result = set_coords(embed_result, coords)

    return ax if plot_type == "sns" else None
