from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from drnb.types import EmbedResult


def plot_color(
    embed_result: EmbedResult | dict[str, Any],
    values: np.ndarray | Sequence[float],
    *,
    backend: str = "seaborn",
    **kwargs,
):
    """Dispatch to a backend-specific color-by helper."""
    backend_norm = backend.lower()
    if backend_norm == "seaborn":
        from drnb.plot.seaborn import plot_color as sns_plot_color

        return sns_plot_color(embed_result, values, **kwargs)
    if backend_norm == "plotly":
        from drnb.plot.plotly import plot_color as plotly_plot_color

        return plotly_plot_color(embed_result, values, **kwargs)
    raise ValueError(f"Unknown plot backend '{backend}'")
