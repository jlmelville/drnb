from abc import abstractmethod
from typing import Protocol

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from drnb.embed.context import EmbedContext
from drnb.types import EmbedResult


# pylint: disable=too-few-public-methods
class PlotterProtocol(Protocol):
    """A protocol for a plotting results of an embedding. Usually the result is
    a scatterplot of the embedded data, but it could be a histogram or other plot."""

    @abstractmethod
    def plot(
        self,
        embedding_result: EmbedResult,
        data: np.ndarray | None = None,
        y: pd.DataFrame | pd.Series | np.ndarray | range | None = None,
        ctx: EmbedContext | None = None,
        ax: Axes | None = None,
    ) -> Axes | None:
        """Plot the embedding result.

        Args:
            embedding_result: Result of the embedding
            data: Original input data
            target: Target variables/labels
            ctx: Optional embedding context
        """


class NoPlotter:
    """A do-nothing plotter object."""

    @classmethod
    def new(cls, **kwargs):
        """Create a new NoPlotter object."""
        return cls(**kwargs)

    def plot(
        self,
        _: tuple | dict | np.ndarray,
        __: np.ndarray | None = None,
        ___: pd.DataFrame | pd.Series | np.ndarray | range | None = None,
        ____: EmbedContext | None = None,
        _____: Axes | None = None,
    ) -> Axes | None:
        """Do nothing."""
