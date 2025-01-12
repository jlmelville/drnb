import abc
from dataclasses import dataclass, field

import numpy as np

from drnb.embed.context import EmbedContext
from drnb.types import EmbedResult


@dataclass
class Embedder(abc.ABC):
    """Base class for embedding algorithms.

    Attributes:
        params: Parameters for the embedding algorithm.
    """

    params: dict = field(default_factory=dict)

    def embed(self, x: np.ndarray, ctx: EmbedContext | None = None) -> EmbedResult:
        """Embed the data."""
        params = dict(self.params)
        return self.embed_impl(x, params, ctx)

    @abc.abstractmethod
    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        """Embed the data."""
