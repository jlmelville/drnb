from dataclasses import dataclass

import numpy as np
import sklearn.manifold

import drnb.embed
import drnb.embed.base
from drnb.embed.context import EmbedContext
from drnb.log import log
from drnb.types import EmbedResult


@dataclass
class BaseMds(drnb.embed.base.Embedder):
    """Base class for MDS embedding implementations using sklearn."""

    precomputed_init: np.ndarray | None = None
    max_iter: int = 300

    def embed_impl(
        self, x: np.ndarray, params: dict, ctx: EmbedContext | None = None
    ) -> EmbedResult:
        fit_transform_params = {}
        if self.precomputed_init is not None:
            log.info("Using precomputed init")
            fit_transform_params["init"] = self.precomputed_init

        return drnb.embed.fit_transform_embed(
            x,
            params,
            sklearn.manifold.MDS,
            "sklearn-MDS",
            n_components=2,
            metric=self._get_metric(),
            n_init=1,
            normalized_stress=False,
            max_iter=self.max_iter,
            fit_transform_params=fit_transform_params,
        )

    def _get_metric(self) -> bool:
        """Return whether to use metric MDS. To be implemented by subclasses."""
        raise NotImplementedError


@dataclass
class Mmds(BaseMds):
    """Metric Multidimensional Scaling (MMDS) embedding using sklearn."""

    def _get_metric(self) -> bool:
        return True


@dataclass
class Nmds(BaseMds):
    """Non-metric Multidimensional Scaling (NMDS) embedding using sklearn."""

    def _get_metric(self) -> bool:
        return False
