from dataclasses import dataclass

import numpy as np

import drnb.embed
import drnb.embed.base
from drnb.embed.context import EmbedContext
from drnb.log import log
from drnb.types import EmbedResult
from drnb.yinit import add_noise, scale_coords, tsne_scale_coords


@dataclass
class Rescale(drnb.embed.base.Embedder):
    """Rescale coordinates to a fixed range. Exists purely to rescale the output of one
    embedder if using it as initialization into another in an embedding pipeline"""

    precomputed_init: np.ndarray = None
    max_coord: float = 10.0
    noise: float = 0.0

    def embed_impl(
        self, x: np.ndarray, params: dict, _: EmbedContext | None = None
    ) -> EmbedResult:
        max_coord = params.get("max_coord", self.max_coord)
        if self.precomputed_init is not None:
            coords = self.precomputed_init
        else:
            raise ValueError("No coordinates provided")

        log.info("Rescaling to max coordinate %f", max_coord)
        result = scale_coords(coords, max_coord=max_coord)
        if self.noise > 0.0:
            result = add_noise(result, noise=self.noise)
        return result


@dataclass
class TsneRescale(drnb.embed.base.Embedder):
    """Rescale coordinates to a fixed standard deviation, t-SNE style."""

    precomputed_init: np.ndarray = None
    target_std: float = 1e-4
    noise: float = 0.0

    def embed_impl(
        self, x: np.ndarray, params: dict, _: EmbedContext | None = None
    ) -> EmbedResult:
        target_std = params.get("target_std", self.target_std)
        if self.precomputed_init is not None:
            coords = self.precomputed_init
        else:
            raise ValueError("No coordinates provided")

        log.info("Rescaling to target standard deviation %f", target_std)
        result = tsne_scale_coords(coords, target_std=target_std)
        if self.noise > 0.0:
            result = add_noise(result, noise=self.noise)
        return result
