from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import drnb.embed
import drnb.embed.base
from drnb.embed.context import EmbedContext
from drnb.log import log
from drnb.types import EmbedResult


def tsne_scale_coords(coords: np.ndarray, target_std: float = 1e-4) -> np.ndarray:
    """Rescale coordinates to a fixed standard deviation, t-SNE style."""

    # Copied from the openTSNE initialization module to avoid dependency
    x = np.array(coords, copy=True)
    x /= np.std(x[:, 0]) / target_std

    return x


def scale_coords(
    coords: NDArray[np.float32], max_coord: float = 10.0
) -> NDArray[np.float32]:
    """Ensure that the maximum absolute value of the coordinates is `max_coord`."""
    expansion = max_coord / np.abs(coords).max()
    return (coords * expansion).astype(np.float32)


def add_noise(
    coords: NDArray[np.float32], noise: float = 0.0001, seed: int | None = None
) -> NDArray[np.float32]:
    """Add Gaussian noise to the coordinates."""
    rng = np.random.default_rng(seed=seed)
    return coords + rng.normal(scale=noise, size=coords.shape).astype(np.float32)


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
