import numpy as np
import openTSNE.initialization

from drnb.log import log


def scale_coords(coords, max_coord=10.0):
    expansion = max_coord / np.abs(coords).max()
    return (coords * expansion).astype(np.float32)


def noisy_scale_coords(coords, max_coord=10.0, noise=0.0001):
    return scale_coords(coords, max_coord=max_coord) + add_noise(coords, noise=noise)


def add_noise(coords, noise=0.0001, seed=None):
    rng = np.random.default_rng(seed=seed)
    return coords + rng.normal(scale=noise, size=coords.shape).astype(np.float32)


def spca(data):
    log.info("Initializing via openTSNE (scaled) PCA")
    return openTSNE.initialization.pca(data)
