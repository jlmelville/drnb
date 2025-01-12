import numpy as np
from numpy.random import Generator, default_rng


def gaussnd(n: int, ndim: int, sdev: float = 1.0, seed: int = 42) -> np.ndarray:
    """Create n points from an ndim normal distribution"""
    if isinstance(seed, Generator):
        rng = seed
    else:
        rng = default_rng(seed=seed)
    return rng.standard_normal((n, ndim)) * sdev
