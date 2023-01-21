from numpy.random import Generator, default_rng


def gaussnd(n, ndim, seed=42):
    """Create n points from an ndim normal distribution"""
    if isinstance(seed, Generator):
        rng = seed
    else:
        rng = default_rng(seed=seed)
    return rng.standard_normal((n, ndim))
