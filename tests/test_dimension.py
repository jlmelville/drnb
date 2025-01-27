import numpy as np
from sklearn.neighbors import NearestNeighbors

from drnb.dimension import mle_global, mle_local


def _make_test_data(
    n_dim: int, embed_dim: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate test data and compute nearest neighbor distances.

    Parameters
    ----------
    n_dim : int
        Intrinsic dimension of the data
    embed_dim : int, optional
        Dimension to embed the data in. If None, same as n_dim

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X: Generated points
        distances: Distances to k nearest neighbors
    """
    rng = np.random.default_rng(42)
    n_samples = 1000
    k = 10

    # Generate base data
    X = rng.uniform(0, 1, size=(n_samples, n_dim))

    # Embed in higher dimension if requested
    if embed_dim is not None:
        X = np.column_stack([X, np.zeros((n_samples, embed_dim - n_dim))])

    # Compute distances
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nbrs.kneighbors(X)

    return X, distances


def test_mle_local_uniform_cube():
    """Test MLE dimension estimation on points sampled from a unit cube.

    The intrinsic dimension should be approximately 3."""
    _, distances = _make_test_data(n_dim=3)
    dims = mle_local(distances, remove_self=True)
    np.testing.assert_allclose(np.mean(dims), 3.0, rtol=0.2)


def test_mle_local_uniform_line():
    """Test MLE dimension estimation on points sampled from a line.

    The intrinsic dimension should be approximately 1."""
    _, distances = _make_test_data(n_dim=1, embed_dim=3)
    dims = mle_local(distances, remove_self=True)
    np.testing.assert_allclose(np.mean(dims), 1.0, rtol=0.2)


def test_mle_global_uniform_cube():
    """Test global MLE dimension estimation on points sampled from a unit cube.

    The intrinsic dimension should be approximately 3."""
    _, distances = _make_test_data(n_dim=3)
    dim = mle_global(distances, remove_self=True)
    np.testing.assert_allclose(dim, 3.0, rtol=0.2)


def test_mle_global_uniform_line():
    """Test global MLE dimension estimation on points sampled from a line.

    The intrinsic dimension should be approximately 1."""
    _, distances = _make_test_data(n_dim=1, embed_dim=3)
    dim = mle_global(distances, remove_self=True)
    np.testing.assert_allclose(dim, 1.0, rtol=0.2)


def test_mle_global_matches_local_mean():
    """Test that global MLE is close to the harmonic mean of local MLEs."""
    _, distances = _make_test_data(n_dim=3)

    # Compute both estimates
    local_dims = mle_local(distances, remove_self=True)
    global_dim = mle_global(distances, remove_self=True)

    # Global should be close to harmonic mean of local
    n_samples = len(distances)
    harmonic_mean = n_samples / np.sum(1.0 / local_dims)
    np.testing.assert_allclose(global_dim, harmonic_mean, rtol=1e-10)
