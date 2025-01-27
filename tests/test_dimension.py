import numpy as np
from sklearn.neighbors import NearestNeighbors

from drnb.dimension import mle_local


def test_mle_local_uniform_cube():
    """Test MLE dimension estimation on points sampled from a unit cube.

    The intrinsic dimension should be approximately 3."""
    # Set random seed for reproducibility
    rng = np.random.default_rng(42)

    # Generate 1000 points uniformly from a 3D cube
    n_samples = 1000
    X = rng.uniform(0, 1, size=(n_samples, 3))

    # Get 10 nearest neighbors
    k = 10
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)  # k+1 to include self
    distances, _ = nbrs.kneighbors(X)

    # Compute local dimensions
    dims = mle_local(distances, remove_self=True)

    # Check mean dimension is close to 3
    mean_dim = np.mean(dims)
    np.testing.assert_allclose(mean_dim, 3.0, rtol=0.2)


def test_mle_local_uniform_line():
    """Test MLE dimension estimation on points sampled from a line.

    The intrinsic dimension should be approximately 1."""
    rng = np.random.default_rng(42)

    # Generate 1000 points uniformly from a line
    n_samples = 1000
    X = rng.uniform(0, 1, size=(n_samples, 1))
    # Embed in 3D space
    X = np.column_stack([X, np.zeros((n_samples, 2))])

    k = 10
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nbrs.kneighbors(X)

    dims = mle_local(distances, remove_self=True)
    mean_dim = np.mean(dims)
    np.testing.assert_allclose(mean_dim, 1.0, rtol=0.2)
