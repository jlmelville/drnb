import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.neighbors import NearestNeighbors

from drnb.neighbors.localscale import locally_scaled_neighbors


def test_locally_scaled_neighbors():
    """Test locally_scaled_neighbors with a small example."""
    # Create a simple 4x4 neighbor index and distance arrays
    # Each point has 4 neighbors (including self)
    idx = np.array(
        [
            [0, 1, 2, 3],  # neighbors of point 0
            [1, 0, 2, 3],  # neighbors of point 1
            [2, 0, 1, 3],  # neighbors of point 2
            [3, 0, 1, 2],  # neighbors of point 3
        ]
    )

    # Create corresponding distances (self-distance = 0)
    dist = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.0, 0.1, 0.2, 0.4],
            [0.0, 0.1, 0.3, 0.5],
            [0.0, 0.2, 0.3, 0.4],
        ]
    )

    # Get 2 locally scaled neighbors, considering all 4 neighbors
    # Use neighbors 1-2 (indices 1-2) for scaling
    scaled_idx, scaled_dist = locally_scaled_neighbors(
        idx, dist, l=2, m=None, scale_from=1, scale_to=2
    )

    # Test shapes
    assert scaled_idx.shape == (4, 2), "Wrong shape for scaled indices"
    assert scaled_dist.shape == (4, 2), "Wrong shape for scaled distances"

    # Test that output distances are sorted
    assert np.all(np.diff(scaled_dist, axis=1) >= 0), "Distances should be sorted"

    # Test error cases
    with pytest.raises(ValueError):
        # l cannot be greater than m
        locally_scaled_neighbors(idx, dist, l=3, m=2)

    with pytest.raises(ValueError):
        # m cannot be greater than k
        locally_scaled_neighbors(idx, dist, l=2, m=5)


def test_locally_scaled_neighbors_m_none():
    """Test that m=None uses all available neighbors."""
    n, k = 5, 4
    idx = (
        np.arange(n).reshape(-1, 1).repeat(k, axis=1)
    )  # each point connected to itself k times
    dist = np.ones((n, k))  # all distances = 1

    # Should work with m=None (uses k=4)
    scaled_idx, scaled_dist = locally_scaled_neighbors(
        idx, dist, l=2, m=None, scale_from=1, scale_to=2
    )

    assert scaled_idx.shape == (n, 2)
    assert scaled_dist.shape == (n, 2)


iris = load_iris()
X = iris.data

k = 66
nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")
nbrs.fit(X)
distances, indices = nbrs.kneighbors(X)


def test_iris():
    """Test locally_scaled_neighbors with the iris dataset. Based on validation via
    R code."""

    scaled_idx, scaled_dist = locally_scaled_neighbors(
        indices,
        distances,
        l=15,  # number of neighbors to keep
        m=None,  # use all available neighbors
        scale_from=4,  # use neighbors 5-7 for scaling
        scale_to=6,
    )
    assert scaled_idx.shape == (150, 15)
    assert scaled_dist.shape == (150, 15)
    assert sorted(indices[-1, :15].tolist()) == [
        63,
        70,
        83,
        101,
        111,
        113,
        121,
        123,
        126,
        127,
        133,
        138,
        142,
        147,
        149,
    ]

    # due to tied distances, it's really hard to compare with other methods output,
    # so we just check that the indices are the same after sorting
    assert sorted(scaled_idx[-1, :15].tolist()) == [
        63,
        70,
        83,
        101,
        113,
        114,
        121,
        123,
        126,
        127,
        133,
        134,
        138,
        142,
        149,
    ]

    assert np.allclose(
        scaled_dist[-1, :15],
        [
            0.0,
            0.2828424,
            0.3162275,
            0.3316624,
            0.3316624,
            0.3605549,
            0.3741656,
            0.4582576,
            0.4690412,
            0.5385165,
            0.5385166,
            0.5830953,
            0.6082762,
            0.6403125,
            0.7810249,
        ],
        atol=1e-6,
    )


def test_iris_no_self():
    # check that we get the same result if we don't include the self-distance
    # and correct the other parameters appropriately
    scaled_idx1, scaled_dist1 = locally_scaled_neighbors(
        indices[:, 1:],
        distances[:, 1:],
        l=14,
        m=65,
        scale_from=3,
        scale_to=5,
    )
    assert sorted(scaled_idx1[-1, :].tolist()) == [
        63,
        70,
        83,
        101,
        113,
        114,
        121,
        123,
        126,
        127,
        133,
        134,
        138,
        142,
    ]

    assert np.allclose(
        scaled_dist1[-1, :],
        [
            0.2828424,
            0.3162275,
            0.3316624,
            0.3316624,
            0.3605549,
            0.3741656,
            0.4582576,
            0.4690412,
            0.5385165,
            0.5385166,
            0.5830953,
            0.6082762,
            0.6403125,
            0.7810249,
        ],
        atol=1e-6,
    )
