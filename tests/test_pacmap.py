import numpy as np
import pytest

from drnb.embed.pacmap import create_neighbor_pairs


def test_create_neighbor_pairs():
    """Test create_neighbor_pairs function with a small example."""
    # Create a simple 4x3 neighbor index array (4 points, 3 neighbors each including self)
    nbr_idx = np.array(
        [
            [0, 1, 2],  # neighbors of point 0: self, 1, 2
            [1, 0, 3],  # neighbors of point 1: self, 0, 3
            [2, 0, 3],  # neighbors of point 2: self, 0, 3
            [3, 1, 2],  # neighbors of point 3: self, 1, 2
        ]
    )

    # Create pairs using 2 neighbors (excluding self)
    pairs = create_neighbor_pairs(nbr_idx, n_neighbors=2)

    # Expected pairs (each point paired with its non-self neighbors)
    expected_pairs = np.array(
        [
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 3],
            [2, 0],
            [2, 3],
            [3, 1],
            [3, 2],
        ]
    )

    # Test shape and content
    assert pairs.shape == (8, 2), "Incorrect shape of output pairs"
    assert np.array_equal(pairs, expected_pairs), "Incorrect pair content"

    # Verify no self-pairs exist in output
    assert not np.any(pairs[:, 0] == pairs[:, 1]), "Self-pairs found in output"


def test_create_neighbor_pairs_invalid_input():
    """Test create_neighbor_pairs with invalid inputs."""
    # Test with n_neighbors larger than input width
    nbr_idx = np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        create_neighbor_pairs(nbr_idx, n_neighbors=3)
