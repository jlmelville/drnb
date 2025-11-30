import numpy as np


# See also:
# https://github.com/YingfanWang/PaCMAP/blob/master/demo/specify_nn_demo.py
def create_neighbor_pairs(nbr_idx: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Create pairs of neighbor indices from k-nearest neighbor indices."""
    if n_neighbors > nbr_idx.shape[1]:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) must be <= number of columns in "
            f"nbr_idx ({nbr_idx.shape[1]})"
        )

    n = len(nbr_idx)
    pairs = np.zeros((n * n_neighbors, 2), dtype=np.int32)
    for i in range(n):
        for j in range(n_neighbors):
            pairs[i * n_neighbors + j] = [i, nbr_idx[i, j + 1]]
    return pairs
